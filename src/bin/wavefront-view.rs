//! `wavefront-view` -- a minimal post-processing tool that turns snapshots
//! out of a `wave_trajectory.bin` file into viewable images.
//!
//! `wave_trajectory.bin` has no header (see `src/engine.rs::serialize_snapshot`
//! and `src/layout.rs::FieldBlock`'s doc comment for the exact on-disk
//! layout), so this tool needs the same `--nx`/`--ny`/`--nz` you ran the
//! simulation with to make sense of the raw bytes.
//!
//! Two render modes (`--mode`):
//! - `slice` (default): one 2D cross-section, holding one axis fixed.
//! - `volume`: a maximum-intensity projection through the *entire* domain
//!   along the chosen axis -- at each pixel, the voxel with the largest
//!   `|value|` anywhere along that ray, keeping its sign. This is the
//!   standard, simplest form of volumetric rendering (used throughout
//!   medical/scientific visualization for exactly this reason: no lighting
//!   or opacity model to get wrong, and every feature in the volume shows up
//!   somewhere in the projection).
//!
//! Two output shapes:
//! - A single `--snapshot <N>` renders one binary PPM (`.ppm`, "P6") image --
//!   deliberately dependency-free, since most image viewers, GIMP, and
//!   ImageMagick's `convert`/`magick` all read it directly.
//! - A `--snapshots <A>:<B>` range renders every snapshot in that (inclusive)
//!   range into one animated GIF (`.gif`, GIF89a) instead -- also
//!   hand-written, for the same reason the PPM writer is: this crate keeps
//!   its dependency set small and pinned (see `Cargo.toml`), and GIF's LZW
//!   compression, while more involved than PPM's raw bytes, is still a
//!   fully-specified, implementable-by-hand format. Encoded here as *real*
//!   LZW (an adaptive dictionary, not a fixed-width placeholder), since
//!   that's the same standard this crate holds its other from-scratch
//!   implementations (`io_uring` snapshot I/O, the SIMD Yee kernels) to.
//!
//! Colors are normalized against the *global* max `|value|` across every
//! frame being rendered (a single snapshot's own max, in the single-frame
//! case -- so this is not a behavior change for existing single-image use),
//! so brightness is comparable across an animation's frames instead of each
//! frame separately auto-contrasting itself.
//!
//! The trajectory file is memory-mapped (via `memmap2`, already a
//! dependency) rather than read into a `Vec` -- consistent with the rest of
//! the crate's zero-copy philosophy, and it means only the handful of pages
//! this tool actually touches ever get faulted in, regardless of how large
//! the trajectory file is.

use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::process::ExitCode;
use wavefront::layout::{FieldBlock, GridDims, BLOCK_DIM, VOXELS_PER_BLOCK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy)]
enum Component {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    /// Sum of squares of all 6 components -- always non-negative and
    /// nonzero wherever the wave has reached, so it's a good default that
    /// needs no sign-aware colormap.
    Energy,
}

impl Component {
    /// Index of this component's `[f32; VOXELS_PER_BLOCK]` array within
    /// `FieldBlock`, matching its `repr(C)` declaration order. `None` for
    /// `Energy`, which reads all six.
    fn field_index(self) -> Option<usize> {
        match self {
            Component::Ex => Some(0),
            Component::Ey => Some(1),
            Component::Ez => Some(2),
            Component::Hx => Some(3),
            Component::Hy => Some(4),
            Component::Hz => Some(5),
            Component::Energy => None,
        }
    }

    fn is_signed(self) -> bool {
        !matches!(self, Component::Energy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Slice,
    Volume,
}

#[derive(Debug, Clone, Copy)]
enum SnapshotSpec {
    Single(usize),
    Range(usize, usize),
}

struct Config {
    input: PathBuf,
    nx: usize,
    ny: usize,
    nz: usize,
    snapshots: SnapshotSpec,
    mode: Mode,
    axis: Axis,
    slice: Option<usize>,
    component: Component,
    output: PathBuf,
    fps: u32,
}

fn print_usage() {
    eprintln!(
        "wavefront-view -- render wave_trajectory.bin snapshot(s) as PPM or animated GIF\n\n\
         USAGE:\n    wavefront-view --input <PATH> --nx <N> --ny <N> --nz <N> [OPTIONS]\n\n\
         REQUIRED:\n\
         \x20   --input <PATH>       wave_trajectory.bin (or equivalent) to read\n\
         \x20   --nx/--ny/--nz <N>   grid dimensions the simulation was run with\n\n\
         OPTIONS:\n\
         \x20   --snapshot <N>       which snapshot to render, 0-indexed [default: 0]\n\
         \x20   --snapshots <A>:<B>  render an inclusive range of snapshots as one\n\
         \x20                        animated GIF instead of a single PPM -- requires\n\
         \x20                        --output to end in .gif; mutually exclusive with\n\
         \x20                        --snapshot\n\
         \x20   --fps <N>            animation playback rate for --snapshots [default: 10]\n\
         \x20   --mode <M>           slice (one 2D cross-section) or volume (maximum-\n\
         \x20                        intensity projection through the whole domain along\n\
         \x20                        --axis; --slice is ignored in this mode) [default: slice]\n\
         \x20   --axis <x|y|z>       which axis to hold fixed (slice) or project along\n\
         \x20                        (volume) [default: z]\n\
         \x20   --slice <N>          index along --axis to slice at (slice mode only)\n\
         \x20                        [default: middle]\n\
         \x20   --component <C>      ex, ey, ez, hx, hy, hz, or energy [default: energy]\n\
         \x20   --output <PATH>      output path [default: slice.ppm]\n\
         \x20   -h, --help           print this message"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut input: Option<PathBuf> = None;
    let mut nx: Option<usize> = None;
    let mut ny: Option<usize> = None;
    let mut nz: Option<usize> = None;
    let mut snapshot: Option<usize> = None;
    let mut snapshots_range: Option<(usize, usize)> = None;
    let mut mode = Mode::Slice;
    let mut axis = Axis::Z;
    let mut slice: Option<usize> = None;
    let mut component = Component::Energy;
    let mut output = PathBuf::from("slice.ppm");
    let mut fps = 10u32;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next_value = |name: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        let parse_num = |name: &str, s: String| -> Result<usize, String> {
            s.parse::<usize>()
                .map_err(|_| format!("invalid integer for {name}: {s}"))
        };

        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(next_value("--input")?)),
            "--nx" => nx = Some(parse_num("--nx", next_value("--nx")?)?),
            "--ny" => ny = Some(parse_num("--ny", next_value("--ny")?)?),
            "--nz" => nz = Some(parse_num("--nz", next_value("--nz")?)?),
            "--snapshot" => snapshot = Some(parse_num("--snapshot", next_value("--snapshot")?)?),
            "--snapshots" => {
                let raw = next_value("--snapshots")?;
                let (a, b) = raw.split_once(':').ok_or_else(|| {
                    format!("invalid --snapshots {raw:?}: expected <A>:<B>, e.g. 0:20")
                })?;
                let a = parse_num("--snapshots", a.to_string())?;
                let b = parse_num("--snapshots", b.to_string())?;
                if a > b {
                    return Err(format!(
                        "invalid --snapshots {raw:?}: start ({a}) must be <= end ({b})"
                    ));
                }
                snapshots_range = Some((a, b));
            }
            "--fps" => {
                fps = next_value("--fps")?
                    .parse::<u32>()
                    .map_err(|_| "invalid --fps: expected a positive integer".to_string())?;
                if fps == 0 {
                    return Err("invalid --fps: must be at least 1".to_string());
                }
            }
            "--mode" => {
                mode = match next_value("--mode")?.as_str() {
                    "slice" => Mode::Slice,
                    "volume" => Mode::Volume,
                    other => {
                        return Err(format!("invalid --mode: {other} (expected slice or volume)"))
                    }
                }
            }
            "--axis" => {
                axis = match next_value("--axis")?.as_str() {
                    "x" => Axis::X,
                    "y" => Axis::Y,
                    "z" => Axis::Z,
                    other => return Err(format!("invalid --axis: {other} (expected x, y, or z)")),
                }
            }
            "--slice" => slice = Some(parse_num("--slice", next_value("--slice")?)?),
            "--component" => {
                component = match next_value("--component")?.as_str() {
                    "ex" => Component::Ex,
                    "ey" => Component::Ey,
                    "ez" => Component::Ez,
                    "hx" => Component::Hx,
                    "hy" => Component::Hy,
                    "hz" => Component::Hz,
                    "energy" => Component::Energy,
                    other => {
                        return Err(format!(
                            "invalid --component: {other} (expected ex, ey, ez, hx, hy, hz, or energy)"
                        ))
                    }
                }
            }
            "--output" => output = PathBuf::from(next_value("--output")?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    let snapshots = match (snapshot, snapshots_range) {
        (Some(_), Some(_)) => {
            return Err("--snapshot and --snapshots are mutually exclusive".to_string())
        }
        (Some(n), None) => SnapshotSpec::Single(n),
        (None, Some((a, b))) => {
            let is_gif = output
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gif"));
            if !is_gif {
                return Err(format!(
                    "--snapshots requires --output to end in .gif (got {output:?})"
                ));
            }
            SnapshotSpec::Range(a, b)
        }
        (None, None) => SnapshotSpec::Single(0),
    };

    Ok(Config {
        input: input.ok_or("--input is required")?,
        nx: nx.ok_or("--nx is required")?,
        ny: ny.ok_or("--ny is required")?,
        nz: nz.ok_or("--nz is required")?,
        snapshots,
        mode,
        axis,
        slice,
        component,
        output,
        fps,
    })
}

/// Reads one field component at voxel `(x, y, z)` out of `snapshot_bytes`
/// (the byte slice for exactly one snapshot's worth of blocks), given the
/// grid's block dimensions.
fn read_component(
    snapshot_bytes: &[u8],
    bx_n: usize,
    by_n: usize,
    x: usize,
    y: usize,
    z: usize,
    field_index: usize,
) -> f32 {
    let (bx, by, bz) = (x / BLOCK_DIM, y / BLOCK_DIM, z / BLOCK_DIM);
    let (lx, ly, lz) = (x % BLOCK_DIM, y % BLOCK_DIM, z % BLOCK_DIM);
    let block_index = (bz * by_n + by) * bx_n + bx;
    let local = FieldBlock::local_index(lx, ly, lz);

    let block_bytes = std::mem::size_of::<FieldBlock>();
    let component_bytes = VOXELS_PER_BLOCK * std::mem::size_of::<f32>();
    let offset =
        block_index * block_bytes + field_index * component_bytes + local * std::mem::size_of::<f32>();

    let bytes: [u8; 4] = snapshot_bytes[offset..offset + 4]
        .try_into()
        .expect("computed offset is always in bounds for a validated snapshot");
    f32::from_le_bytes(bytes)
}

/// Samples one component at one voxel. Signed components return their
/// signed value; `Energy` returns the (always non-negative) sum of squares
/// of all six.
fn sample(snapshot_bytes: &[u8], bx_n: usize, by_n: usize, x: usize, y: usize, z: usize, component: Component) -> f32 {
    match component.field_index() {
        Some(idx) => read_component(snapshot_bytes, bx_n, by_n, x, y, z, idx),
        None => (0..6)
            .map(|idx| {
                let v = read_component(snapshot_bytes, bx_n, by_n, x, y, z, idx);
                v * v
            })
            .sum(),
    }
}

/// Computes one frame's `width x height` grid of raw (uncolored) values,
/// row-major, for either render mode.
///
/// - `Mode::Slice`: samples the single plane at `slice_index` along `axis`.
/// - `Mode::Volume`: for each pixel, scans the *entire* extent along `axis`
///   and keeps the sample with the largest `|value|` (a maximum-intensity
///   projection) -- the whole domain collapsed onto one 2D image.
fn compute_values(
    snapshot_bytes: &[u8],
    dims: &GridDims,
    mode: Mode,
    axis: Axis,
    slice_index: usize,
    component: Component,
) -> (usize, usize, Vec<f32>) {
    let (bx_n, by_n, _) = dims.block_dims();
    let (width, height, extent_along_axis) = match axis {
        Axis::X => (dims.ny, dims.nz, dims.nx),
        Axis::Y => (dims.nx, dims.nz, dims.ny),
        Axis::Z => (dims.nx, dims.ny, dims.nz),
    };

    let mut values = vec![0.0f32; width * height];
    for v in 0..height {
        for u in 0..width {
            let value = match mode {
                Mode::Slice => {
                    let (x, y, z) = match axis {
                        Axis::X => (slice_index, u, v),
                        Axis::Y => (u, slice_index, v),
                        Axis::Z => (u, v, slice_index),
                    };
                    sample(snapshot_bytes, bx_n, by_n, x, y, z, component)
                }
                Mode::Volume => {
                    let mut best = 0.0f32;
                    for d in 0..extent_along_axis {
                        let (x, y, z) = match axis {
                            Axis::X => (d, u, v),
                            Axis::Y => (u, d, v),
                            Axis::Z => (u, v, d),
                        };
                        let s = sample(snapshot_bytes, bx_n, by_n, x, y, z, component);
                        if s.abs() > best.abs() {
                            best = s;
                        }
                    }
                    best
                }
            };
            values[v * width + u] = value;
        }
    }
    (width, height, values)
}

/// Maps a normalized value `t` (in `[-1, 1]` for signed components, `[0, 1]`
/// for energy) to an RGB pixel: white at zero, blending to blue for
/// negative and red for positive (or just white-to-red for energy, since
/// it's never negative).
fn colormap(t: f32, signed: bool) -> [u8; 3] {
    let lerp = |a: u8, b: u8, f: f32| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
    const WHITE: [u8; 3] = [255, 255, 255];
    const RED: [u8; 3] = [214, 39, 40];
    const BLUE: [u8; 3] = [31, 119, 180];

    let (from, to, f) = if signed && t < 0.0 {
        (WHITE, BLUE, (-t).clamp(0.0, 1.0))
    } else {
        (WHITE, RED, t.clamp(0.0, 1.0))
    };

    [
        lerp(from[0], to[0], f),
        lerp(from[1], to[1], f),
        lerp(from[2], to[2], f),
    ]
}

fn write_ppm(path: &std::path::Path, width: usize, height: usize, pixels: &[[u8; 3]]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(File::create(path)?);
    write!(file, "P6\n{width} {height}\n255\n")?;
    for pixel in pixels {
        file.write_all(pixel)?;
    }
    Ok(())
}

/// A minimal, dependency-free GIF89a encoder: one global 256-color palette
/// (built from `colormap`, so animations use exactly the same colors as a
/// single-frame PPM would), a `NETSCAPE2.0` looping extension, and real
/// (adaptive-dictionary) LZW-compressed image data per frame -- not a
/// fixed-width placeholder encoding, since this crate holds its other
/// from-scratch formats (the PPM writer, the on-disk snapshot layout) to
/// the same "actually implement the real thing" standard.
mod gif {
    use std::collections::HashMap;
    use std::io::{self, Write};

    const MIN_CODE_SIZE: u8 = 8; // 256-color palette
    const CLEAR_CODE: u16 = 1 << MIN_CODE_SIZE; // 256
    const END_CODE: u16 = CLEAR_CODE + 1; // 257
    const FIRST_FREE_CODE: u16 = CLEAR_CODE + 2; // 258
    const MAX_CODE: u16 = 4095; // 12-bit code space

    /// Packs a stream of variable-width LZW codes into GIF's sub-block
    /// format (LSB-first bit packing, 255-byte sub-blocks, zero-length
    /// block terminator).
    struct BitWriter {
        bytes: Vec<u8>,
        bit_buf: u32,
        bit_count: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter { bytes: Vec::new(), bit_buf: 0, bit_count: 0 }
        }

        fn write_code(&mut self, code: u16, width: u8) {
            self.bit_buf |= (code as u32) << self.bit_count;
            self.bit_count += width as u32;
            while self.bit_count >= 8 {
                self.bytes.push((self.bit_buf & 0xFF) as u8);
                self.bit_buf >>= 8;
                self.bit_count -= 8;
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bit_count > 0 {
                self.bytes.push((self.bit_buf & 0xFF) as u8);
            }
            self.bytes
        }
    }

    /// LZW-compresses one frame's palette-index pixels into GIF sub-blocks,
    /// including the leading minimum-code-size byte and trailing
    /// zero-length terminator.
    fn lzw_encode(indices: &[u8]) -> Vec<u8> {
        let mut writer = BitWriter::new();
        let mut code_width = MIN_CODE_SIZE + 1;
        let mut next_code = FIRST_FREE_CODE;
        // Dictionary: (prefix code, next byte) -> code. Reset alongside a
        // Clear Code whenever the 12-bit code space fills up.
        let mut dict: HashMap<(u16, u8), u16> = HashMap::new();

        writer.write_code(CLEAR_CODE, code_width);

        let mut iter = indices.iter().copied();
        let Some(first) = iter.next() else {
            writer.write_code(END_CODE, code_width);
            let mut out = vec![MIN_CODE_SIZE];
            pack_subblocks(&mut out, &writer.finish());
            return out;
        };

        let mut prefix = first as u16; // single-byte codes equal their byte value
        for byte in iter {
            match dict.get(&(prefix, byte)) {
                Some(&code) => prefix = code,
                None => {
                    writer.write_code(prefix, code_width);
                    dict.insert((prefix, byte), next_code);
                    next_code += 1;
                    if next_code > MAX_CODE {
                        writer.write_code(CLEAR_CODE, code_width);
                        dict.clear();
                        next_code = FIRST_FREE_CODE;
                        code_width = MIN_CODE_SIZE + 1;
                    } else if next_code > (1 << code_width) {
                        code_width += 1;
                    }
                    prefix = byte as u16;
                }
            }
        }
        writer.write_code(prefix, code_width);
        writer.write_code(END_CODE, code_width);

        let mut out = vec![MIN_CODE_SIZE];
        pack_subblocks(&mut out, &writer.finish());
        out
    }

    fn pack_subblocks(out: &mut Vec<u8>, data: &[u8]) {
        for chunk in data.chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0); // block terminator
    }

    /// Writes a complete animated GIF: one global palette, looped playback,
    /// one Graphic Control Extension + Image Descriptor + LZW data block
    /// per frame.
    pub fn write_animated_gif(
        path: &std::path::Path,
        width: usize,
        height: usize,
        palette: &[[u8; 3]; 256],
        frames: &[Vec<u8>],
        delay_centiseconds: u16,
    ) -> io::Result<()> {
        let mut file = io::BufWriter::new(std::fs::File::create(path)?);

        file.write_all(b"GIF89a")?;
        file.write_all(&(width as u16).to_le_bytes())?;
        file.write_all(&(height as u16).to_le_bytes())?;
        // Packed byte: global color table present, color resolution 7,
        // not sorted, global color table size = 2^(7+1) = 256.
        file.write_all(&[0b1111_0111, 0, 0])?; // background index 0, no aspect ratio

        for [r, g, b] in palette {
            file.write_all(&[*r, *g, *b])?;
        }

        // NETSCAPE2.0 application extension: loop forever.
        file.write_all(&[0x21, 0xFF, 11])?;
        file.write_all(b"NETSCAPE2.0")?;
        file.write_all(&[3, 1, 0, 0, 0])?;

        for frame in frames {
            // Graphic Control Extension: no transparency, given delay.
            file.write_all(&[0x21, 0xF9, 4, 0b0000_0000])?;
            file.write_all(&delay_centiseconds.to_le_bytes())?;
            file.write_all(&[0, 0])?;

            // Image Descriptor: full-frame, no local color table.
            file.write_all(&[0x2C])?;
            file.write_all(&0u16.to_le_bytes())?; // left
            file.write_all(&0u16.to_le_bytes())?; // top
            file.write_all(&(width as u16).to_le_bytes())?;
            file.write_all(&(height as u16).to_le_bytes())?;
            file.write_all(&[0])?; // no local color table, no interlace

            file.write_all(&lzw_encode(frame))?;
        }

        file.write_all(&[0x3B])?; // trailer
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A minimal LZW *decoder* matching the GIF89a algorithm, written
        /// independently of `lzw_encode`'s implementation -- so a bug shared
        /// between encoder and decoder can't hide a real mismatch. Used only
        /// to round-trip verify the encoder's output decodes back to the
        /// original indices, the way a real GIF reader would.
        fn lzw_decode(data: &[u8]) -> Vec<u8> {
            assert_eq!(data[0], MIN_CODE_SIZE, "unexpected minimum code size");

            let mut bitstream = Vec::new();
            let mut i = 1;
            loop {
                let len = data[i] as usize;
                i += 1;
                if len == 0 {
                    break;
                }
                bitstream.extend_from_slice(&data[i..i + len]);
                i += len;
            }

            let mut bit_pos = 0usize;
            let mut read_code = |width: u8| -> u16 {
                let mut code = 0u32;
                for b in 0..width as usize {
                    let byte = bitstream[(bit_pos + b) / 8];
                    let bit = (byte >> ((bit_pos + b) % 8)) & 1;
                    code |= (bit as u32) << b;
                }
                bit_pos += width as usize;
                code as u16
            };

            let mut code_width = MIN_CODE_SIZE + 1;
            // Codes 0..256 are literal single bytes; 256/257 are Clear/End
            // placeholders (never looked up); dynamic entries start at 258.
            let base_dict: Vec<Vec<u8>> = (0..CLEAR_CODE)
                .map(|b| vec![b as u8])
                .chain([vec![], vec![]])
                .collect();
            let mut dict = base_dict.clone();
            let mut out = Vec::new();
            let mut prev: Option<Vec<u8>> = None;

            loop {
                let code = read_code(code_width);
                if code == CLEAR_CODE {
                    dict = base_dict.clone();
                    code_width = MIN_CODE_SIZE + 1;
                    prev = None;
                    continue;
                }
                if code == END_CODE {
                    break;
                }
                let entry = if (code as usize) < dict.len() {
                    dict[code as usize].clone()
                } else if let Some(p) = &prev {
                    // The "KwKwK" case: a code that references the entry
                    // about to be added this very iteration.
                    let mut e = p.clone();
                    e.push(p[0]);
                    e
                } else {
                    panic!("invalid LZW code stream: unresolvable code {code}");
                };
                out.extend_from_slice(&entry);
                if let Some(p) = &prev {
                    let mut new_entry = p.clone();
                    new_entry.push(entry[0]);
                    dict.push(new_entry);
                    // `>=`, not `>`: the decoder's table is always exactly
                    // one entry behind the encoder's (it only learns a new
                    // (prefix, byte) pair *after* decoding the code that
                    // reveals the byte, whereas the encoder knows it one
                    // iteration earlier, from the input it's compressing).
                    // Bumping at `>` here reads the code the encoder wrote
                    // just after its own bump using the *old*, now
                    // one-bit-too-narrow width, corrupting every code after
                    // it -- caught by
                    // `lzw_round_trips_non_repeating_data_forcing_a_dictionary_reset`.
                    if dict.len() >= (1 << code_width) && code_width < 12 {
                        code_width += 1;
                    }
                }
                prev = Some(entry);
            }
            out
        }

        #[test]
        fn lzw_round_trips_uniform_data() {
            let indices = vec![7u8; 1000];
            assert_eq!(lzw_decode(&lzw_encode(&indices)), indices);
        }

        #[test]
        fn lzw_round_trips_a_short_repeating_pattern() {
            let indices: Vec<u8> = (0..2000).map(|i| (i % 4) as u8).collect();
            assert_eq!(lzw_decode(&lzw_encode(&indices)), indices);
        }

        #[test]
        fn lzw_round_trips_a_single_byte() {
            let indices = vec![42u8];
            assert_eq!(lzw_decode(&lzw_encode(&indices)), indices);
        }

        #[test]
        fn lzw_round_trips_empty_input() {
            let indices: Vec<u8> = vec![];
            assert_eq!(lzw_decode(&lzw_encode(&indices)), indices);
        }

        #[test]
        fn lzw_round_trips_non_repeating_data_forcing_a_dictionary_reset() {
            // Enough distinct byte-pairs that the 12-bit code space (4096
            // entries) fills up and `lzw_encode` has to emit a mid-stream
            // Clear Code, exercising the reset path -- and, along the way,
            // every code-width transition from 9 up through 12 bits, which
            // is exactly where an encoder/decoder timing mismatch would
            // first desync.
            let indices: Vec<u8> = (0..6000u32).map(|i| ((i * 37) % 256) as u8).collect();
            assert_eq!(lzw_decode(&lzw_encode(&indices)), indices);
        }

        #[test]
        fn lzw_encoding_actually_compresses_repetitive_data() {
            let indices = vec![0u8; 10_000];
            let encoded = lzw_encode(&indices);
            assert!(
                encoded.len() < 200,
                "10,000 repeated bytes should compress to well under 200 bytes, got {}",
                encoded.len()
            );
        }
    }
}

/// Builds a 256-entry palette by sampling `colormap` across the value range
/// a frame's normalized `t` can take: the full `[-1, 1]` for signed
/// components, or `[0, 1]` (packed into the same 256 slots for finer
/// gradation) for `Energy`.
fn build_palette(signed: bool) -> [[u8; 3]; 256] {
    let mut palette = [[0u8; 3]; 256];
    for (i, entry) in palette.iter_mut().enumerate() {
        let frac = i as f32 / 255.0;
        let t = if signed { -1.0 + 2.0 * frac } else { frac };
        *entry = colormap(t, signed);
    }
    palette
}

/// Maps a normalized value in `[-1, 1]` (signed) or `[0, 1]` (energy) to a
/// palette index built by `build_palette`.
fn palette_index(t: f32, signed: bool) -> u8 {
    let frac = if signed { (t + 1.0) / 2.0 } else { t };
    (frac.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn run(config: Config) -> Result<(), String> {
    let dims = GridDims::new(config.nx, config.ny, config.nz);
    let (bx_n, by_n, bz_n) = dims.block_dims();

    let file = File::open(&config.input).map_err(|e| format!("failed to open {:?}: {e}", config.input))?;
    // SAFETY: `Mmap::map` is unsafe because the file could in principle be
    // truncated or modified by another process while mapped, which would
    // turn later reads into a SIGBUS/torn read. This is a short-lived,
    // read-only, single-shot CLI tool reading a completed simulation's
    // output file, not a long-running process racing a concurrent writer,
    // so that hazard is not a realistic concern here.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("failed to mmap {:?}: {e}", config.input))?;

    let block_bytes = std::mem::size_of::<FieldBlock>();
    let snapshot_bytes = bx_n * by_n * bz_n * block_bytes;
    if snapshot_bytes == 0 || mmap.len() % snapshot_bytes != 0 {
        return Err(format!(
            "{:?} ({} bytes) is not a whole number of {}x{}x{} snapshots ({} bytes each) -- \
             check --nx/--ny/--nz match the run that produced this file",
            config.input,
            mmap.len(),
            dims.nx,
            dims.ny,
            dims.nz,
            snapshot_bytes
        ));
    }
    let num_snapshots = mmap.len() / snapshot_bytes;

    let indices: Vec<usize> = match config.snapshots {
        SnapshotSpec::Single(n) => vec![n],
        SnapshotSpec::Range(a, b) => (a..=b).collect(),
    };
    for &n in &indices {
        if n >= num_snapshots {
            return Err(format!(
                "--snapshot(s) {n} out of range: file has {num_snapshots} snapshot(s)"
            ));
        }
    }

    let extent_along_axis = match config.axis {
        Axis::X => dims.nx,
        Axis::Y => dims.ny,
        Axis::Z => dims.nz,
    };
    let slice_index = config.slice.unwrap_or(extent_along_axis / 2);
    if config.mode == Mode::Slice && slice_index >= extent_along_axis {
        return Err(format!(
            "--slice {slice_index} out of range for axis {:?} (extent {extent_along_axis})",
            config.axis
        ));
    }

    let mut frames: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(indices.len());
    for &n in &indices {
        let bytes = &mmap[n * snapshot_bytes..(n + 1) * snapshot_bytes];
        frames.push(compute_values(
            bytes,
            &dims,
            config.mode,
            config.axis,
            slice_index,
            config.component,
        ));
    }

    let global_max_abs = frames
        .iter()
        .flat_map(|(_, _, values)| values.iter())
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    let signed = config.component.is_signed();

    match config.snapshots {
        SnapshotSpec::Single(n) => {
            let (width, height, values) = &frames[0];
            let pixels: Vec<[u8; 3]> = values
                .iter()
                .map(|&v| {
                    let t = if global_max_abs > 0.0 { v / global_max_abs } else { 0.0 };
                    colormap(t, signed)
                })
                .collect();
            write_ppm(&config.output, *width, *height, &pixels)
                .map_err(|e| format!("failed to write {:?}: {e}", config.output))?;
            eprintln!(
                "wavefront-view: wrote {:?} ({width}x{height}, snapshot {n}/{num_snapshots}, mode {:?}, axis {:?}, \
                 component {:?}, max |value| = {global_max_abs:e})",
                config.output, config.mode, config.axis, config.component
            );
        }
        SnapshotSpec::Range(a, b) => {
            let (width, height, _) = &frames[0];
            let (width, height) = (*width, *height);
            let palette = build_palette(signed);
            let frame_indices: Vec<Vec<u8>> = frames
                .iter()
                .map(|(_, _, values)| {
                    values
                        .iter()
                        .map(|&v| {
                            let t = if global_max_abs > 0.0 { v / global_max_abs } else { 0.0 };
                            palette_index(t, signed)
                        })
                        .collect()
                })
                .collect();
            let delay_centiseconds = (100 / config.fps.max(1)).max(1) as u16;
            gif::write_animated_gif(
                &config.output,
                width,
                height,
                &palette,
                &frame_indices,
                delay_centiseconds,
            )
            .map_err(|e| format!("failed to write {:?}: {e}", config.output))?;
            eprintln!(
                "wavefront-view: wrote {:?} ({width}x{height}, {} frame(s) [{a}..={b}], {} fps, mode {:?}, \
                 axis {:?}, component {:?}, max |value| = {global_max_abs:e})",
                config.output,
                frame_indices.len(),
                config.fps,
                config.mode,
                config.axis,
                config.component
            );
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("wavefront-view: {e}\n");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match run(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wavefront-view: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes `blocks` the same way `engine.rs::serialize_snapshot`
    /// does (block-major, `Ex,Ey,Ez,Hx,Hy,Hz` per block) -- built
    /// independently of this file's own `read_component`/`sample`, so a
    /// round-trip test through it exercises the real on-disk format rather
    /// than mirroring whatever offset math the reader happens to use.
    fn build_snapshot_bytes(blocks: &[FieldBlock]) -> Vec<u8> {
        let mut out = Vec::with_capacity(std::mem::size_of_val(blocks));
        for block in blocks {
            for arr in [&block.ex, &block.ey, &block.ez, &block.hx, &block.hy, &block.hz] {
                for v in arr {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    fn zero_blocks(dims: &GridDims) -> Vec<FieldBlock> {
        let (bx_n, by_n, bz_n) = dims.block_dims();
        vec![FieldBlock::ZERO; bx_n * by_n * bz_n]
    }

    fn set_ez(blocks: &mut [FieldBlock], dims: &GridDims, x: usize, y: usize, z: usize, value: f32) {
        let (bx_n, by_n, _) = dims.block_dims();
        let (bx, by, bz) = (x / BLOCK_DIM, y / BLOCK_DIM, z / BLOCK_DIM);
        let (lx, ly, lz) = (x % BLOCK_DIM, y % BLOCK_DIM, z % BLOCK_DIM);
        blocks[(bz * by_n + by) * bx_n + bx].ez[FieldBlock::local_index(lx, ly, lz)] = value;
    }

    #[test]
    fn sample_reads_correct_component_and_voxel() {
        let dims = GridDims::new(8, 8, 8); // exactly one block
        let (bx_n, by_n, _) = dims.block_dims();
        let mut blocks = zero_blocks(&dims);
        blocks[0].ex[FieldBlock::local_index(3, 4, 5)] = 42.0;
        let bytes = build_snapshot_bytes(&blocks);

        assert_eq!(sample(&bytes, bx_n, by_n, 3, 4, 5, Component::Ex), 42.0);
        assert_eq!(sample(&bytes, bx_n, by_n, 0, 0, 0, Component::Ex), 0.0);
        assert_eq!(sample(&bytes, bx_n, by_n, 3, 4, 5, Component::Ey), 0.0);
    }

    #[test]
    fn energy_component_sums_squares_of_all_six() {
        let dims = GridDims::new(8, 8, 8);
        let (bx_n, by_n, _) = dims.block_dims();
        let mut blocks = zero_blocks(&dims);
        let local = FieldBlock::local_index(1, 1, 1);
        blocks[0].ex[local] = 3.0;
        blocks[0].hz[local] = 4.0;
        let bytes = build_snapshot_bytes(&blocks);

        assert_eq!(sample(&bytes, bx_n, by_n, 1, 1, 1, Component::Energy), 25.0);
    }

    #[test]
    fn compute_values_slice_matches_manual_placement_at_a_fixed_plane() {
        let dims = GridDims::new(16, 16, 16); // 2x2x2 blocks
        let mut blocks = zero_blocks(&dims);
        set_ez(&mut blocks, &dims, 0, 0, 5, 1.0);
        set_ez(&mut blocks, &dims, 15, 15, 5, 2.0);
        set_ez(&mut blocks, &dims, 7, 3, 5, -3.5);
        set_ez(&mut blocks, &dims, 10, 10, 10, 99.0); // different z: must not leak into slice 5
        let bytes = build_snapshot_bytes(&blocks);

        let (width, height, values) =
            compute_values(&bytes, &dims, Mode::Slice, Axis::Z, 5, Component::Ez);

        assert_eq!((width, height), (16, 16));
        assert_eq!(values[0], 1.0); // (x=0, y=0)
        assert_eq!(values[15 * width + 15], 2.0);
        assert_eq!(values[3 * width + 7], -3.5);
        assert_eq!(values[10 * width + 10], 0.0);
    }

    #[test]
    fn compute_values_volume_mode_picks_max_abs_preserving_sign() {
        let dims = GridDims::new(8, 8, 16); // extent 16 along Z (2 blocks deep)
        let mut blocks = zero_blocks(&dims);
        // Along the ray at (x=2, y=3): the largest-magnitude sample is
        // negative, so the projection must report -3.0, not +2.5.
        set_ez(&mut blocks, &dims, 2, 3, 0, 1.0);
        set_ez(&mut blocks, &dims, 2, 3, 5, -3.0);
        set_ez(&mut blocks, &dims, 2, 3, 10, 2.5);
        set_ez(&mut blocks, &dims, 2, 3, 15, 0.5);
        let bytes = build_snapshot_bytes(&blocks);

        let (width, _height, values) =
            compute_values(&bytes, &dims, Mode::Volume, Axis::Z, 0, Component::Ez);

        assert_eq!(values[3 * width + 2], -3.0);
    }

    #[test]
    fn palette_endpoints_match_colormap() {
        let signed = build_palette(true);
        assert_eq!(signed[0], colormap(-1.0, true));
        assert_eq!(signed[255], colormap(1.0, true));

        let energy = build_palette(false);
        assert_eq!(energy[0], colormap(0.0, false));
        assert_eq!(energy[255], colormap(1.0, false));
    }

    #[test]
    fn palette_index_maps_back_near_the_matching_colormap_entry() {
        // 256 buckets across the range means `palette[palette_index(t)]`
        // can be off by a shade from `colormap(t)` at bucket boundaries;
        // allow a small per-channel tolerance rather than exact equality.
        let palette = build_palette(true);
        for t in [-1.0, -0.5, 0.0, 0.3, 1.0] {
            let idx = palette_index(t, true);
            let got = palette[idx as usize];
            let expected = colormap(t, true);
            for c in 0..3 {
                assert!(
                    (got[c] as i16 - expected[c] as i16).abs() <= 3,
                    "t={t} idx={idx} got={got:?} expected={expected:?}"
                );
            }
        }
    }
}
