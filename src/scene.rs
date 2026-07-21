//! Plain-text scene description format for voxelizing material structures
//! into a [`MaterialGrid`], as a general alternative to a hardcoded demo
//! shape.
//!
//! Each non-empty, non-`#`-comment line describes one primitive, applied in
//! file order (later primitives overwrite earlier ones where they
//! overlap):
//!
//! ```text
//! # comment
//! sphere <eps_r> <mu_r> <sigma> <cx> <cy> <cz> <radius>
//! box    <eps_r> <mu_r> <sigma> <x0> <y0> <z0> <x1> <y1> <z1>
//! ```
//!
//! All geometric parameters are in voxel-index units, not meters -- this
//! format describes a voxelized structure, not a physical CAD model. Each
//! distinct `(eps_r, mu_r, sigma)` triple encountered is assigned its own
//! [`MaterialId`] the first time it's seen (up to 255 distinct materials;
//! `MaterialId(0)` is reserved for vacuum).
//!
//! This intentionally doesn't pull in a CAD/mesh/STL dependency: the crate's
//! dependency set is deliberately small and pinned (see `Cargo.toml`), and a
//! flat, line-oriented primitive list is enough to build up interesting
//! structural test scenes without one.

use crate::layout::{GridDims, MaterialGrid, MaterialId, MaterialTable};
use std::collections::HashMap;
use std::path::Path;

/// A hashable stand-in for `(eps_r, mu_r, sigma)`, used to deduplicate
/// materials by their exact bit pattern (so two primitives specifying the
/// identical physical constants share one [`MaterialId`] instead of eating
/// two of the 255 available slots).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialKey {
    eps_bits: u32,
    mu_bits: u32,
    sigma_bits: u32,
}

impl MaterialKey {
    fn new(eps_r: f32, mu_r: f32, sigma: f32) -> Self {
        Self {
            eps_bits: eps_r.to_bits(),
            mu_bits: mu_r.to_bits(),
            sigma_bits: sigma.to_bits(),
        }
    }
}

enum Primitive {
    Sphere {
        eps_r: f32,
        mu_r: f32,
        sigma: f32,
        center: (f32, f32, f32),
        radius: f32,
    },
    Box {
        eps_r: f32,
        mu_r: f32,
        sigma: f32,
        lo: (f32, f32, f32),
        hi: (f32, f32, f32),
    },
}

/// A parsed scene: an ordered list of geometric primitives, each tagged
/// with the material constants to voxelize it as.
pub struct Scene {
    primitives: Vec<Primitive>,
}

impl Scene {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("failed to read scene file {:?}: {e}", path.as_ref()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut primitives = Vec::new();

        for (lineno, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line_no = lineno + 1;
            let parts: Vec<&str> = line.split_whitespace().collect();

            let parse_nums = |tokens: &[&str]| -> Result<Vec<f32>, String> {
                tokens
                    .iter()
                    .map(|t| {
                        t.parse::<f32>()
                            .map_err(|_| format!("scene line {line_no}: invalid number '{t}'"))
                    })
                    .collect()
            };

            match parts.first().copied() {
                Some("sphere") => {
                    let v = parse_nums(&parts[1..])?;
                    if v.len() != 7 {
                        return Err(format!(
                            "scene line {line_no}: expected 'sphere eps_r mu_r sigma cx cy cz radius', got {} values",
                            v.len()
                        ));
                    }
                    primitives.push(Primitive::Sphere {
                        eps_r: v[0],
                        mu_r: v[1],
                        sigma: v[2],
                        center: (v[3], v[4], v[5]),
                        radius: v[6],
                    });
                }
                Some("box") => {
                    let v = parse_nums(&parts[1..])?;
                    if v.len() != 9 {
                        return Err(format!(
                            "scene line {line_no}: expected 'box eps_r mu_r sigma x0 y0 z0 x1 y1 z1', got {} values",
                            v.len()
                        ));
                    }
                    primitives.push(Primitive::Box {
                        eps_r: v[0],
                        mu_r: v[1],
                        sigma: v[2],
                        lo: (v[3], v[4], v[5]),
                        hi: (v[6], v[7], v[8]),
                    });
                }
                Some(other) => {
                    return Err(format!("scene line {line_no}: unknown primitive '{other}'"))
                }
                None => {}
            }
        }

        Ok(Self { primitives })
    }

    /// Voxelizes every primitive into `grid`, registering each distinct
    /// material into `table` (using `dt`/`dx` to derive its Yee update
    /// coefficients). Returns the number of distinct non-vacuum materials
    /// used.
    pub fn voxelize(
        &self,
        grid: &mut MaterialGrid,
        table: &mut MaterialTable,
        dt: f32,
        dx: f32,
    ) -> Result<usize, String> {
        let dims = grid.dims();
        let mut ids: HashMap<MaterialKey, MaterialId> = HashMap::new();
        let mut next_id: u16 = 1; // 0 is reserved for vacuum

        for prim in &self.primitives {
            let (eps_r, mu_r, sigma) = match *prim {
                Primitive::Sphere {
                    eps_r, mu_r, sigma, ..
                } => (eps_r, mu_r, sigma),
                Primitive::Box {
                    eps_r, mu_r, sigma, ..
                } => (eps_r, mu_r, sigma),
            };
            let key = MaterialKey::new(eps_r, mu_r, sigma);
            let id = match ids.get(&key) {
                Some(id) => *id,
                None => {
                    if next_id > 255 {
                        return Err("scene uses more than 255 distinct materials".to_string());
                    }
                    let id = MaterialId(next_id as u8);
                    table.set_material(id, eps_r, mu_r, sigma, dt, dx);
                    ids.insert(key, id);
                    next_id += 1;
                    id
                }
            };

            Self::rasterize(prim, dims, grid, id);
        }

        Ok(ids.len())
    }

    fn rasterize(prim: &Primitive, dims: GridDims, grid: &mut MaterialGrid, id: MaterialId) {
        match *prim {
            Primitive::Sphere {
                center: (cx, cy, cz),
                radius,
                ..
            } => {
                let radius_sq = radius * radius;
                for z in 0..dims.nz {
                    for y in 0..dims.ny {
                        for x in 0..dims.nx {
                            let dx = x as f32 + 0.5 - cx;
                            let dy = y as f32 + 0.5 - cy;
                            let dz = z as f32 + 0.5 - cz;
                            if dx * dx + dy * dy + dz * dz <= radius_sq {
                                grid.set_material_at(x, y, z, id);
                            }
                        }
                    }
                }
            }
            Primitive::Box {
                lo: (x0, y0, z0),
                hi: (x1, y1, z1),
                ..
            } => {
                let clamp_lo = |v: f32| v.max(0.0).floor() as usize;
                let clamp_hi = |v: f32, n: usize| (v.ceil().max(0.0) as usize).min(n);
                let (xlo, xhi) = (clamp_lo(x0.min(x1)), clamp_hi(x0.max(x1), dims.nx));
                let (ylo, yhi) = (clamp_lo(y0.min(y1)), clamp_hi(y0.max(y1), dims.ny));
                let (zlo, zhi) = (clamp_lo(z0.min(z1)), clamp_hi(z0.max(z1), dims.nz));
                for z in zlo..zhi {
                    for y in ylo..yhi {
                        for x in xlo..xhi {
                            grid.set_material_at(x, y, z, id);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{GridDims, MaterialTable};

    #[test]
    fn parses_comments_blank_lines_and_both_primitives() {
        let text = "\
            # a comment\n\
            \n\
            sphere 4.0 1.0 0.0 16 16 16 5\n\
            box 2.0 1.0 0.01 0 0 0 8 8 8\n";
        let scene = Scene::parse(text).expect("valid scene should parse");
        assert_eq!(scene.primitives.len(), 2);
    }

    #[test]
    fn rejects_unknown_primitive() {
        let Err(err) = Scene::parse("cone 1.0 1.0 0.0 0 0 0 1") else {
            panic!("expected a parse error");
        };
        assert!(err.contains("unknown primitive"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_wrong_argument_count() {
        let Err(err) = Scene::parse("sphere 1.0 2.0 3.0") else {
            panic!("expected a parse error");
        };
        assert!(err.contains("expected 'sphere"), "unexpected error: {err}");
    }

    #[test]
    fn deduplicates_identical_materials_into_one_id() {
        let dims = GridDims::new(32, 32, 32);
        let path = std::env::temp_dir().join(format!(
            "wavefront_test_scene_{}_{:?}.grid",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut grid = crate::layout::MaterialGrid::create(&path, dims).unwrap();
        let mut table = MaterialTable::vacuum_filled(1.5e-12, 1.0e-3);

        // Two spheres with the exact same (eps_r, mu_r, sigma) should share
        // one material slot, not consume two.
        let scene = Scene::parse(
            "sphere 4.0 1.0 0.0 8 8 8 3\nsphere 4.0 1.0 0.0 24 24 24 3\n",
        )
        .unwrap();
        let n = scene.voxelize(&mut grid, &mut table, 1.5e-12, 1.0e-3).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(n, 1);
    }
}
