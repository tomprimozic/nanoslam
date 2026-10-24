# nanoslam turbo

Monocular SLAM in Python, CPU only, geometry only: no learned components, no descriptors, no place recognition (loops
are closed by projecting the map). Hot loops are hand-written numba kernels (patch alignment, bundle adjustment,
triangulation, pose graph); the map is a handful of flat numpy tables.

```
cd turbo
../.venv/bin/python run.py household                     # one sequence (names: datasets.py / run.py all)
../.venv/bin/python run.py household --rerun             # with visualization (--rerun-save x.rrd: to a file)
../.venv/bin/python run.py all --out runs/NAME            # all sequences in parallel, then a summary table
../.venv/bin/python run.py all --out runs/NAME --starts 0,7,14            # ensemble over start frames (use this!)
../.venv/bin/python run.py all --out runs/X --set loop=False final_ba_iters=0   # override any Config value
../.venv/bin/python run.py all --out runs/P --perturb 3   # every numeric parameter randomly scaled by 0.7-1.4x
../.venv/bin/python run.py summary runs/A runs/B          # compare runs (metrics recomputed from trajectories)
```

Datasets live in `../datasets` (override with `NANOSLAM_DATASETS`): TUM RGB-D `rgbd_dataset_freiburg*/`, EuRoC
`euroc/<name>/mav0/` (cam0 + ground truth extracted from the zips), TartanAir `tartanair/<name>_sample_P00x/P00x/`
(left images + poses).

| file | |
|---|---|
| `slam.py` | the system: `Config`, tracking, mapping, loop closing |
| `map.py` | the map: keyframe / point / observation tables |
| `lk.py` | patch alignment (numba): template from a keyframe, warped into the current view |
| `optim.py` | bundle adjustment (Schur LM), pose refinement, triangulation, Sim3 pose graph (numba) |
| `geometry.py` | SE3 / Sim3 helpers |
| `datasets.py` | TUM RGB-D, EuRoC, TartanAir loaders (undistorted grayscale + ground truth) |
| `evaluate.py` | drift-oriented evaluation (alignment on the initial section only) |
| `run.py` | runner, parallel benchmark with code snapshots, summaries, parameter perturbation |

## How it works

**Map** (`map.py`). Three tables with growable columns (`Table`): keyframes (pose, frame, parent in the spanning tree,
segment, alive), points (position, state, reference keyframe + pixel, observation / visibility counts) and observations
(keyframe, point, pixel). There is no Python object per point: everything is selected with masks and gathered with
index arrays, and covisibility is a sparse incidence-matrix product. Every detected feature is a point from the start:
a *candidate* collects keyframe observations while it is tracked and becomes *mapped* when multi-view triangulation
succeeds (>= 1 deg parallax), like DSO's immature points, so no observations are thrown away.

**Every frame.** CLAHE; KLT (OpenCV) in two passes: tracked map points from their projection at the constant-velocity
prediction (shallow pyramid), everything else from its previous position; forward-backward check. RANSAC PnP (poselib)
with Huber refinement. Candidate tracks far from the epipolar line of their first observation are dropped. If fewer
than 50 map points are tracked: *relocalization by projection* (local map, wide window). If tracking is lost (fast
rotation + motion blur, e.g. V2_03): several pose hypotheses, DSO-style: the last good pose rotated by +-15 / 30 deg,
then, round-robin, all keyframe poses (nearest first), each searched with its own points.

**Patch alignment** (`lk.py`). The template comes from a keyframe that observed the point: sampled around that
observation, warped with the local affine approximation of the homography of the plane through the point
(fronto-parallel in the keyframe), from the pyramid level whose resolution matches the target level. Inverse
compositional LK on zero-mean / unit-variance patches (gain + bias invariant), coarse to fine, optionally starting with
a brute-force ZNCC search on the top level for wide windows. ~0.03 px on synthetic warps, 3-15 us per point.

**Every keyframe** (rotation-compensated median parallax >= 1.75 deg, or tracking getting thin):
1. *re-measure* tracked map points with patches from their reference keyframe: KLT positions accumulate template
   drift, these don't;
2. *search by projection*: every untracked map point that projects into the keyframe is searched from the keyframe
   that saw it from the most similar viewpoint; found points are *fused* with a track at the same place (duplicates
   merged, the point with more observations survives) or become tracks again (long tracks = less drift);
3. triangulate candidates (multi-view, from all their keyframe observations);
4. local BA: the keyframe and its 9 most covisible ones; every other keyframe observing their points is held fixed;
   outliers (chi2, behind the camera) are removed;
5. loop detection (below);
6. culling: points rarely found when they should be visible (ORB-SLAM's found / visible ratio) or seen by <= 2
   keyframes; keyframes that are near-duplicates of an older one (same place, orientation and points: revisits,
   hovering);
7. new corners in empty grid cells (the strongest min-eigenvalue corner per cell).

**Loops by projection** (no place recognition). Points not seen for 20 keyframes are searched with a wide window (top
pyramid level 2 + brute-force search). If they agree on a pose (PnP on them alone) that differs from the current one
by < 5 px, they are just fused (map reuse: local BA pulls the keyframe in); if by more: scale drift from the depth
ratio of old vs. nearby current points, Sim3 pose graph over the *essential graph* (spanning tree + the 5 strongest
covisibility links per keyframe, all at their current relative poses, so only the new loop edge has an error), points
move with their reference keyframe, old points are re-searched and fused, global BA.

**Optimization** (`optim.py`). BA: Levenberg-Marquardt, Huber, points eliminated with the Schur complement; the
reduced camera system is dense (Cholesky), fine for local BA and for global BA up to a few hundred keyframes. Poses are
updated as R <- exp(w) R, t <- t + v, so that a single translation component can be held fixed: the monocular gauge is
7 DoF (the oldest keyframe fixed + the scale along its baseline to the next one), so the initialization pair is refined
too (with two fixed keyframes it never is). Sim3 pose graph: LM with per-edge numerical Jacobians on local increments
and a sparse solve (400 nodes in 0.15 s).

**Evaluation** (`evaluate.py`). The estimate is Sim3-aligned to the ground truth on the *initial section* only: the
shortest prefix (from the first tracked frame) whose ground truth covers 10% of the full extent along both main axes
(a trajectory spanning 20 x 10 m is aligned on the prefix that spans 2 x 1 m). Then `drift` = RMSE over all frames,
`final` = RMSE over the last 10% of frames, `drift%` = drift / path length, `ate` = the usual whole-trajectory
alignment (for reference), `scale` = scale of the final section relative to the initial one. Frames where tracking was
lost are scored with the last pose (losing track must not be better than tracking badly); ground truth gaps (fr2/desk
has a 12 s mocap dropout) are skipped. The default trajectory uses the final keyframe poses (after loops / final BA).

Single runs are chaotic (a few points surviving a hard segment decide the scale): compare configurations with
`--starts 0,7,14` (median over start frames per sequence, geometric mean over sequences), and in VO mode
(`--set loop=False final_ba_iters=0`) for drift.

## Results

Median over 3 start frames (0, 7, 14) per sequence, lost frames scored with the last pose, metres. *full* = the
default system (loops + final BA), *VO* = `loop=False final_ba_iters=0` (local map reuse only, no loop closing, no
final BA). One config for all 17 sequences; only the camera calibration comes from the dataset. (`runs/final_full`,
`runs/final_vo`)

| sequence | drift (full) | final (full) | ATE (full) | drift (VO) | ATE (VO) | tracked |
|---|---|---|---|---|---|---|
| fr2/desk | 0.035 | 0.009 | 0.010 | 0.066 | 0.013 | 0.99 |
| fr3/long_office_household | 0.067 | 0.013 | 0.011 | 0.106 | 0.041 | 0.98 |
| MH_01_easy | 0.037 | 0.029 | 0.013 | 0.118 | 0.087 | 1.00 |
| MH_02_easy | 0.073 | 0.027 | 0.018 | 0.073 | 0.040 | 1.00 |
| MH_03_medium | 0.087 | 0.037 | 0.026 | 0.260 | 0.164 | 1.00 |
| MH_04_difficult | 0.303 | 0.047 | 0.050 | 0.347 | 0.122 | 0.99 |
| MH_05_difficult | 0.150 | 0.074 | 0.043 | 0.173 | 0.111 | 0.96 |
| V1_01_easy | 0.187 | 0.103 | 0.033 | 0.192 | 0.055 | 0.97 |
| V1_02_medium | 0.025 | 0.017 | 0.008 | 0.146 | 0.113 | 0.96 |
| V1_03_difficult | 0.037 | 0.024 | 0.014 | 0.106 | 0.065 | 0.95 |
| V2_01_easy | 0.032 | 0.013 | 0.014 | 0.147 | 0.065 | 0.97 |
| V2_02_medium | 0.029 | 0.032 | 0.010 | 0.225 | 0.102 | 0.98 |
| V2_03_difficult | 0.624 | 0.071 | 0.579 | 0.741 | 0.725 | 0.86 |
| abandonedfactory | 0.262 | 0.712 | 0.209 | 0.371 | 0.210 | 0.97 |
| carwelding | 0.081 | 0.057 | 0.035 | 0.305 | 0.131 | 0.99 |
| seasidetown | 2.988 | 5.551 | 2.893 | 3.240 | 2.895 | 0.26 |
| seasonsforest | 0.180 | 0.219 | 0.057 | 0.197 | 0.079 | 0.99 |
| **geometric mean** | **0.107** | **0.057** | **0.038** | **0.216** | **0.110** | |

- On the frames it tracks, V2_03 is ~3 cm ATE; its large numbers come from a ~170 frame stretch through a part of
  the room that was never mapped, after motion blur (relocalization can't succeed there; `reinit=True` starts a new
  map segment but rarely merges it correctly). abandonedfactory loses track in its last 10 frames (dark wall, fast),
  seasidetown after frame 100 (0.25 m / frame around a pillar into a dark room, ~10 fps rendering).
- For orientation only (the numbers vary between papers): ORB-SLAM3 monocular on EuRoC is reported at MH_01
  1.6-2.9 cm, MH_04 8.5-13.8 cm, V2_03 36-52 cm or a failure ([1](https://arxiv.org/pdf/2405.03413),
  [2](https://arxiv.org/pdf/2012.11863), [3](https://arxiv.org/pdf/2405.13748)).
- The human-written `nanoslam.py` on household, same evaluation and start frames (online poses): drift 0.266, ATE
  0.134 (0.044 / 0.134 / 0.173 for starts 0 / 7 / 14).
- `drift` is dominated by the alignment lever arm: an error of the initial section's orientation of a degree makes
  tens of cm at the far end (V1_01: ATE 3 cm, drift 19 cm). Better initialization is where it improves next.
- Speed (single process, M5 Pro): household 96 fps, MH_01 100 fps, V1_02 39 fps (fast motion = many keyframes; local
  BA dominates). Tracking ~3 ms / frame, keyframes ~30-70 ms.
- Robustness to the parameters: 4 random perturbations of *all* numeric parameters (each scaled by 0.7-1.4x):
  geometric mean ATE 1.9-2.4 cm vs 2.3 cm with the defaults (single runs, lost frames not scored; `runs/v5_perturb*`).

## What was tried

Measured on all sequences; "ensemble" = 3 start frames, lost frames scored; differences below ~5% of the geometric
mean are noise.

| change | effect | |
|---|---|---|
| loop edges at their *current* relative poses (not stale measurements) + no pose graph for < 5 px corrections (just fuse) | median ATE 6.7 -> 2.3 cm, V1_02 1.8 m -> 0.8 cm (runaway corrections before) | kept |
| essential graph (tree + 5 strongest links) instead of all covisibility edges | pose graph 137 s -> 2.6 s on V2_03, same accuracy | kept |
| multi-hypothesis relocalization (rotations, round-robin keyframes) | V2_03 1.38 m -> 2.6 cm (on tracked frames) | kept |
| 7 DoF gauge (vs two fixed keyframes) | VO ensemble: ATE 12.0 -> 11.0 cm, drift 0.227 -> 0.216 | kept |
| re-measuring tracks from their reference keyframe | without: VO ATE 7.2 -> 8.3 cm, drift +22% | kept |
| near-duplicate keyframe culling | 10-30% fewer keyframes, same accuracy (ORB-SLAM's "90% seen by 3 others" culled 90% and broke the scale) | kept |
| re-measuring candidates too (rotation-only warp) | VO slightly worse | removed |
| patch source: reference keyframe first (vs most similar view) | VO ensemble worse (ATE 11.0 -> 11.6 cm) | removed |
| tracking the local map every frame (ORB-SLAM's TrackLocalMap) | VO ensemble neutral (big wins and losses per sequence), 20% slower | option, off |
| new map segment when lost (re-initialization, merged by a Sim3 loop) | only changes seasidetown (26% -> 83% tracked, still wrong) | option, off |
| intensity stretch before CLAHE (dark images) | no effect | removed |
| VO: ba_window 15, ba_iters 20, 30 cells, kf_parallax 1.0 / 2.5, tri_min_angle 0.5 / 2.0 | none better (the defaults are on a plateau) | - |

Bugs that cost the most time: `scipy.linalg.solve(assume_a='pos')` (scipy 1.18.1 / Accelerate) corrupts the heap,
the crash then shows up anywhere (numba cache loading, GC at exit) - BA uses `cho_factor`; scoring only tracked frames
made losing track look *better* than tracking badly; fr2/desk ground truth interpolated over a 12 s mocap gap.

## Next

- initialization (DfUSMC / Instant VO-style small-motion init): the start of the trajectory dominates `drift`;
- voxel map instead of keyframes for reprojection and loops; dense / semi-dense reconstruction;
- tracking through very fast motion (TartanAir) and re-joining new map segments reliably;
- sparse reduced camera system for global BA beyond ~1000 keyframes; DSO-style marginalization in the local window.
