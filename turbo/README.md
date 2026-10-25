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
../.venv/bin/python run.py summary runs/A0+runs/A1 runs/B0+runs/B1   # ... pooled over seeds
../.venv/bin/python run.py view runs/A/household@0       # rerun view of saved runs (alignment, errors over time)
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
prediction (shallow pyramid), everything else from its previous position; forward-backward check. The global image
shift from phase correlation (pyramid level 2) corrects both initial guesses when the motion is not what the velocity
predicted (fast rotations). RANSAC PnP (poselib)
with Huber refinement. Candidate tracks far from the epipolar line of their first observation are dropped. If fewer
than 50 map points are tracked: *relocalization by projection* (local map, wide window). If tracking is lost (fast
rotation + motion blur, e.g. V2_03): several pose hypotheses, DSO-style: the last good pose rotated by +-15 / 30 deg,
then, round-robin, all keyframe poses (nearest first), each searched with its own points.

**Initialization** (`try_initialize_window`; the design follows `../../slamfold/DESIGN.md`). Corners detected in the
first frame are tracked; every frame (once their median parallax is >= 0.5 deg) a batch BA runs over the frames since
then, up to 20 of them chosen evenly in parallax (a long still start collapses to a few): rotations start from the
*rotation-only* solution of each frame (robust Wahba on the bearings), translations from the 5-point direction scaled
by the frame's parallax, points from two-view triangulation; gauge = first frame fixed + scale along the baseline.
Translation counts as established when the likelihood ratio of the full model against the rotation-only one exceeds
2x its extra parameters and >= 50 points (and >= half of the inliers) have >= 1 deg of parallax; the window's points
seed the map, the last frame becomes the second keyframe, and the earlier frames get their poses with hindsight.
Compared with the two-view initialization (5-point + 2.5 deg median parallax) it initializes earlier where the start
is slow (household frame 14 vs 61, MH_05 42 vs 80) with the same early accuracy (first 150 frames: position 1.00x,
rotation 0.86x); downstream it is neutral (final 0.98 [0.88, 1.07]). `rotation_only_poses=True` also emits the
rotation-only pose (t = 0) for every frame before that.

**Patch alignment** (`lk.py`). The template comes from a keyframe that observed the point: sampled around that
observation, warped with the local affine approximation of the homography of the plane through the point
(fronto-parallel in the keyframe), from the pyramid level whose resolution matches the target level. Inverse
compositional LK on zero-mean / unit-variance patches (gain + bias invariant), coarse to fine, optionally starting with
a brute-force ZNCC search on the top level for wide windows. ~0.03 px on synthetic warps, 3-15 us per point. Keyframe
pyramids live in one (slots, h, w) array per level (`lk.ImageStore`), so all points are aligned in one call whatever
keyframe their patch comes from.

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

**Loops by projection** (no place recognition). At every 3rd keyframe, points not seen for 20 keyframes are searched
with a wide window (top pyramid level 2 + brute-force search). If they agree on a pose (PnP on them alone) that differs from the current one
by < 5 px, they are just fused (map reuse: local BA pulls the keyframe in); if by more: scale drift from the depth
ratio of old vs. nearby current points, Sim3 pose graph over the *essential graph* (spanning tree + the 5 strongest
covisibility links per keyframe, all at their current relative poses, so only the new loop edge has an error), points
move with their reference keyframe, old points are re-searched and fused, then a global BA - in a background thread
on a snapshot of the map, merged 60 frames later ORB-SLAM style (keyframes created meanwhile move with their nearest
corrected ancestor, points with their reference keyframe). It is merged at a fixed frame, never "when it is done",
so runs stay deterministic.

**Optimization** (`optim.py`). BA: Levenberg-Marquardt, Huber, points eliminated with the Schur complement; the
reduced camera system is dense (Cholesky), fine for local BA and for global BA up to a few hundred keyframes. Poses are
updated as R <- exp(w) R, t <- t + v, so that a single translation component can be held fixed: the monocular gauge is
7 DoF (the oldest keyframe fixed + the scale along its baseline to the next one), so the initialization pair is refined
too (with two fixed keyframes it never is). Sim3 pose graph: LM with per-edge numerical Jacobians on local increments
and a sparse solve (400 nodes in 0.15 s).

**Evaluation** (`evaluate.py`). The estimate is Sim3-aligned to the ground truth on the *initial section* only: the
shortest prefix (from the first tracked frame) whose ground truth covers 20% of the full extent along both main axes
(a trajectory spanning 20 x 10 m is aligned on the prefix that spans 4 x 2 m). Poses are the *live* ones (as estimated
at that frame); `--trajectory final` uses the keyframe poses with hindsight (after later loop closures / final BA).

- `final` (headline): RMSE over the last 10% of frames
- `rot`: rotation drift, RMS over the last 10% of frames of the error of the rotation since the end of the initial
  section (deg; independent of the alignment)
- `drift`: RMSE over all frames; `ate`: the usual whole-trajectory alignment (for reference); `scale`: scale of the
  final section relative to the initial one; `tracked`: fraction of frames with a pose
- frames where tracking was lost are scored with the last pose (losing track must not be better than tracking badly);
  ground truth gaps (fr2/desk has a 12 s mocap dropout) are skipped

Per frame, runs also save intrinsic diagnostics (`stats_*` in `trajectory.npz`): residual scale of the PnP (MAD),
inliers, image flow, sharpness, tracked / candidate counts, latency.

**Noise.** Runs are deterministic (same start frame + seed = identical poses, also with the background BA thread), but
chaotic: compare configurations with `--starts 0,5,10,15,20`, preferably pooled over two seeds (`summary
runs/A0+runs/A1 runs/B0+runs/B1`). The summary gives per sequence the median over the runs, the geometric mean over
sequences, the *spread* (max / min over the start frames: how chaotic it is) and a paired comparison with the first
column (geometric mean of the per-sequence ratios with a bootstrap 95% interval, sequences better / worse). Measured
noise floor: changing only the RANSAC seed moves the geometric mean of `final` by 6-9% (95% interval up to +-20-30%),
so with 5 starts x 17 sequences effects below ~15% are not resolvable.

## Results

*Live* poses (as estimated at each frame), median over 10 runs per sequence (start frames 0, 5, 10, 15, 20 x RANSAC
seeds 0, 1), 20% coverage alignment, lost frames scored with the last pose. *final* = RMSE of the last 10% of frames
[m], *rot* = rotation drift [deg], *ATE* = whole-trajectory alignment [m]. *full* = the default system, *VO* =
`loop=False final_ba_iters=0` (no loop closing). One config for all 17 sequences; only the camera calibration comes
from the dataset. (`runs/f_final_seed*`, `runs/f_final_vo_seed*`). *lost* = tracking losses after initialization
(mean over the runs; frames before initialization are not counted).

| sequence | final (full) | rot (full) | ATE (full) | final (VO) | rot (VO) | ATE (VO) | lost: episodes / frames |
|---|---|---|---|---|---|---|---|
| fr2/desk | 0.008 | 1.16 | 0.011 | 0.025 | 1.35 | 0.013 | 0.0 / 0.0% |
| fr3/long_office_household | 0.012 | 0.75 | 0.031 | 0.139 | 2.23 | 0.045 | 0.0 / 0.0% |
| MH_01_easy | 0.031 | 3.55 | 0.022 | 0.219 | 3.73 | 0.084 | 0.0 / 0.0% |
| MH_02_easy | 0.039 | 1.16 | 0.036 | 0.131 | 1.30 | 0.044 | 0.0 / 0.0% |
| MH_03_medium | 0.022 | 1.84 | 0.049 | 0.161 | 0.98 | 0.148 | 0.0 / 0.0% |
| MH_04_difficult | 0.079 | 0.30 | 0.077 | 0.173 | 0.59 | 0.120 | 0.0 / 0.0% |
| MH_05_difficult | 0.064 | 1.09 | 0.059 | 0.235 | 0.78 | 0.128 | 0.0 / 0.0% |
| V1_01_easy | 0.072 | 3.30 | 0.035 | 0.101 | 3.55 | 0.057 | 0.0 / 0.0% |
| V1_02_medium | 0.015 | 0.41 | 0.016 | 0.200 | 3.81 | 0.145 | 0.0 / 0.0% |
| V1_03_difficult | 0.065 | 0.73 | 0.048 | 0.203 | 6.76 | 0.085 | 0.0 / 0.0% |
| V2_01_easy | 0.013 | 0.36 | 0.035 | 0.242 | 1.15 | 0.065 | 0.0 / 0.0% |
| V2_02_medium | 0.059 | 2.03 | 0.029 | 0.301 | 3.70 | 0.089 | 0.0 / 0.0% |
| V2_03_difficult | 0.197 | 2.04 | 0.640 | 0.427 | 3.90 | 0.629 | 1.9 / 9.3% |
| abandonedfactory P001 | 0.383 | 2.61 | 0.077 | 0.383 | 2.61 | 0.077 | 1.0 / 0.9% |
| carwelding P007 | 0.293 | 0.52 | 0.129 | 0.548 | 0.66 | 0.150 | 0.0 / 0.0% |
| seasidetown P003 | 9.207 | 101.42 | 3.105 | 9.207 | 101.42 | 3.105 | 1.0 / 57.1% |
| seasonsforest P002 | 0.489 | 0.27 | 0.082 | 0.489 | 0.27 | 0.082 | 0.0 / 0.0% |
| **geometric mean** | **0.075** | **1.32** | **0.062** | **0.254** | **2.15** | **0.108** | |

- With hindsight (`--trajectory final`: keyframe poses after all loop closures and the final BA) the full system's
  ATE is ~2x lower (geometric mean ~3.8 cm in the first round); live is what a robot would have had.
- The TartanAir samples are short point-to-point flights that barely revisit anything (0-2% of frames come back
  within 1 m with a similar view; seasidetown 13%, but tracking is lost there by then), so full ~ VO there. For
  comparison V1_02 57%, household 97%.
- On the frames it tracks, V2_03 is ~3 cm ATE; its large numbers come from a ~170 frame stretch through a part of
  the room that was never mapped, after motion blur (relocalization can't succeed there; `reinit=True` starts a new
  map segment but rarely merges it correctly). abandonedfactory loses track in its last 10 frames (dark wall, fast),
  seasidetown after frame 100 (0.25 m / frame around a pillar into a dark room, ~10 fps rendering).
- For orientation only (the numbers vary between papers): ORB-SLAM3 monocular on EuRoC is reported at MH_01
  1.6-2.9 cm, MH_04 8.5-13.8 cm, V2_03 36-52 cm or a failure ([1](https://arxiv.org/pdf/2405.03413),
  [2](https://arxiv.org/pdf/2012.11863), [3](https://arxiv.org/pdf/2405.13748)).
- The human-written `nanoslam.py` on household, same evaluation (live, 20%), starts 0 / 7 / 14: final 0.084 /
  0.367 / 0.560, ATE 0.044 / 0.134 / 0.173 (turbo: final 0.013 full, 0.154 VO).
- `drift` (all frames, initial-section alignment) is dominated by the alignment lever arm: a degree of error in the
  initial section's orientation makes tens of cm at the far end; `rot` measures the rotation drift without it.
- Speed (single process, M5 Pro, full system): household 144 fps, MH_01 120, MH_03 58, V2_02 57, V1_02 52.
  Latency per frame type (ms, p50 / p99 / max; `stats_ms`, `stats_keyframe`, `stats_loop_search`, `stats_loop_closed`):

  | | frames | keyframes | keyframes with loop search (every 3rd) | loop closures |
  |---|---|---|---|---|
  | household | 2.4 / 3.5 / 101 | 22 / 69 / 81 | 26 / 99 / 123 | 226 / 307 / 308 (3) |
  | MH_01 | 2.5 / 3.8 / 80 | 31 / 135 / 237 | 41 / 221 / 357 | - |
  | MH_03 | 2.8 / 4.1 / 97 | 40 / 147 / 175 | 71 / 240 / 259 | 1049 (1) |
  | V1_02 | 2.8 / 4.4 / 102 | 29 / 94 / 149 | 48 / 159 / 189 | 642 / 1614 / 1633 (3) |
  | V2_02 | 2.7 / 4.2 / 89 | 28 / 92 / 122 | 58 / 195 / 248 | 128 / 150 / 150 (2) |

  Loop closures are synchronous for the pose graph (0.1-0.8 s: numerical Jacobians over all keyframes) and the
  re-search; the global BA (3 iterations) runs in the background. The ~100 ms frame maxima are merges of the
  background BA / relocalization.
- Robustness to the parameters (first round, hindsight ATE, single runs, lost frames not scored): 4 random
  perturbations of *all* numeric parameters (each scaled by 0.7-1.4x): geometric mean ATE 1.9-2.4 cm vs 2.3 cm with
  the defaults (`runs/v5_perturb*`) - i.e. within the noise floor measured later.

## What was tried

First round: hindsight poses, 10% coverage; "ensemble" = 3 start frames (the noise floor was only measured later, see
the second round: differences below ~10-15% are not resolvable).

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

Second round (live poses, 20% coverage; VO = `loop=False final_ba_iters=0`, full = default system; 5 start frames;
ratio of `final` vs the baseline = geometric mean of per-sequence ratios [bootstrap 95% interval]; for reference,
changing only the RANSAC seed gives 1.09 [0.98, 1.20] (VO) and 1.06 [0.89, 1.32] (full)):

| change | ratio | |
|---|---|---|
| no motion model (constant velocity -> none), VO, 3 starts | 1.21 | keep constant velocity |
| constant acceleration, VO, 3 starts | 1.03 | no |
| no 3x3 blur before the corner score, VO | 1.18 [1.08, 1.31] | keep the blur |
| keep tracking PnP outliers ("soft"), VO | 1.21 [1.05, 1.42] | no |
| Cauchy instead of Huber in BA, VO | 1.04 [0.86, 1.28] | noise |
| no keyframe culling, VO | 1.08 [0.96, 1.23] | keep culling |
| + ORB-style thinning of old keyframes (redundant, > 20 keyframes old), VO | 0.98 [0.86, 1.08], 14% fewer keyframes | option |
| adaptive thresholds: PnP inliers at the live noise level, VO | 0.94 [0.76, 1.18] | option (`adaptive_pnp`) |
| adaptive thresholds: keyframe noise as BA weights, VO | 1.19 [1.07, 1.35] | no |
| drop tracks that slipped > 2 px at re-measurement, VO, 3 starts | 0.97 | option |
| no second local BA after fusing old points, full | 1.16 [1.03, 1.33] | keep it |
| search old points only every 3rd keyframe, full | 0.99 [0.88, 1.12], ~1/3 of the loop search time | adopted |
| no global BA after loop closures, full | 1.21 [0.92, 1.73], rotation 1.23 [1.04, 1.56] | keep it |
| no loop correction at all (old points only fused, no pose graph / global BA), full | 1.11 [0.73, 1.70], rotation 0.92 [0.72, 1.12] | inconclusive (only ~10 sequences close loops) |
| global BA with 3 iterations instead of 10, full, 2 seeds | 0.92 [0.84, 0.99], ATE 0.91 [0.83, 0.98], rotation 1.03 [1.00, 1.07] | adopted (3x cheaper) |
| phase-correlation prior for KLT (global image shift on pyramid level 2, `cv2.phaseCorrelate`), full, 2 seeds | final 1.04 [0.87, 1.22], rotation **0.88 [0.77, 0.99]** (14/3), lost frames 5.0 -> 3.9% | adopted |
| trim the 20% largest residuals inside each patch-alignment iteration (robust LK), 2 seeds | full 1.10 [1.01, 1.20], VO 1.17 [1.04, 1.33] | no (9x9 patches are too small to lose 20%) |
| window BA initialization (vs two-view 5-point at 2.5 deg), VO, 2 seeds | final 0.98 [0.88, 1.07], rotation 1.08 [1.01, 1.20] (V2_03); first 150 frames after init: position 1.00x, rotation 0.86x, initialized earlier on slow starts | adopted |
| window init with an attempt gate of 1.5 / 2.5 deg instead of 0.5, VO, 1 seed, 8 sequences | 1.16 / 1.01 vs two-view (noise) | 0.5 |
| intermediate keyframes taken over from the init window | never triggers (init happens before 2x the keyframe parallax) | removed |
| re-referencing the init window when the first frame's tracks die | the first keyframe is only ever given up on seasidetown (3-7 restarts in its first 20 frames) | not needed |
| V1_01's 3 deg early rotation error (both initializations) | not the focal length (3.1-3.3 deg for 0.97-1.02x); half is a constant 1.6 deg camera-frame roll offset vs the ground truth (hand-eye fit), specific to that sequence | open |
| our BA vs Ceres (pycolmap) on the same 359-keyframe problem, 1 thread, 10 iterations | ours 3.1 s, Ceres 4.8 (sparse Schur) - 6.6 s (dense Schur), same cost | - |

Where the error comes from: rotation error at the end is 90-100% *systematic* (the same for all start frames) on
almost every sequence; part of it is the orientation of the initial-section alignment (V1_01: 6 deg right after the
initial section, then < 1 deg of drift), part real drift (MH_01: 4 deg of yaw over 80 m, ~0.8 deg per 1% of focal
length error). Local rotation drift correlates best with image flow (rank correlation +0.39, fast motion), less with
the PnP residual scale (+0.18), sharpness (-0.15) or the number of tracked map points (-0.18).

Bugs that cost the most time: `scipy.linalg.solve(assume_a='pos')` (scipy 1.18.1 / Accelerate) corrupts the heap,
the crash then shows up anywhere (numba cache loading, GC at exit) - BA uses `cho_factor`; scoring only tracked frames
made losing track look *better* than tracking badly; fr2/desk ground truth interpolated over a 12 s mocap gap.

## Next

- initialization beyond the window BA: the depth-integrated (marginal) likelihood of slamfold instead of MAP depths,
  a trailing window that re-references when the first frame's tracks die (V1_01's rotation-heavy start), small-motion
  (DfUSMC-style) depth for the very first frames;
- cheaper global BA: it is 1-4 s for 300-600 keyframes (the LM loop in Python competes for the GIL with tracking;
  move the whole LM into numba, fewer iterations, or only the loop's region) - the remaining latency spikes;
- the chaos is mostly the random walk of the drift itself (max / min over start frames of a random walk's endpoint
  is naturally ~2x): it goes down with the per-step error, not with "smoother" decisions (none of the soft /
  adaptive variants reduced it measurably). Image flow is the best intrinsic predictor of where rotation drift is
  made: a candidate signal for more keyframes / more points under fast motion;
- voxel map instead of keyframes for reprojection and loops (search by projection and the old-point search are O(map));
  dense / semi-dense reconstruction;
- tracking through very fast motion (TartanAir) and re-joining new map segments reliably.
