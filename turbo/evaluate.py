'''
Trajectory evaluation focused on drift.

The estimate is Sim3-aligned to the ground truth on the *initial section* only: the shortest prefix (from the first
tracked frame) whose ground truth covers at least `coverage` (20%) of the whole trajectory's extent along each of its
two main axes (e.g. a trajectory spanning 20 m x 10 m is aligned on the prefix that spans 4 m x 2 m). Everything
after that is extrapolation, so the error at the end measures the accumulated drift. By default the poses are the
live ones (as estimated at that frame), not corrected with hindsight by later loop closures / the final BA.

  final     RMSE over the last 10% of frames, initial-section alignment (the headline number)
  rot       rotation drift (deg, RMS over the last 10% of frames): rotation since the end of the initial section,
            estimated vs true (does not depend on the alignment)
  tracked   fraction of frames with a pose; lost_episodes / lost%: tracking losses after initialization
  drift     RMSE over all frames, initial-section alignment
  drift%    drift relative to the ground truth path length
  ate       standard ATE: RMSE after aligning the whole trajectory (for reference)
  scale     scale of the final section relative to the initial section (1 = no scale drift)

Dense mapping (dense()): every keyframe depth map against ground-truth depth, and the fused surface / dense points
against the ground-truth cloud, each keyframe scaled to metres by its own sparse points (SLAM drift not counted).
'''
import numpy as np

import geometry


def principal_axes(p):
  c = p.mean(axis=0)
  _, _, VT = np.linalg.svd(p - c, full_matrices=False)
  return c, VT


def initial_section(gt, coverage=0.2, min_frames=10):
  '''number of frames of the shortest prefix that covers `coverage` of the full extent along the 2 main axes'''
  gt = gt[np.isfinite(gt).all(axis=1)]
  c, VT = principal_axes(gt)
  proj = (gt - c) @ VT[:2].T
  extent = proj.max(axis=0) - proj.min(axis=0)
  lo = np.minimum.accumulate(proj, axis=0)
  hi = np.maximum.accumulate(proj, axis=0)
  covered = np.all(hi - lo >= coverage * extent, axis=1)
  n = int(np.argmax(covered)) + 1 if covered.any() else len(gt)
  return max(n, min_frames)


def align(est, gt):
  s, R, t = geometry.umeyama(gt, est)
  return lambda X: s * X @ R.T + t, s


def hold(est):
  '''fill untracked frames (after the first tracked one) with the last estimate'''
  ok = np.isfinite(est).all(axis=1)
  last = np.maximum.accumulate(np.where(ok, np.arange(len(est)), -1))
  return np.where((last >= 0)[:, None], est[np.maximum(last, 0)], np.nan)


def lost_stats(est, lost=None):
  '''
  tracking losses after the first tracked frame: number of episodes, fraction of frames (`lost`: per-frame flags of the
  system, e.g. when it keeps reporting rotation-only poses while lost; default: frames without an estimate)
  '''
  ok = np.isfinite(est).all(axis=1)
  if lost is not None:
    ok &= ~lost
  if not ok.any():
    return {'lost_episodes': 0, 'lost%': 100.0}
  ok = ok[np.argmax(ok):]
  return {'lost_episodes': int(np.sum(ok[:-1] & ~ok[1:])), 'lost%': float(100 * (~ok).mean())}


def evaluate(est, gt, coverage=0.2, final_frac=0.1, est_R=None, gt_R=None, align_est=None, lost=None):
  '''
  est, gt: (N, 3) camera centres (rows NaN where not tracked / no ground truth). Lost frames are scored with the last
  estimate (as a system that keeps reporting its last pose would be), so losing track is not better than tracking badly
  '''
  has_gt = np.isfinite(gt).all(axis=1)
  tracked = np.isfinite(est).all(axis=1) & has_gt
  lost = lost_stats(est, lost)
  if est_R is not None:
    est_R = hold(est_R.reshape(len(est_R), 9)).reshape(-1, 3, 3)
  est = hold(est)
  valid = np.isfinite(est).all(axis=1) & has_gt
  out = {'tracked': float(tracked.sum() / has_gt.sum())} | lost
  if valid.sum() < 10:
    return out | {'drift': np.inf, 'final': np.inf, 'ate': np.inf}
  idx = np.flatnonzero(valid)
  first = idx[0]
  g = gt[first:][has_gt[first:]]
  n_init = np.flatnonzero(has_gt[first:])[initial_section(g, coverage) - 1] + 1
  init = idx[idx < first + n_init]
  # the section starts where the estimate starts to move (before, positions are held: rotation-only poses before
  # translation was established)
  moving = idx[np.linalg.norm(est[idx] - est[first], axis=1) > 1e-9]
  if len(moving) and moving[0] > first:
    g2 = gt[moving[0]:][has_gt[moving[0]:]]
    n2 = np.flatnonzero(has_gt[moving[0]:])[initial_section(g2, coverage) - 1] + 1
    init = idx[(idx >= moving[0]) & (idx < moving[0] + n2)]
  if len(init) < 3:
    init = idx[:3] if not len(moving) else moving[:3]
  if align_est is not None:   # alignment from other estimates of the initial section (e.g. hindsight poses)
    a = hold(align_est)
    init_a = init[np.isfinite(a[init]).all(axis=1)]
    to_gt, s_init = align(a[init_a], gt[init_a])
  else:
    to_gt, s_init = align(est[init], gt[init])
  err = np.full(len(gt), np.nan)
  err[idx] = np.linalg.norm(to_gt(est[idx]) - gt[idx], axis=1)
  final = idx[idx >= int(len(gt) * (1 - final_frac))]
  to_gt_all, _ = align(est[idx], gt[idx])
  err_all = np.linalg.norm(to_gt_all(est[idx]) - gt[idx], axis=1)
  path = np.nansum(np.linalg.norm(np.diff(gt[first:], axis=0), axis=1))
  scale = np.nan
  if len(final) >= 3 and np.ptp(gt[final], axis=0).max() > 0.05 * np.ptp(gt[idx], axis=0).max():
    _, s_final = align(est[final], gt[final])
    scale = s_final / s_init
  drift = float(np.sqrt(np.nanmean(err[idx] ** 2)))
  rot = np.nan
  if est_R is not None and len(final):
    # rotation drift: rotation from the end of the initial section to the final section, estimated vs true (deg;
    # independent of the alignment)
    a = init[-1]
    R_est = est_R[a].T @ est_R[final]
    R_gt = gt_R[a].T @ gt_R[final]
    rot = float(np.sqrt(np.mean(np.degrees(geometry.rotation_angle(np.swapaxes(R_gt, 1, 2) @ R_est)) ** 2)))
  return out | {
    'drift': drift,
    'final': float(np.sqrt(np.mean(err[final] ** 2))) if len(final) else np.inf,
    'drift%': 100 * drift / path,
    'ate': float(np.sqrt(np.mean(err_all ** 2))),
    'rot': rot,
    'scale': float(scale),
    'init_frames': int(n_init),
    'path': float(path),
    '_err': err,
    '_aligned': to_gt(np.where(valid[:, None], est, np.nan)),
    '_init': init,
  }


def plot(m, gt, path, title=''):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  c, VT = principal_axes(gt[np.isfinite(gt).all(axis=1)])
  g = (gt - c) @ VT[:2].T
  a = (m['_aligned'] - c) @ VT[:2].T
  fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(14, 6))
  ax.plot(g[:, 0], g[:, 1], color='gray', lw=1, label='ground truth')
  ax.plot(a[:, 0], a[:, 1], color='tab:blue', lw=1, label='estimate')
  ax.plot(a[m['_init'], 0], a[m['_init'], 1], color='tab:green', lw=3, label='initial (aligned) section')
  ax.set_aspect('equal')
  ax.legend()
  ax.set_title(title)
  ax_err.plot(m['_err'], color='tab:blue')
  ax_err.set_xlabel('frame')
  ax_err.set_ylabel('position error [m] (initial-section alignment)')
  ax_err.grid(True)
  fig.tight_layout()
  fig.savefig(path, dpi=80)
  plt.close(fig)


def dense(mapper, depth, step=4, thresholds=(0.02, 0.05)):
  '''
  dense mapping against ground-truth depth (depth: frame -> metres, 0 unknown, or None). Every keyframe map is scaled to
  metres by its sparse points against the ground truth (so SLAM scale drift is not counted as reconstruction error).
  Per keyframe (over the pixels with ground truth): completeness, acc10 (fraction of all within 10% in inverse depth),
  precision (of the estimates within 10%), outliers (beyond 25%); whether sigma is informative: the precision of the
  more certain half (by relative sigma) and the rank correlation of sigma and error. 3D: the fused surface and the
  dense points against the ground-truth cloud (every keyframe's ground truth placed like its map): precision
  (reconstruction -> ground truth) and coverage (ground truth -> reconstruction) within the thresholds [m]; the
  surface also over a range of certainty thresholds (surface_curve: precision 2 / 5 cm, coverage 2 / 5 cm)
  '''
  import scipy.spatial
  from dense import backproject
  rows, gt_pts, mpu = [], [], []
  for r in mapper.records.values():
    gt = depth(r.frame)
    if gt is None:
      continue
    ui, vi = np.round(r.uv).astype(int).T
    h, w = gt.shape
    ok = (r.z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    g = gt[vi[ok], ui[ok]]
    if (g > 0).sum() < 5:
      continue
    sc = float(np.median(g[g > 0] / r.z[ok][g > 0]))       # metres per map unit (as computed)
    gt = gt[::2, ::2][:r.rho.shape[0], :r.rho.shape[1]]
    valid = gt > 0
    est = valid & np.isfinite(r.rho) & (r.rho > 0)
    err = np.abs(r.rho[est] / sc - 1 / gt[est]) * gt[est]
    n = max(valid.sum(), 1)
    # is sigma informative: precision of the more certain half (relative sigma), rank correlation sigma ~ error
    rel_sig = r.sig[est] / r.rho[est]
    certain = rel_sig <= np.median(rel_sig) if est.any() else est[est]
    rank = lambda x: np.argsort(np.argsort(x))
    rows.append([est.sum() / n, (err < 0.1).sum() / n, np.mean(err < 0.1) if est.any() else np.nan,
                 np.mean(err > 0.25) if est.any() else np.nan, np.mean(err[certain] < 0.1) if est.any() else np.nan,
                 np.corrcoef(rank(rel_sig), rank(err))[0, 1] if est.sum() > 10 else np.nan])
    X, _ = backproject(mapper.Kh, r.T, np.where(valid, sc / (np.where(valid, gt, 1) * r.scale), np.nan), step)
    gt_pts.append(X)
    mpu.append(sc / r.scale)
  if not rows:
    return {}
  rows = np.array(rows)
  out = {'keyframes': len(rows), 'completeness': float(np.mean(rows[:, 0])), 'acc10': float(np.mean(rows[:, 1])),
         'precision': float(np.nanmean(rows[:, 2])), 'outliers': float(np.nanmean(rows[:, 3])),
         'precision_certain_half': float(np.nanmean(rows[:, 4])),
         'sigma_error_rank_corr': float(np.nanmean(rows[:, 5]))}
  gt_pts = np.concatenate(gt_pts)
  m = float(np.median(mpu))
  tree = scipy.spatial.cKDTree(gt_pts)
  def score(P):
    d_acc = tree.query(P, workers=-1)[0] * m
    d_cov = scipy.spatial.cKDTree(P).query(gt_pts, workers=-1)[0] * m
    return [float(np.mean(d < t)) for d in (d_acc, d_cov) for t in thresholds]
  for name, (P, _) in [('surface', mapper.surface()), ('points', mapper.dense_points())]:
    if len(P) == 0:
      continue
    out[f'{name}_n'] = len(P)
    s = score(P)
    for i, t in enumerate(thresholds):
      out[f'{name}_precision_{int(100 * t)}cm'] = s[i]
      out[f'{name}_coverage_{int(100 * t)}cm'] = s[len(thresholds) + i]
  # the surface's precision / coverage curve over the certainty threshold (fused sigma, or observations with unit
  # weights): changes that rescale sigma move along it, so compare at equal coverage
  ts = mapper.tsdf
  levels = ([1 / (s * ts.voxel) ** 2 for s in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)] if ts.sigma_weights
            else [1, 2, 3, 4, 6, 8, 12, 16, 24])
  out['surface_curve'] = [score(P) for w in levels if len(P := ts.surface(min_weight=w)[0]) > 0]
  return out
