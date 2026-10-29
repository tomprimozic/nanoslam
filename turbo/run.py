'''
python run.py household                                   # one sequence (names: datasets.all_names())
python run.py household --rerun                           # visualize (--rerun-save x.rrd: record to a file)
python run.py all --out runs/NAME [--starts 0,7,14]        # all sequences in parallel (+ start-frame ensemble)
python run.py summary runs/A runs/B                        # compare runs (metrics recomputed from trajectories;
                                                           # runs/A1+runs/A2 pools runs, e.g. of two seeds)
python run.py view runs/A/household@0 ...                  # rerun view of saved runs (alignment, errors over time)
options: --set name=value ... (override Config), --perturb SEED (scale all numeric parameters by 0.7-1.4x),
         --start S (skip the first S frames), --max-frames N, --only a,b (subset of sequences for 'all')
'''
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import sys
import json
import time
import shutil
import argparse
import subprocess
import faulthandler
import concurrent.futures
from pathlib import Path

import cv2
import numpy as np

import datasets
import evaluate
import geometry
from slam import SLAM, Config


def make_config(overrides):
  config = Config()
  for kv in overrides:
    k, v = kv.split('=', 1)
    if not hasattr(Config, k):
      raise ValueError(f'unknown config {k}')
    default = getattr(Config, k)
    setattr(config, k, v if isinstance(default, str) else type(default)(eval(v)))
  return config


def perturbed(seed, factor=1.4):
  '''every numeric Config value scaled by a random log-uniform factor in [1/factor, factor] (integers rounded)'''
  rng = np.random.default_rng(seed)
  overrides = []
  for k in sorted(vars(Config)):
    v = getattr(Config, k)
    if k.startswith('_') or isinstance(v, bool) or not isinstance(v, (int, float)) or k in ('ransac_seed',):
      continue
    f = np.exp(rng.uniform(-np.log(factor), np.log(factor)))
    x = 1 - (1 - v) * f if 'ncc' in k else v * f      # correlation thresholds: perturb the distance to 1
    overrides.append(f'{k}={max(1, round(x)) if isinstance(v, int) else round(x, 4)}')
  return overrides


def frame_range(x, metrics, n):
  '''the frames of a saved run (older runs without "start" went to the end of the sequence)'''
  start = metrics.get('start')
  return x[start:start + n] if start is not None else x[-n:]


def lost_flags(n, frames, lost):
  '''per-frame "tracking lost" flags from the per-frame stats'''
  flags = np.zeros(n, bool)
  flags[frames[lost > 0]] = True
  return flags


def run(name, config, out=None, start=0, max_frames=None, rerun=None, quiet=False):
  '''rerun: None, 'spawn' (a viewer) or the path of a recording'''
  cv2.setNumThreads(1)
  data = datasets.load(name)
  stop = len(data) if max_frames is None else min(len(data), start + max_frames)
  slam = SLAM(config, data.K, data.width, data.height)
  view = View(data, start, stop, slam.K, None if rerun == 'spawn' else rerun) if rerun else None
  t0 = time.perf_counter()
  latency = []
  for i, img in data.images(start, stop):
    t = time.perf_counter()
    slam.on_frame(i - start, img)
    latency.append(time.perf_counter() - t)
    if view is not None:
      view.frame(slam, i, img)
    if not quiet and (i - start) % 500 == 0:
      print(f'{name} frame {i}: tracks {slam.stats.tracks[-1]} mapped {slam.stats.mapped[-1]} lost {slam.lost}', flush=True)
  slam.finish()
  elapsed = time.perf_counter() - t0
  n = stop - start
  T_final, T_online = slam.poses(n, online=False), slam.poses(n)
  gt, gt_R = data.gt[start:stop, :3, 3], data.gt[start:stop, :3, :3]
  lost = lost_flags(n, slam.stats.frame, slam.stats.lost)
  m_final = evaluate.evaluate(T_final[:, :3, 3], gt, est_R=T_final[:, :3, :3], gt_R=gt_R, lost=lost)
  m_online = evaluate.evaluate(T_online[:, :3, 3], gt, est_R=T_online[:, :3, :3], gt_R=gt_R, lost=lost)
  events = {}
  for e in slam.events:
    events[e['type']] = events.get(e['type'], 0) + 1
  result = {
    'name': name, 'frames': n, 'start': start, 'seconds': elapsed, 'fps': n / elapsed,
    'final': {k: v for k, v in m_final.items() if not k.startswith('_')},
    'online': {k: v for k, v in m_online.items() if not k.startswith('_')},
    'keyframes': int(slam.map.kfs.alive.sum()), 'keyframes_created': len(slam.map.kfs),
    'points': int((slam.map.pts.state == 1).sum()), 'events': events,
    'timing': dict(slam.timing), 'counts': dict(slam.counts),
    'latency_ms': {'mean': 1e3 * np.mean(latency), 'p99': 1e3 * np.percentile(latency, 99), 'max': 1e3 * np.max(latency)},
  }
  if out is not None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'trajectory.npz', final=T_final[:, :3, 3], online=T_online[:, :3, 3], gt=gt, poses=T_final,
             online_poses=T_online,
             **{f'stats_{k}': getattr(slam.stats, k) for k in slam.stats._spec})
    (out / 'metrics.json').write_text(json.dumps(result | {'event_list': slam.events}, indent=1, default=float))
    evaluate.plot(m_online, gt, out / 'trajectory.png', title=f'{name} (live)')
  return result


def format_result(r):
  '''
  live: the poses as estimated at each frame (final = last 10% RMSE [m], rot [deg], drift, ate [m], scale); hindsight:
  the keyframe-relative poses at the end; keyframes alive / created; events: pre_keyframe (made before translation
  was established, rotation-only pose), init, resume (map continued after being lost), mapless (lost), reloc
  (relocalized), cull_kf, loop, merge
  '''
  o, f = r['online'], r['final']
  return (f"{r['name']:32s} live: final={o['final']:.3f} rot={o.get('rot', np.nan):.2f} drift={o['drift']:.3f} "
          f"ate={o['ate']:.3f} scale={o.get('scale', np.nan):.2f} | hindsight: final={f['final']:.3f} ate={f['ate']:.3f} | "
          f"tracked={o['tracked']:.2f} keyframes={r['keyframes']}/{r['keyframes_created']} fps={r['fps']:.1f} "
          f"max_latency={r['latency_ms']['max']:.0f}ms | {' '.join(f'{k}={v}' for k, v in r['events'].items())}")


def run_all(names, overrides, out, jobs, starts=(0,), max_frames=None):
  out.mkdir(parents=True, exist_ok=True)
  (out / 'config.json').write_text(json.dumps(overrides))
  # longest first, for better packing
  names = sorted(names, key=lambda n: -len(datasets.load(n)))
  # snapshot the code, so that sources can be edited while the benchmark runs
  code = out / 'code'
  code.mkdir(parents=True, exist_ok=True)
  for f in Path(__file__).parent.glob('*.py'):
    shutil.copy(f, code / f.name)
  os.environ['NANOSLAM_DATASETS'] = str(datasets.ROOT)
  # compile (numba cache) once before running in parallel
  subprocess.run([sys.executable, str(code / 'run.py'), 'household', '--max-frames', '300', '--quiet'],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def launch(job):
    n, start = job
    d = out / (n if len(starts) == 1 else f'{n}@{start}')
    cmd = [sys.executable, str(code / 'run.py'), n, '--out', str(d.resolve()), '--start', str(start), '--quiet']
    cmd += (['--max-frames', str(max_frames)] if max_frames else []) + (['--set', *overrides] if overrides else [])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'log.txt', 'w') as log:
      subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    f = d / 'metrics.json'
    return json.loads(f.read_text()) if f.exists() else {'name': n, 'error': (d / 'log.txt').read_text()[-300:]}
  with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
    return list(pool.map(launch, [(n, s) for s in starts for n in names]))


def summary(run_dirs, metrics=('final', 'rot', 'drift', 'ate', 'lost_episodes', 'lost%'), trajectory='online'):
  '''per-sequence table; metrics are recomputed from the saved trajectories (so evaluation changes apply)'''
  rows, gts = {}, {}
  for d in run_dirs:
    runs = {}
    for f in sorted(f for part in d.split('+') for f in Path(part).glob('*/log.txt')):   # A+B: pooled
      name = f.parent.name.split('@')[0]       # NAME@START: the same sequence from another start frame
      if not (f.parent / 'trajectory.npz').exists():
        runs.setdefault(name, []).append(None)
        continue
      if name not in gts:
        g = datasets.load(name).gt
        gts[name] = (g[:, :3, 3], g[:, :3, :3])
      t = np.load(f.parent / 'trajectory.npz')
      r = json.loads((f.parent / 'metrics.json').read_text())
      key = 'online_poses' if trajectory == 'online' else 'poses'
      gt = [frame_range(x, r, len(t['final'])) for x in gts[name]]
      R = (t[key][:, :3, :3], gt[1]) if key in t else (None, None)
      lost = lost_flags(len(t['final']), t['stats_frame'], t['stats_lost']) if 'stats_lost' in t else None
      runs.setdefault(name, []).append(evaluate.evaluate(t[trajectory], gt[0], est_R=R[0], gt_R=R[1], lost=lost)
                                       | {'fps': r['fps'], 'kfs': r['keyframes']})
    for name, rs in runs.items():   # median over start frames (a failed run counts as infinitely bad)
      if all(r is None for r in rs):
        rows.setdefault(name, {})[d] = None
        continue
      keys = [k for k in next(r for r in rs if r is not None) if not k.startswith('_')]
      rows.setdefault(name, {})[d] = {k: float(np.median([r[k] if r is not None else np.inf for r in rs]))
                                      for k in keys}
      # chaos: how much the result depends on the start frame (max / min over the starts)
      rows[name][d] |= {f'{k}_spread': float(np.max(v) / max(np.min(v), 1e-6))
                        for k in ('final', 'rot', 'drift', 'ate')
                        if len(v := [r[k] if r is not None else np.inf for r in rs]) > 1}
  names = sorted(rows, key=lambda n: datasets.all_names().index(n) if n in datasets.all_names() else 999)
  header = lambda title: print(f'\n{title:34s}' + ''.join(f' | {Path(d).name[-14:]:>14s}' for d in run_dirs))
  for metric in metrics:
    header(f"{metric} [{dict(rot='deg', lost_episodes='count', **{'lost%': '%'}).get(metric, 'm')}]")
    for n in names:
      cells = [f"{'FAIL':>14s}" if rows[n].get(d) is None and d in rows[n] else f"{'-':>14s}" if rows[n].get(d) is None
               else f"{rows[n][d][metric]:8.3f} ({rows[n][d]['tracked']:.2f})" for d in run_dirs]
      print(f'{n:34s}' + ''.join(f' | {c}' for c in cells))
    geo = lambda v: np.exp(np.mean(np.log(np.maximum(v, 1e-4))))
    if metric.startswith('lost'):     # counts, mostly zero: plain mean
      vals = [[rows[n][d][metric] if rows[n].get(d) else np.nan for n in names] for d in run_dirs]
      print(f"{'mean':34s}" + ''.join(f' | {np.nanmean(v):14.2f}' for v in vals))
      continue
    for agg, fn, key in [('median', np.median, metric), ('geo. mean', geo, metric),
                         ('spread over starts (geo. mean)', geo, f'{metric}_spread')]:
      vals = [[rows[n][d].get(key, np.nan) if rows[n].get(d) else np.inf for n in names] for d in run_dirs]
      print(f'{agg:34s}' + ''.join(f' | {fn(v):14.4f}' for v in vals))
    # paired comparison with the first run: per-sequence ratio, geometric mean with a bootstrap 95% interval
    cells = []
    for d in run_dirs[1:]:
      r = np.array([rows[n][d][metric] / rows[n][run_dirs[0]][metric] for n in names
                    if rows[n].get(d) and rows[n].get(run_dirs[0]) and np.isfinite(rows[n][d][metric])
                    and np.isfinite(rows[n][run_dirs[0]][metric]) and rows[n][run_dirs[0]][metric] > 0])
      if len(r) == 0:
        cells.append(f"{'-':>14s}")
        continue
      boot = np.exp(np.log(r)[np.random.default_rng(0).integers(0, len(r), (2000, len(r)))].mean(axis=1))
      lo, hi = np.percentile(boot, [2.5, 97.5])
      cells.append(f'{np.exp(np.log(r).mean()):.2f} [{lo:.2f},{hi:.2f}] {int((r < 1).sum())}/{int((r > 1).sum())}')
    if cells:
      print(f"{'vs first: ratio [95% CI] bet/worse':34s} | {'':>14s}" + ''.join(f' | {c:>14s}' for c in cells))
  header('fps / keyframes')
  for n in names:
    print(f'{n:34s}' + ''.join(f" | {rows[n][d]['fps']:6.1f} {rows[n][d]['kfs']:5.0f}  " if rows[n].get(d) else
                               f" | {'-':>14s}" for d in run_dirs))


def view(run_dirs, save=None):
  '''
  rerun view of saved runs (NAME or NAME@START dirs): ground truth, the estimate aligned on the initial section (10%
  and 20% coverage; the section itself highlighted), live and hindsight trajectories, errors over time
  '''
  import rerun as rr
  rr.init('nanoslam-turbo-view', spawn=save is None)
  if save is not None:
    rr.save(save)
  for d in map(Path, run_dirs):
    name = d.name.split('@')[0]
    t = np.load(d / 'trajectory.npz')
    gt = frame_range(datasets.load(name).gt, json.loads((d / 'metrics.json').read_text()), len(t['final']))[:, :3, 3]
    rr.log(f'{d.name}/gt', rr.LineStrips3D([gt[np.isfinite(gt).all(axis=1)]], radii=0.004, colors=[150, 150, 150]),
           static=True)
    for coverage, color in [(0.1, [0, 120, 255]), (0.2, [255, 140, 0])]:
      for kind in ['online', 'final']:
        m = evaluate.evaluate(t[kind], gt, coverage=coverage)
        a = m['_aligned']
        path = f'{d.name}/coverage_{coverage:.1f}/{kind}'
        rr.log(f'{path}/trajectory', rr.LineStrips3D([a[np.isfinite(a).all(axis=1)]], radii=0.003,
                                                      colors=color if kind == 'online' else [c // 2 for c in color]),
               static=True)
        if kind == 'online':
          rr.log(f'{d.name}/coverage_{coverage:.1f}/initial_section', rr.LineStrips3D([gt[m['_init']]], radii=0.012,
                                                                                       colors=[0, 200, 0]), static=True)
        for i in range(0, len(gt), 2):
          rr.set_time('frame', sequence=i)
          if np.isfinite(m['_err'][i]):
            rr.log(f'error/{d.name}/coverage_{coverage:.1f}/{kind}', rr.Scalars(float(m['_err'][i])))
          if kind == 'online' and np.isfinite(a[i]).all():
            rr.log(f'{path}/camera', rr.Points3D([a[i]], radii=0.03, colors=color))
            rr.log(f'{d.name}/gt_camera', rr.Points3D([gt[i]], radii=0.03, colors=[150, 150, 150]))
        rr.reset_time()
        print(f"{d.name} coverage {coverage:.1f} {kind:6s}: init section {m['init_frames']} frames, "
              f"final {m['final']:.3f} drift {m['drift']:.3f} ate {m['ate']:.3f}")


class View:
  '''
  live rerun view, in metres with the ground truth's up as +Z: the SLAM (camera with its image, path with hindsight,
  keyframes, map points coloured from their keyframe's image) Sim3-aligned into the ground truth on the initial section
  (as in the evaluation) and then frozen, so what follows is the drift. Sizes are fractions of the ground truth's
  horizontal span; the 3D view starts behind the first camera, looking where it looks (level)
  '''
  SIZES = {'gt': 0.001, 'path': 0.001, 'keyframes': 0.002, 'points': 0.0006, 'image': 0.04}

  def __init__(self, data, start, stop, K, save=None):
    import rerun as rr
    import rerun.blueprint as rrb
    self.rr, self.data, self.start, self.K = rr, data, start, K
    gt = data.gt[start:stop]
    up = data.up / np.linalg.norm(data.up)
    v, c = np.cross(up, [0, 0, 1]), float(up[2])        # the rotation taking up to +Z
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    self.R_up = np.eye(3) + V + V @ V / (1 + c) if c > -1 + 1e-9 else np.diag([1.0, -1.0, -1.0])
    ok = np.isfinite(gt[:, 0, 3])
    self.gt = np.where(ok[:, None], gt[:, :3, 3] @ self.R_up.T, np.nan)
    span = float(np.max(np.ptp(self.gt[ok, :2], axis=0)))
    self.size = {k: f * span for k, f in self.SIZES.items()}
    self.align_until = evaluate.initial_section(self.gt) - 1      # (frames since start)
    self.sim = None                    # SLAM -> display: (s, R, t)
    self.n_kfs = 0
    self.kf_rgb = {}                   # keyframe -> RGB image (half resolution), to colour its points
    self.colors = np.zeros((0, 3), np.uint8)
    self.colored = np.zeros(0, bool)
    rr.init('nanoslam-turbo', spawn=save is None)
    if save is not None:
      rr.save(save)
    i0 = int(np.argmax(ok))
    C0, f = self.gt[i0], self.R_up @ gt[i0, :3, 2]
    f = np.array([f[0], f[1], 0.0]) / max(np.linalg.norm(f[:2]), 1e-9)      # (looking direction, level)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
      rrb.Spatial3DView(origin='world', eye_controls=rrb.EyeControls3D(
        position=C0 - 0.3 * span * f + [0, 0, 0.15 * span], look_target=C0 + 0.5 * span * f, eye_up=[0, 0, 1])),
      rrb.Spatial2DView(origin='world/camera/image'), column_shares=[3, 1]), collapse_panels=True))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log('world/gt', rr.LineStrips3D([self.gt[ok]], radii=self.size['gt'], colors=[150, 150, 150]), static=True)
    rr.log('world/camera/image', rr.Pinhole(image_from_camera=K, width=data.width, height=data.height,
                                            image_plane_distance=self.size['image']), static=True)

  def to_display(self, X):
    s, R, t = self.sim
    return s * X @ R.T + t

  def frame(self, slam, i, img):
    from geometry import camera_center, inv_pose
    from map import MAPPED
    rr, m, f = self.rr, slam.map, i - self.start
    rr.set_time('frame', sequence=i)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (x, y), mapped in zip(slam.tr_uv, slam.mapped_tracks()):      # green: tracked map points, red: candidates
      cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0) if mapped else (255, 0, 0), -1)
    rr.log('world/camera/image', rr.Image(vis).compress(80))
    for k in range(self.n_kfs, len(m.kfs)):           # new keyframes: keep their colours
      if m.kfs.frame[k] == f:
        rgb = self.data.load_color(i)
        self.kf_rgb[k] = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    self.n_kfs = len(m.kfs)
    if not slam.initialized:
      return
    path = slam.poses(f + 1, online=False)[:, :3, 3]
    valid = np.isfinite(path).all(axis=1) & np.isfinite(self.gt[:f + 1]).all(axis=1)
    if valid.sum() < 10:
      return
    if self.sim is None or f <= self.align_until or valid.sum() < 30:     # (then frozen: what follows is the drift)
      s, R, t = geometry.umeyama(self.gt[:f + 1][valid], path[valid])
      self.sim = (s, R, t)
    T_wc = inv_pose(slam.T_cw)
    rr.log('world/camera', rr.Transform3D(translation=self.to_display(T_wc[:3, 3]), mat3x3=self.sim[1] @ T_wc[:3, :3]))
    if f % 5:
      return
    ok = np.isfinite(path).all(axis=1)
    rr.log('world/slam/path', rr.LineStrips3D([self.to_display(path[ok])], radii=self.size['path'], colors=[0, 120, 255]))
    alive = np.flatnonzero(m.kfs.alive)
    rr.log('world/slam/keyframes', rr.Points3D(self.to_display(camera_center(m.kfs.T[alive])),
                                               radii=self.size['keyframes'], colors=[255, 140, 0]))
    pts = np.flatnonzero(m.pts.state == MAPPED)
    if len(self.colors) < len(m.pts):
      self.colors = np.r_[self.colors, np.full((len(m.pts) - len(self.colors), 3), 160, np.uint8)]
      self.colored = np.r_[self.colored, np.zeros(len(m.pts) - len(self.colored), bool)]
    new = pts[~self.colored[pts]]
    for k in np.unique(m.pts.ref[new]):                # colour from the reference keyframe's image
      if k in self.kf_rgb:
        sel = new[m.pts.ref[new] == k]
        uv = np.clip(np.round(m.pts.ref_uv[sel] / 2).astype(int), 0, np.array(self.kf_rgb[k].shape[1::-1]) - 1)
        self.colors[sel] = self.kf_rgb[k][uv[:, 1], uv[:, 0]]
        self.colored[sel] = True
    for k in [k for k in self.kf_rgb if not m.kfs.alive[k]]:
      del self.kf_rgb[k]
    rr.log('world/slam/points', rr.Points3D(self.to_display(m.pts.X[pts]), radii=self.size['points'],
                                            colors=self.colors[pts]))


def main():
  parser = argparse.ArgumentParser('nanoslam turbo')
  parser.add_argument('dataset', help="a dataset name, 'all' or 'summary'")
  parser.add_argument('runs', nargs='*', help='run directories (summary)')
  parser.add_argument('--out', type=Path)
  parser.add_argument('--set', nargs='*', default=[])
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--starts', help='"all": comma separated start frames (an ensemble, summarized by the median)')
  parser.add_argument('--max-frames', type=int)
  parser.add_argument('--jobs', type=int, default=os.cpu_count())
  parser.add_argument('--only', help='comma separated subset of datasets for "all"')
  parser.add_argument('--rerun', action='store_true', help='visualize with rerun')
  parser.add_argument('--rerun-save', help='write the rerun recording to this .rrd file instead')
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--trajectory', default='online', help="summary: 'online' (live poses) or 'final' (hindsight)")
  parser.add_argument('--perturb', type=int, help='perturb all numeric parameters randomly (seed)')
  args = parser.parse_args()
  if args.dataset == 'summary':
    return summary(args.runs, trajectory=args.trajectory)
  if args.dataset == 'view':
    return view(args.runs, args.rerun_save)
  if args.perturb is not None:
    args.set = perturbed(args.perturb) + args.set
    print('perturbed config:', ' '.join(args.set))
  if args.dataset == 'all':
    names = args.only.split(',') if args.only else datasets.all_names()
    starts = [int(x) for x in args.starts.split(',')] if args.starts else [args.start]
    results = run_all(names, args.set, args.out, args.jobs, starts, args.max_frames)
    for r in results:
      print(format_result(r) if 'error' not in r else f"{r['name']:32s} ERROR {r['error']}")
    return summary([str(args.out)])
  faulthandler.enable()
  rerun = args.rerun_save or ('spawn' if args.rerun else None)
  r = run(args.dataset, make_config(args.set), args.out, args.start, args.max_frames, rerun)
  print(format_result(r))
  print('timing: ' + ' '.join(f'{k}={v:.1f}s' for k, v in r['timing'].items()))


if __name__ == '__main__':
  main()
