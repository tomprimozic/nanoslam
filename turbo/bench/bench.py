'''
Baseline monocular SLAM systems on turbo's datasets, in turbo's run format (trajectory.npz + metrics.json + log.txt per
run), so that `run.py summary` and evaluate.py score every system with the same code.

  python bench/bench.py run SYSTEM NAME [--start S] --out DIR      # one run
  python bench/bench.py all SYSTEM --out runs/X [--starts 0,5,10,15,20] [--only a,b] [--jobs N]
  python bench/bench.py convert SYSTEM NAME DIR [--start S]        # (re)convert the raw outputs in DIR

Systems: the patched checkouts in $BENCH_LIB (default ~/my/lib, patches in bench/patches, built by
bench/build_baselines.sh). Every system gets the same frames (datasets.py: the ground truth's time span) from the start
frame on, as a "timestamp path" list; zip archives are extracted, and images whose distortion the system cannot model
(TUM fr1 / fr2's k3, monoVO's FOV fisheye) undistorted by datasets.py, into $BENCH_CACHE (default
~/.cache/nanoslam-bench). Each driver writes
  raw_live.txt     the pose of every frame as estimated at that frame (TUM format, T_wc)       -> online_poses
  raw_beliefs.txt  after every frame, the keyframes whose pose changed ("S <time>", then "<keyframe time> <T_wc>
                   [map]"; culled keyframes follow their parent where the system supports it)
The converter turns the beliefs into turbo's belief updates (upd_*: the system's estimate of every past frame over
time: a frame moves with the keyframe before it, at its live pose relative to that keyframe); the last belief is the
hindsight trajectory (poses). Frames without a pose are NaN (lost frames are scored with the last pose).
'''
import os
import re
import sys
import json
import time
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cv2          # noqa: E402
import datasets     # noqa: E402
import evaluate     # noqa: E402

LIB = Path(os.environ.get('BENCH_LIB', Path.home() / 'my/lib'))
CACHE = Path(os.environ.get('BENCH_CACHE', Path.home() / '.cache/nanoslam-bench'))


# ---------------------------------------------------------------------------------------------------- images, cameras

def fps(data):
  return float(1 / np.median(np.diff(data.timestamps)))


def camera(data):
  '''
  what the C++ systems get: ('raw', K, radtan distortion or None) for images they can read and model as stored,
  else ('undistorted', K, None): the images undistorted by datasets.py (pinhole data.K)
  '''
  if isinstance(data, datasets.EuRoC):
    i = data.camera['intrinsics']
    return 'raw', datasets.make_K(*i), list(data.camera['distortion_coefficients'])
  if data.maps is None:
    return 'raw', data.K, None
  return 'undistorted', data.K, None


def image_files(data, kind):
  '''image paths on disk: as stored (zip members extracted to the cache) or undistorted (cached PNGs)'''
  if kind == 'undistorted':
    d = CACHE / 'undistorted' / data.name
    files = [d / f'{i:06d}.png' for i in range(len(data))]
    if not all(f.exists() for f in files):
      d.mkdir(parents=True, exist_ok=True)
      for i, img in data.images():
        cv2.imwrite(str(files[i]), img)
    return files
  if not any(isinstance(f, datasets.ZipMember) for f in data.files):
    return [Path(f) for f in data.files]
  d = CACHE / 'raw' / data.name
  files = [d / Path(f.member).name for f in data.files]
  if not all(f.exists() for f in files):
    d.mkdir(parents=True, exist_ok=True)
    for f, out in zip(data.files, files):
      if not out.exists():
        out.write_bytes(f.read())
  return files


def frames_file(data, files, start, path):
  with open(path, 'w') as f:
    for i in range(start, len(data)):
      f.write(f'{data.timestamps[i]:.9f} {files[i]}\n')


# ---------------------------------------------------------------------------------------------------- systems

def orb_settings(data, version, K, dist, d):
  '''the settings shipped with ORB-SLAM where there are some (TUM1/2/3, EuRoC, KITTI), else TUM1.yaml with this camera'''
  ex = LIB / f'ORB_SLAM{version}/Examples/Monocular'
  if isinstance(data, datasets.TUM) and camera(data)[0] == 'raw':
    return ex / f'{data.name.split("_")[0].replace("freiburg", "TUM")}.yaml'
  if isinstance(data, datasets.EuRoC):
    return ex / 'EuRoC.yaml'
  elif isinstance(data, datasets.KITTI):
    n = int(data.name.split('_')[1])
    return ex / ('KITTI00-02.yaml' if n <= 2 else 'KITTI03.yaml' if n == 3 else 'KITTI04-12.yaml')
  cam = 'Camera1' if version == 3 else 'Camera'
  values = {f'{cam}.fx': K[0, 0], f'{cam}.fy': K[1, 1], f'{cam}.cx': K[0, 2], f'{cam}.cy': K[1, 2],
            f'{cam}.k1': 0.0, f'{cam}.k2': 0.0, f'{cam}.p1': 0.0, f'{cam}.p2': 0.0, f'{cam}.k3': 0.0,
            'Camera.width': data.width, 'Camera.height': data.height, 'Camera.fps': int(round(fps(data)))}
  lines = []
  for line in (ex / 'TUM1.yaml').read_text().splitlines():
    key = line.split(':')[0].strip()
    if key in values:
      line = f'{key}: {values.pop(key)}'
    lines.append(line)
  if version == 3:      # (TUM1.yaml has no k3 / size entries in every version)
    lines += [f'{k}: {v}' for k, v in values.items() if k in ('Camera.width', 'Camera.height')]
  (d / 'orb.yaml').write_text('\n'.join(lines) + '\n')
  return d / 'orb.yaml'


def setup_orb(version, loop=True):
  def setup(data, start, d):
    kind, K, dist = camera(data)
    frames_file(data, image_files(data, kind), start, d / 'frames.txt')
    exe = LIB / f'ORB_SLAM{version}/Examples/Monocular/mono_bench'
    voc = LIB / f'ORB_SLAM{version}/Vocabulary/ORBvoc.txt'
    cmd = [str(exe), str(voc), str(orb_settings(data, version, K, dist, d)), str(d / 'frames.txt'), str(d / 'raw')]
    # without loop closing: loop detection switched off in the bench patch (relocalization still uses the keyframe database)
    return cmd if loop else ['env', f'ORB_SLAM{version}_NO_LOOP=1'] + cmd
  return setup


def setup_dso(data, start, d):
  kind, K, dist = camera(data)
  w, h = data.width, data.height
  if dist is not None and len(dist) == 4:     # DSO's usual EuRoC setup: RadTan, cropped to 640 x 480
    calib = f'RadTan {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]} {" ".join(map(str, dist))}\n{w} {h}\ncrop\n640 480\n'
  elif w % 16 or h % 16:     # DSO's pyramid needs halvable sizes (KITTI, ETH3D): the same K, right / bottom cropped
    ow, oh = w // 16 * 16, h // 16 * 16
    out = f'{K[0, 0] / ow} {K[1, 1] / oh} {(K[0, 2] + 0.5) / ow} {(K[1, 2] + 0.5) / oh} 0'
    calib = f'Pinhole {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]} 0\n{w} {h}\n{out}\n{ow} {oh}\n'
  else:
    calib = f'Pinhole {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]} 0\n{w} {h}\nnone\n{w} {h}\n'
  (d / 'camera.txt').write_text(calib)
  frames_file(data, image_files(data, kind), start, d / 'frames.txt')
  return [str(LIB / 'dso/build/bin/dso_bench'), str(d / 'camera.txt'), str(d / 'frames.txt'), str(d / 'raw')]


def setup_ov2slam(loop):
  def setup(data, start, d):
    kind, K, dist = camera(data)
    dist = dist or [0.0] * 4
    # the authors' "accurate" monocular EuRoC settings, with this camera
    s = (LIB / 'ov2slam/parameters_files/accurate/euroc/euroc_mono.yaml').read_text()
    values = {'left_nwidth': data.width, 'left_nheight': data.height, 'fxl': K[0, 0], 'fyl': K[1, 1], 'cxl': K[0, 2],
              'cyl': K[1, 2], 'k1l': dist[0], 'k2l': dist[1], 'p1l': dist[2], 'p2l': dist[3]}
    lines = []
    for line in s.splitlines():
      key = line.split(':')[0].strip()
      if key.startswith('Camera.') and key[7:] in values:
        line = f'{key}: {values[key[7:]]}'
      elif key == 'force_realtime':
        line = 'force_realtime: 0'
      elif key == 'buse_loop_closer':
        line = f'buse_loop_closer: {int(loop)}'
      lines.append(line)
    (d / 'params.yaml').write_text('\n'.join(lines) + '\n')
    frames_file(data, image_files(data, kind), start, d / 'frames.txt')
    return [str(LIB / 'ov2slam/standalone/build/ov2slam_bench'), str(d / 'params.yaml'), str(d / 'frames.txt'),
            str(d / 'raw')]
  return setup


SYSTEMS = {
  'orbslam2': setup_orb(2),
  'orbslam2_vo': setup_orb(2, loop=False),
  'orbslam3': setup_orb(3),
  'dso': setup_dso,
  'ov2slam': setup_ov2slam(loop=True),
  'ov2slam_vo': setup_ov2slam(loop=False),
}


# ---------------------------------------------------------------------------------------------------- conversion

def parse_poses(rows):
  '''rows of "time tx ty tz qx qy qz qw [extra...]" -> times, (N, 4, 4) T_wc, (N, k) extra (malformed rows skipped)'''
  rows = [r for r in rows if len(r) >= 8]
  if not rows:
    return np.zeros(0), np.zeros((0, 4, 4)), np.zeros((0, 0))
  width = max(set(map(len, rows)), key=[len(r) for r in rows].count)
  x = np.array([[float(v) for v in r] for r in rows if len(r) == width])
  x = x[np.isfinite(x).all(axis=1) & (np.linalg.norm(x[:, 4:8], axis=1) > 0.5)]
  T = np.tile(np.eye(4), (len(x), 1, 1))
  T[:, :3, :3] = Rotation.from_quat(x[:, 4:8]).as_matrix()
  T[:, :3, 3] = x[:, 1:4]
  return x[:, 0], T, x[:, 8:]


def read_poses(path):
  if not path.exists():
    return parse_poses([])
  return parse_poses([line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')])


def read_beliefs(path):
  '''raw_beliefs.txt -> [(time, keyframe times, (n, 4, 4) T_wc, maps or None)]'''
  out = []
  if not path.exists():
    return out
  block = None
  for line in path.read_text().splitlines():
    if line.startswith('S '):
      if block is not None:
        out.append((block[0], *parse_poses(block[1])))
      block = (float(line.split()[1]), [])
    elif line.strip() and block is not None:
      block[1].append(line.split())
  if block is not None:
    out.append((block[0], *parse_poses(block[1])))
  return [(t, kt, T, extra[:, 0] if extra.shape[1] else None) for t, kt, T, extra in out]


def nearest(t, timestamps, tol=2e-3):
  '''frame index of each time (-1 if none within tol)'''
  if len(t) == 0:
    return np.zeros(0, int)
  i = np.clip(np.searchsorted(timestamps, t), 1, len(timestamps) - 1)
  i = np.where(np.abs(timestamps[i - 1] - t) < np.abs(timestamps[i] - t), i - 1, i)
  return np.where(np.abs(timestamps[i] - t) < tol, i, -1)


def to_frames(t, T, timestamps):
  '''(N, 4, 4) per frame (NaN where none) from timestamped poses; the last pose wins for a repeated frame'''
  out = np.full((len(timestamps), 4, 4), np.nan)
  i = nearest(t, timestamps)
  out[i[i >= 0]] = T[i >= 0]
  return out


def belief_updates(timestamps, live, live_map, beliefs, single_map=False):
  '''
  turbo's belief updates (time, frame, pose: position + quaternion) and the final estimate of every frame from the
  keyframe logs: a frame moves with the last keyframe at or before it (of its map) at its live pose relative to it.
  single_map: the final estimate only of the frames of the map with the most keyframes (ORB-SLAM3 saves the active
  map's trajectory)
  '''
  n = len(timestamps)
  kf_map, kf_frames = {}, set()
  for _, kt, _, km in beliefs:
    fi = nearest(kt, timestamps)
    for j, f in enumerate(fi):
      if f >= 0 and np.isfinite(live[f]).all():
        kf_frames.add(int(f))
        kf_map[int(f)] = None if km is None else km[j]
  kfs = np.array(sorted(kf_frames), int)
  ok = np.isfinite(live).all(axis=(1, 2))
  ref = np.full(n, -1)
  maps = [None] if live_map is None else sorted(set(live_map[np.isfinite(live_map)]))
  for mp in maps:
    k_sel = kfs if mp is None else kfs[[kf_map[k] == mp for k in kfs]] if len(kfs) else kfs
    f_sel = np.flatnonzero(ok if mp is None else ok & (live_map == mp))
    j = np.searchsorted(k_sel, f_sel, side='right') - 1
    ref[f_sel[j >= 0]] = k_sel[j[j >= 0]]
  rel = np.full((n, 4, 4), np.nan)
  has = ref >= 0
  rel[has] = np.linalg.inv(live[ref[has]]) @ live[has]
  members = {}
  for f in np.flatnonzero(has):
    members.setdefault(int(ref[f]), []).append(f)
  members = {k: np.array(v) for k, v in members.items()}
  t_out, i_out, p_out = [], [], []
  final_kf = {}
  for t, kt, T, km in beliefs:
    fi = nearest(np.array([t]), timestamps)[0]
    if fi < 0:
      continue
    for kf, Tk in zip(nearest(kt, timestamps), T):
      if kf < 0 or kf not in members:
        continue
      final_kf[int(kf)] = Tk
      F = members[kf][members[kf] <= fi]
      if len(F) == 0:
        continue
      P = Tk @ rel[F]
      t_out.append(np.full(len(F), fi, np.int32))
      i_out.append(F.astype(np.int32))
      p_out.append(np.c_[P[:, :3, 3], Rotation.from_matrix(P[:, :3, :3]).as_quat()].astype(np.float32))
  final = np.full((n, 4, 4), np.nan)
  keep = set(final_kf)
  if single_map and kf_map and live_map is not None:
    ids, counts = np.unique([kf_map[k] for k in final_kf], return_counts=True)
    keep = {k for k in final_kf if kf_map[k] == ids[np.argmax(counts)]}
  for k in keep:
    final[members[k]] = final_kf[k] @ rel[members[k]]
  cat = lambda x, shape: np.concatenate(x) if x else np.zeros(shape)
  return {'upd_t': cat(t_out, 0).astype(np.int32), 'upd_i': cat(i_out, 0).astype(np.int32),
          'upd_pose': cat(p_out, (0, 7)).astype(np.float32)}, final, len(keep)


def convert(system, name, d, start=0):
  data = datasets.load(name)
  ts = data.timestamps[start:]
  t, T, extra = read_poses(d / 'raw_live.txt')
  if extra.shape[1:] == (2,):     # ORB-SLAM3: map id, tracking state; only OK poses (not RECENTLY_LOST predictions)
    ok = extra[:, 1] == 2
    t, T, extra = t[ok], T[ok], extra[ok, :1]
  live = to_frames(t, T, ts)
  live_map = None
  if extra.shape[1:] == (1,):
    live_map = to_frames(t, np.broadcast_to(extra[:, :1, None], (len(t), 4, 4)), ts)[:, 0, 0]
  updates, final, n_kfs = belief_updates(ts, live, live_map, read_beliefs(d / 'raw_beliefs.txt'), single_map=system == 'orbslam3')
  info = {}
  if (d / 'raw_info.txt').exists():
    for line in (d / 'raw_info.txt').read_text().split():
      k, v = line.split('=')
      info[k] = float(v)
  gt, gt_R = data.gt[start:, :3, 3], data.gt[start:, :3, :3]
  m_final = evaluate.evaluate(final[:, :3, 3], gt, est_R=final[:, :3, :3], gt_R=gt_R)
  m_online = evaluate.evaluate(live[:, :3, 3], gt, est_R=live[:, :3, :3], gt_R=gt_R,
                               updates=(updates['upd_t'], updates['upd_i'], updates['upd_pose'][:, :3]))
  n = len(ts)
  seconds = info.get('seconds', np.nan)
  result = {
    'name': data.name, 'system': system, 'frames': n, 'start': start, 'seconds': seconds, 'fps': n / seconds,
    'keyframes': n_kfs,
    'final': {k: v for k, v in m_final.items() if not k.startswith('_')},
    'online': {k: v for k, v in m_online.items() if not k.startswith('_')},
    'info': info,
  }
  np.savez_compressed(d / 'trajectory.npz', final=final[:, :3, 3], online=live[:, :3, 3], gt=gt, poses=final,
                      online_poses=live, **updates)
  (d / 'metrics.json').write_text(json.dumps(result, indent=1, default=float))
  if '_aligned' in m_online:
    evaluate.plot(m_online, gt, d / 'trajectory.png', title=f'{data.name} {system} (live)')
  return result


# ---------------------------------------------------------------------------------------------------- running

def run(system, name, d, start=0, timeout=3600):
  d = Path(d).resolve()
  d.mkdir(parents=True, exist_ok=True)
  data = datasets.load(name)
  cmd = SYSTEMS[system](data, start, d)
  with open(d / 'log.txt', 'w') as log:
    log.write(' '.join(cmd) + '\n')
    log.flush()
    t0 = time.perf_counter()
    code = None
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d, start_new_session=True)
    try:
      code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      code = 'timeout'
    finally:
      try:
        os.killpg(p.pid, 9)       # (own process group: worker processes left behind are killed with it)
      except (ProcessLookupError, PermissionError):
        pass
    log.write(f'\n[bench] exit={code} wall={time.perf_counter() - t0:.1f}s\n')
  if not (d / 'raw_info.txt').exists():   # crashed: whatever poses were written are still the live ones
    (d / 'raw_info.txt').write_text(f'seconds={time.perf_counter() - t0}\ncrashed=1\n')
  return convert(system, name, d, start)


def fmt(r):
  o, f = r['online'], r['final']
  return (f"{r['name']:32s} {r['start']:3d} live: final={o['final']:.3f} rot={o.get('rot', np.nan):.2f} "
          f"ate={o['ate']:.3f} | hindsight: final={f['final']:.3f} ate={f['ate']:.3f} | tracked={o['tracked']:.2f} "
          f"fps={r['fps']:.1f} {r['info']}")


def run_all(system, names, out, starts, jobs):
  out.mkdir(parents=True, exist_ok=True)
  names = sorted(names, key=lambda n: -len(datasets.load(n)))
  for n in names:     # extract / undistort the images once, before running in parallel
    image_files(datasets.load(n), camera(datasets.load(n))[0])

  def launch(job):
    n, s = job
    d = out / (n if len(starts) == 1 else f'{n}@{s}')
    try:
      return fmt(run(system, n, d, s))
    except Exception as e:
      return f'{n} {s} ERROR {e!r}'
  with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
    for line in pool.map(launch, [(n, s) for s in starts for n in names]):
      print(line, flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('cmd', choices=['run', 'all', 'convert'])
  p.add_argument('system', choices=list(SYSTEMS))
  p.add_argument('args', nargs='*')
  p.add_argument('--out', type=Path)
  p.add_argument('--start', type=int, default=0)
  p.add_argument('--starts', default='0')
  p.add_argument('--only')
  p.add_argument('--jobs', type=int, default=4)
  a = p.parse_args()
  if a.cmd == 'run':
    print(fmt(run(a.system, a.args[0], a.out, a.start)))
  elif a.cmd == 'convert':
    print(fmt(convert(a.system, a.args[0], Path(a.args[1]), a.start)))
  else:
    names = a.only.split(',') if a.only else datasets.all_names()
    run_all(a.system, names, a.out, [int(s) for s in a.starts.split(',')], a.jobs)


if __name__ == '__main__':
  main()
