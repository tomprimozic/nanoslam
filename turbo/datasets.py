'''
Dataset loaders. Every dataset yields undistorted grayscale images (so everything downstream is pinhole) and
ground truth camera-to-world poses T_wc in the usual optical frame (x right, y down, z forward); for the evaluation
of dense mapping, ground-truth depth per frame (TUM: the Kinect, EuRoC: the laser scan rendered, TartanAir: rendered).

  TUM RGB-D       fr3/long_office_household, fr2/desk              rgbd_dataset_freiburg*/
  EuRoC MAV       MH_01..05, V1_01..03, V2_01..03                  euroc/<name>/mav0/   (cam0 only)
  TartanAir       abandonedfactory, carwelding, seasidetown, ...   tartanair/<name>_sample_P00x/P00x/
'''
import os
import re
import concurrent.futures
from pathlib import Path

import cv2
import yaml
import numpy as np
import pandas as pd
import scipy.spatial.transform
from scipy.spatial.transform import Rotation


ROOT = Path(os.environ.get('NANOSLAM_DATASETS', Path(__file__).resolve().parent.parent / 'datasets'))


def make_K(fx, fy, cx, cy):
  return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def interpolate_poses(t_query, t, pos, quat_xyzw, max_gap=0.1):
  '''
  linear position + slerp rotation at `t_query` (must lie within `t`); NaN where the ground truth has a gap
  (e.g. fr2/desk has a 12 s mocap dropout)
  '''
  keep = np.r_[True, np.diff(t) > 0]     # a few ground truth files have duplicate timestamps
  t, pos, quat_xyzw = t[keep], pos[keep], quat_xyzw[keep]
  P = np.stack([np.interp(t_query, t, pos[:, i]) for i in range(3)], axis=1)
  R = scipy.spatial.transform.Slerp(t, Rotation.from_quat(quat_xyzw))(t_query).as_matrix()
  T = np.tile(np.eye(4), (len(t_query), 1, 1))
  T[:, :3, :3], T[:, :3, 3] = R, P
  i = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
  T[t[i] - t[i - 1] > max_gap] = np.nan
  return T


class Dataset:
  name: str
  K: np.ndarray         # pinhole intrinsics of the *undistorted* images
  width: int
  height: int
  files: list           # image paths
  timestamps: np.ndarray
  gt: np.ndarray        # (N, 4, 4) T_wc

  def setup_undistortion(self, K, dist):
    self.maps = None
    if dist is not None and np.any(dist):
      K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (self.width, self.height), alpha=0)
      self.maps = cv2.initUndistortRectifyMap(K, dist, None, K_new, (self.width, self.height), cv2.CV_16SC2)
      K = K_new
    self.K = K

  def __len__(self):
    return len(self.files)

  def depth(self, i):
    '''ground-truth depth [m] of frame i in the (undistorted) camera, 0 where unknown; None if the dataset has none'''
    return None

  def load(self, i):
    img = cv2.imread(str(self.files[i]), cv2.IMREAD_GRAYSCALE)
    if self.maps is not None:
      img = cv2.remap(img, *self.maps, cv2.INTER_LINEAR)
    return img

  def load_color(self, i):
    '''RGB (grey images repeated)'''
    img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
    if self.maps is not None:
      img = cv2.remap(img, *self.maps, cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  def images(self, start=0, stop=None, prefetch=8):
    '''yields (index, image), loading ahead in background threads (cv2 releases the GIL)'''
    stop = len(self) if stop is None else min(stop, len(self))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
      futures = [pool.submit(self.load, i) for i in range(start, min(start + prefetch, stop))]
      for i in range(start, stop):
        img = futures.pop(0).result()
        if i + prefetch < stop:
          futures.append(pool.submit(self.load, i + prefetch))
        yield i, img


class TUM(Dataset):
  CAMERAS = {   # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    'freiburg1': ([517.3, 516.5, 318.6, 255.3], [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    'freiburg2': ([520.9, 521.0, 325.1, 249.7], [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    'freiburg3': ([535.4, 539.2, 320.1, 247.6], None),    # fr3 images are already undistorted
  }

  def __init__(self, path: Path):
    self.name = path.name.removeprefix('rgbd_dataset_')
    rgb = pd.read_csv(path / 'rgb.txt', sep=' ', comment='#', names=['t', 'file'])
    gt = pd.read_csv(path / 'groundtruth.txt', sep=' ', comment='#', names='t tx ty tz qx qy qz qw'.split()).to_numpy()
    rgb = rgb[(rgb.t >= gt[0, 0]) & (rgb.t <= gt[-1, 0])]
    self.files = [path / f for f in rgb.file]
    self.timestamps = rgb.t.to_numpy()
    self.gt = interpolate_poses(self.timestamps, gt[:, 0], gt[:, 1:4], gt[:, 4:8])
    self.height, self.width = cv2.imread(str(self.files[0]), cv2.IMREAD_GRAYSCALE).shape
    intrinsics, dist = self.CAMERAS[self.name.split('_')[0]]
    self.setup_undistortion(make_K(*intrinsics), None if dist is None else np.array(dist))
    self.depth_files = None
    if (path / 'depth.txt').exists() and (path / 'depth').is_dir():
      dep = pd.read_csv(path / 'depth.txt', sep=' ', comment='#', names=['t', 'file'])
      j = np.clip(np.searchsorted(dep.t.to_numpy(), self.timestamps), 1, len(dep) - 1)
      j = np.where(np.abs(dep.t.to_numpy()[j - 1] - self.timestamps) < np.abs(dep.t.to_numpy()[j] - self.timestamps), j - 1, j)
      close = np.abs(dep.t.to_numpy()[j] - self.timestamps) < 0.02
      self.depth_files = [path / dep.file.iloc[k] if ok else None for k, ok in zip(j, close)]

  def depth(self, i):
    '''the Kinect depth registered to the RGB image (nearest timestamp within 20 ms), undistorted like the image'''
    if self.depth_files is None or self.depth_files[i] is None:
      return None
    d = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
    if self.maps is not None:
      d = cv2.remap(d, *self.maps, cv2.INTER_NEAREST)
    return d


class EuRoC(Dataset):
  def __init__(self, path: Path):
    self.name = path.name
    mav = path / 'mav0'
    cam = yaml.safe_load((mav / 'cam0/sensor.yaml').read_text())
    images = pd.read_csv(mav / 'cam0/data.csv')
    gt = pd.read_csv(mav / 'state_groundtruth_estimate0/data.csv').to_numpy()
    t_img = images.iloc[:, 0].to_numpy()
    keep = (t_img >= gt[0, 0]) & (t_img <= gt[-1, 0])
    self.files = [mav / 'cam0/data' / f for f in images.iloc[:, 1][keep]]
    self.timestamps = t_img[keep] / 1e9
    # ground truth is the body (IMU) pose, quaternions are w x y z
    T_wb = interpolate_poses(self.timestamps, gt[:, 0] / 1e9, gt[:, 1:4], gt[:, [5, 6, 7, 4]])
    T_bc = np.array(cam['T_BS']['data']).reshape(4, 4)
    self.gt = T_wb @ T_bc
    self.width, self.height = cam['resolution']
    self.setup_undistortion(make_K(*cam['intrinsics']), np.array(cam['distortion_coefficients']))
    self.scan_path = mav / 'pointcloud0' / 'data.ply'
    self._scan = None

  def scan(self):
    '''the laser scan (N, 3) in the world frame (cached as .npy next to the .ply), or None'''
    if self._scan is None and self.scan_path.exists():
      cache = self.scan_path.with_suffix('.npy')
      if not cache.exists():
        with open(self.scan_path) as f:
          n = 0
          for line in f:
            if line.startswith('element vertex'):
              n = int(line.split()[2])
            if line.startswith('end_header'):
              break
          xyz = np.loadtxt(f, usecols=(0, 1, 2), max_rows=n, dtype=np.float32)
        np.save(cache, xyz)
      self._scan = np.load(cache)
    return self._scan

  def depth(self, i):
    '''the laser scan rendered into frame i (z-buffer, 2x2 splats; pixels behind a nearer surface dropped)'''
    X = self.scan()
    if X is None or not np.isfinite(self.gt[i]).all():
      return None
    T_cw = np.linalg.inv(self.gt[i])
    Xc = X @ T_cw[:3, :3].T.astype(np.float32) + T_cw[:3, 3].astype(np.float32)
    Xc = Xc[Xc[:, 2] > 0.1]
    u = self.K[0, 0] * Xc[:, 0] / Xc[:, 2] + self.K[0, 2]
    v = self.K[1, 1] * Xc[:, 1] / Xc[:, 2] + self.K[1, 2]
    z = np.full(self.height * self.width, np.inf, np.float32)
    order = np.argsort(-Xc[:, 2])         # (nearest last: it wins)
    for du in (0, 1):
      for dv in (0, 1):
        ui, vi = np.floor(u - 0.5).astype(int) + du, np.floor(v - 0.5).astype(int) + dv
        ok = (ui >= 0) & (ui < self.width) & (vi >= 0) & (vi < self.height)
        o = order[ok[order]]
        np.minimum.at(z, vi[o] * self.width + ui[o], Xc[o, 2])
    z = z.reshape(self.height, self.width)
    near = cv2.erode(np.where(np.isfinite(z), z, 1e9).astype(np.float32), np.ones((5, 5), np.uint8))
    z[~np.isfinite(z) | (z > 1.1 * near)] = 0
    return z


class TartanAir(Dataset):
  def __init__(self, path: Path):
    self.name = path.parent.name.replace('_sample', '')
    self.files = sorted((path / 'image_left').glob('*.png'))
    pose = np.loadtxt(path / 'pose_left.txt')
    self.timestamps = np.arange(len(self.files)) / 10.0
    T = np.tile(np.eye(4), (len(pose), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(pose[:, 3:7]).as_matrix()
    T[:, :3, 3] = pose[:, :3]
    # TartanAir poses are in NED (x forward, y right, z down): rotate the body frame into the optical frame
    T_ned_optical = np.eye(4)
    T_ned_optical[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    self.gt = T @ T_ned_optical
    self.width, self.height = 640, 480
    self.setup_undistortion(make_K(320, 320, 320, 240), None)
    self.depth_dir = path / 'depth_left'

  def depth(self, i):
    '''the rendered depth (sky and beyond 100 m: unknown)'''
    f = self.depth_dir / self.files[i].name.replace('_left.png', '_left_depth.npy')
    if not f.exists():
      return None
    d = np.load(f).astype(np.float32)
    d[~(d < 100)] = 0
    return d


def all_names():
  names = [p.name.removeprefix('rgbd_dataset_') for p in sorted(ROOT.glob('rgbd_dataset_*')) if p.is_dir()]
  names += [p.name for p in sorted((ROOT / 'euroc').glob('*')) if (p / 'mav0').is_dir()]
  names += [p.name.replace('_sample', '') for p in sorted((ROOT / 'tartanair').glob('*')) if p.is_dir()]
  return names


ALIASES = {'household': 'freiburg3_long_office_household', 'desk': 'freiburg2_desk'}


def load(name: str) -> Dataset:
  name = ALIASES.get(name, name)
  if (p := ROOT / f'rgbd_dataset_{name}').is_dir():
    return TUM(p)
  for p in sorted((ROOT / 'euroc').glob('*')):
    if p.name.lower().startswith(name.lower()):    # 'MH_01' or 'MH_01_easy'
      return EuRoC(p)
  for p in sorted((ROOT / 'tartanair').glob('*')):
    if p.name.replace('_sample', '').startswith(name):
      return TartanAir(next(q for q in p.iterdir() if q.is_dir() and re.fullmatch(r'P\d+', q.name)))
  raise ValueError(f'unknown dataset {name}, available: {all_names()}')
