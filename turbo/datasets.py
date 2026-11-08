'''
Dataset loaders. Every dataset yields undistorted grayscale images (so everything downstream is pinhole) and
ground truth camera-to-world poses T_wc in the usual optical frame (x right, y down, z forward). Images are read
from directories or straight from zip archives (fast; tar archives have to be extracted).

  TUM RGB-D       freiburg1_desk, freiburg3_long_office_household, ...  rgbd_dataset_freiburg*/   (anywhere below)
  EuRoC MAV       MH_01_easy ... V2_03_difficult (cam0 only)           euroc|EuRoC/<name>/mav0/ or <name>.zip
  TartanAir       abandonedfactory_P001, ... (samples)                 tartanair|TartanAir/<name>_sample_P00x/P00x/
                  ME000 ... ME015 (CVPR 2020 SLAM challenge, mono)     TartanAir/cvpr_2020_test_mono/MExxx/ + .txt
  KITTI odometry  KITTI_00 ... KITTI_10 (image_0; ground truth poses)  KITTI/NN/image_0/, KITTI/poses/NN.txt
  TUM monoVO      monoVO_01 ... monoVO_50 (ground truth only at the    TUM_monoVO/sequence_NN/images.zip
                  start and end, in an arbitrary scale)
  ETH3D SLAM      eth3d_<name> (mono)                                  ETH3D-SLAM/<name>_mono.zip or <name>/
  ICL-NUIM        icl_living_room_traj0, icl_office_room_traj0          ICL-NUIM/<name>_frei_png/
'''
import os
import re
import zipfile
import functools
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


def poses_from(t_xyz_quat_xyzw):
  '''(N, 7) tx ty tz qx qy qz qw -> (N, 4, 4)'''
  x = np.asarray(t_xyz_quat_xyzw, np.float64)
  T = np.tile(np.eye(4), (len(x), 1, 1))
  ok = np.isfinite(x).all(axis=1)
  T[~ok] = np.nan
  T[ok, :3, :3] = Rotation.from_quat(x[ok, 3:7]).as_matrix()
  T[ok, :3, 3] = x[ok, :3]
  return T


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


@functools.lru_cache(maxsize=None)
def _zip(path):
  return zipfile.ZipFile(path)      # (reads of one ZipFile are serialized by its own lock: fine for 2 threads)


class ZipMember:
  '''an image inside a zip archive (str: "archive.zip/member")'''
  def __init__(self, archive, member):
    self.archive, self.member = Path(archive), member

  def read(self):
    return _zip(self.archive).read(self.member)

  def __str__(self):
    return f'{self.archive}/{self.member}'


def read_image(f, flags):
  if isinstance(f, ZipMember):
    return cv2.imdecode(np.frombuffer(f.read(), np.uint8), flags)
  return cv2.imread(str(f), flags)


class Source:
  '''a directory or a zip archive with the same layout (relative paths)'''
  def __init__(self, path, prefix=''):
    self.path, self.prefix = Path(path), prefix
    self.zip = self.path.suffix == '.zip'

  def text(self, rel):
    return _zip(self.path).read(self.prefix + rel).decode() if self.zip else (self.path / rel).read_text()

  def file(self, rel):
    return ZipMember(self.path, self.prefix + rel) if self.zip else self.path / rel

  def table(self, rel, **kw):
    import io
    return pd.read_csv(io.StringIO(self.text(rel)), **kw)


class Dataset:
  name: str
  K: np.ndarray         # pinhole intrinsics of the *undistorted* images
  width: int
  height: int
  files: list           # image paths (or ZipMember)
  timestamps: np.ndarray
  gt: np.ndarray        # (N, 4, 4) T_wc (rows NaN where there is none)
  up = np.array([0.0, 0.0, 1.0])    # the ground truth's world up (display only)
  maps = None

  def setup_undistortion(self, K, dist):
    self.maps = None
    if dist is not None and np.any(dist):
      K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (self.width, self.height), alpha=0)
      self.maps = cv2.initUndistortRectifyMap(K, dist, None, K_new, (self.width, self.height), cv2.CV_16SC2)
      K = K_new
    self.K = K

  def __len__(self):
    return len(self.files)

  def load(self, i):
    img = read_image(self.files[i], cv2.IMREAD_GRAYSCALE)
    if self.maps is not None:
      img = cv2.remap(img, *self.maps, cv2.INTER_LINEAR)
    return img

  def load_color(self, i):
    '''RGB, undistorted like load() (grey images: repeated)'''
    img = read_image(self.files[i], cv2.IMREAD_COLOR)
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


class EuRoC(Dataset):
  '''a directory with mav0/, or the zip archive of the sequence (as downloaded)'''
  def __init__(self, path: Path):
    self.name = path.name.removesuffix('.zip')
    src = Source(path) if path.suffix == '.zip' else Source(path)
    cam = yaml.safe_load(src.text('mav0/cam0/sensor.yaml'))
    images = src.table('mav0/cam0/data.csv')
    gt = src.table('mav0/state_groundtruth_estimate0/data.csv').to_numpy()
    t_img = images.iloc[:, 0].to_numpy()
    keep = (t_img >= gt[0, 0]) & (t_img <= gt[-1, 0])
    self.files = [src.file(f'mav0/cam0/data/{f}') for f in images.iloc[:, 1][keep]]
    self.timestamps = t_img[keep] / 1e9
    # ground truth is the body (IMU) pose, quaternions are w x y z
    T_wb = interpolate_poses(self.timestamps, gt[:, 0] / 1e9, gt[:, 1:4], gt[:, [5, 6, 7, 4]])
    T_bc = np.array(cam['T_BS']['data']).reshape(4, 4)
    self.gt = T_wb @ T_bc
    self.width, self.height = cam['resolution']
    self.setup_undistortion(make_K(*cam['intrinsics']), np.array(cam['distortion_coefficients']))


def _ned_to_optical(pose):
  '''TartanAir poses (tx ty tz qx qy qz qw) are in NED (x forward, y right, z down): rotate the body frame into the
  optical frame'''
  T = poses_from(pose)
  T_ned_optical = np.eye(4)
  T_ned_optical[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
  return T @ T_ned_optical


class TartanAir(Dataset):
  up = np.array([0.0, 0.0, -1.0])     # (NED)

  def __init__(self, path: Path, name=None, images='image_left', pose_file=None):
    self.name = name or path.parent.name.replace('_sample', '')
    self.files = sorted((path / images).glob('*.png'))
    self.timestamps = np.arange(len(self.files)) / 10.0
    self.gt = _ned_to_optical(np.loadtxt(pose_file or path / 'pose_left.txt'))
    self.width, self.height = 640, 480
    self.setup_undistortion(make_K(320, 320, 320, 240), None)


class TartanAirTest(TartanAir):
  '''CVPR 2020 SLAM challenge, monocular track: MExxx/ (images) + MExxx.txt (poses)'''
  def __init__(self, path: Path):
    self.name = f'tartanair_{path.name}'
    self.files = sorted(path.glob('*.png'))
    self.timestamps = np.arange(len(self.files)) / 10.0
    self.gt = _ned_to_optical(np.loadtxt(path.parent / f'{path.name}.txt'))
    self.width, self.height = 640, 480
    self.setup_undistortion(make_K(320, 320, 320, 240), None)


class KITTI(Dataset):
  '''odometry benchmark, left grey camera (image_0, rectified); ground truth: poses/NN.txt (cam0 to world, 3x4)'''
  def __init__(self, path: Path):
    self.name = f'KITTI_{path.name}'
    self.files = sorted((path / 'image_0').glob('*.png'))
    self.timestamps = np.loadtxt(path / 'times.txt')[:len(self.files)]
    P0 = np.array(next(l for l in (path / 'calib.txt').read_text().splitlines() if l.startswith('P0:')).split()[1:],
                  float).reshape(3, 4)
    pose = np.loadtxt(path.parent / 'poses' / f'{path.name}.txt').reshape(-1, 3, 4)
    self.gt = np.tile(np.eye(4), (len(pose), 1, 1))
    self.gt[:, :3] = pose
    self.up = np.array([0.0, -1.0, 0.0])      # (camera frame of the first image: y down)
    self.height, self.width = cv2.imread(str(self.files[0]), cv2.IMREAD_GRAYSCALE).shape
    self.setup_undistortion(make_K(P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]), None)


class MonoVO(Dataset):
  '''
  TUM monoVO: images.zip (1280 x 1024, FOV / ATAN fisheye model), undistorted and cropped to DSO's standard 640 x 480
  pinhole (camera.txt); ground truth only for the first and last ~10-20 s of each sequence (the same place: the
  benchmark measures the drift accumulated over the loop) and in an arbitrary scale
  '''
  def __init__(self, path: Path):
    self.name = f'monoVO_{path.name.removeprefix("sequence_")}'
    lines = (path / 'camera.txt').read_text().split('\n')
    fx, fy, cx, cy, w = map(float, lines[0].split()[:5])
    in_w, in_h = map(int, lines[1].split())
    ox, oy, ocx, ocy = map(float, lines[2].split()[:4])
    self.width, self.height = map(int, lines[3].split())
    # relative calibration, first pixel centre at (0.5, 0.5) (DSO's convention)
    fx, fy, cx, cy = fx * in_w, fy * in_h, cx * in_w - 0.5, cy * in_h - 0.5
    self.K = make_K(ox * self.width, oy * self.height, ocx * self.width - 0.5, ocy * self.height - 0.5)
    u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
    x, y = (u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]
    r = np.sqrt(x * x + y * y)
    factor = np.where(r > 1e-8, np.arctan(2 * r * np.tan(w / 2)) / (w * np.maximum(r, 1e-8)), 2 * np.tan(w / 2) / w)
    map_x, map_y = (fx * x * factor + cx).astype(np.float32), (fy * y * factor + cy).astype(np.float32)
    self.maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    names = sorted(n for n in _zip(path / 'images.zip').namelist() if n.endswith(('.jpg', '.png')))
    self.files = [ZipMember(path / 'images.zip', n) for n in names]
    times = np.loadtxt(path / 'times.txt', usecols=1)
    self.timestamps = times[:len(self.files)]
    gt = np.genfromtxt(path / 'groundtruthSync.txt')        # (time tx ty tz qx qy qz qw, NaN where none)
    self.gt = poses_from(gt[:len(self.files), 1:8])


class ETH3D(Dataset):
  '''ETH3D SLAM benchmark (monocular: rgb/ only, already undistorted), a directory or the downloaded <name>_mono.zip'''
  def __init__(self, path: Path):
    base = path.name.removesuffix('.zip').removesuffix('_mono')
    self.name = f'eth3d_{base}'
    src = Source(path, f'{base}/') if path.suffix == '.zip' else Source(path)
    rgb = src.table('rgb.txt', sep=' ', comment='#', names=['t', 'file'])
    gt = src.table('groundtruth.txt', sep=' ', comment='#', names='t tx ty tz qx qy qz qw'.split()).to_numpy()
    rgb = rgb[(rgb.t >= gt[0, 0]) & (rgb.t <= gt[-1, 0])]
    self.files = [src.file(f) for f in rgb.file]
    self.timestamps = rgb.t.to_numpy()
    self.gt = interpolate_poses(self.timestamps, gt[:, 0], gt[:, 1:4], gt[:, 4:8])
    fx, fy, cx, cy = map(float, src.text('calibration.txt').split()[:4])
    self.height, self.width = read_image(self.files[0], cv2.IMREAD_GRAYSCALE).shape
    self.setup_undistortion(make_K(fx, fy, cx, cy), None)


class ICL(Dataset):
  '''
  ICL-NUIM, TUM-compatible version (<name>_frei_png): its intrinsics have a negative fy (y up). With fy > 0 the
  reconstruction is the mirror image, so the ground truth is mirrored the same way (y -> -y in the world and the
  camera frame: proper rotations again, distances unchanged). Ground truth row k (1-based) is image k (image 0 has none)
  '''
  def __init__(self, path: Path):
    self.name = f'icl_{path.name.removesuffix("_frei_png")}'
    files = sorted((path / 'rgb').glob('*.png'), key=lambda f: int(f.stem))
    self.files = files
    self.timestamps = np.array([int(f.stem) for f in files]) / 30.0
    gt = np.loadtxt(next(path.glob('*.gt.freiburg')))
    self.gt = np.full((len(files), 4, 4), np.nan)
    idx = gt[:, 0].astype(int)
    ok = idx < len(files)
    S = np.diag([1.0, -1.0, 1.0, 1.0])
    self.gt[idx[ok]] = S @ poses_from(gt[ok, 1:8]) @ S
    self.width, self.height = 640, 480
    self.setup_undistortion(make_K(481.2, 480.0, 319.5, 239.5), None)


def _find(*patterns):
  '''paths below ROOT matching any of the glob patterns (relative to ROOT)'''
  out = []
  for p in patterns:
    out += sorted(ROOT.glob(p))
  return out


@functools.lru_cache(maxsize=None)
def index():
  '''name -> (loader class, path) of every dataset found below ROOT'''
  found = {}

  def add(cls, path, name):
    found.setdefault(name, (cls, path))
  for p in _find('rgbd_dataset_*', '*/rgbd_dataset_*'):
    if p.is_dir() and (p / 'rgb.txt').exists():
      add(TUM, p, p.name.removeprefix('rgbd_dataset_'))
  for p in _find('euroc/*', 'EuRoC/*'):
    if (p / 'mav0').is_dir() or (p.suffix == '.zip' and re.match(r'(MH|V\d)_\d\d', p.name)):
      add(EuRoC, p, p.name.removesuffix('.zip'))
  for p in _find('tartanair/*_sample_P*', 'TartanAir/*_sample_P*'):
    if p.is_dir():
      q = next((q for q in p.iterdir() if q.is_dir() and re.fullmatch(r'P\d+', q.name)), None)
      if q is not None:
        add(TartanAir, q, p.name.replace('_sample', ''))
  for p in _find('TartanAir/cvpr_2020_test_mono/ME*', 'tartanair/cvpr_2020_test_mono/ME*'):
    if p.is_dir() and (p.parent / f'{p.name}.txt').exists():
      add(TartanAirTest, p, f'tartanair_{p.name}')
  for p in _find('KITTI/[0-9][0-9]', 'kitti/[0-9][0-9]'):
    if (p / 'image_0').is_dir() and (p.parent / 'poses' / f'{p.name}.txt').exists():
      add(KITTI, p, f'KITTI_{p.name}')
  for p in _find('TUM_monoVO/sequence_*', 'monoVO/sequence_*'):
    if (p / 'images.zip').exists():
      add(MonoVO, p, f'monoVO_{p.name.removeprefix("sequence_")}')
  for p in _find('ETH3D-SLAM/*_mono.zip', 'ETH3D-SLAM/*', 'eth3d/*'):     # (test sequences have no ground truth)
    base = p.name.removesuffix('.zip').removesuffix('_mono')
    if ((p.suffix == '.zip' and f'{base}/groundtruth.txt' in _zip(p).namelist())
        or (p.is_dir() and (p / 'groundtruth.txt').exists())):
      add(ETH3D, p, f'eth3d_{base}')
  for p in _find('ICL-NUIM/*_frei_png', 'icl/*_frei_png'):
    if (p / 'rgb').is_dir():
      add(ICL, p, f'icl_{p.name.removesuffix("_frei_png")}')
  return found


def all_names():
  return list(index())


ALIASES = {'household': 'freiburg3_long_office_household', 'desk': 'freiburg2_desk'}


def load(name: str) -> Dataset:
  name = ALIASES.get(name, name)
  found = index()
  if name in found:
    cls, path = found[name]
    return cls(path)
  for n, (cls, path) in found.items():      # prefixes: 'MH_01' for 'MH_01_easy'
    if n.lower().startswith(name.lower()):
      return cls(path)
  raise ValueError(f'unknown dataset {name}, available: {all_names()}')
