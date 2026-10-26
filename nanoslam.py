# BSD Zero Clause License
# Copyright (C) 2026 by Tom Primozic <tom@primozic.org>
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA
# OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.



import time
import argparse
import collections
from pathlib import Path
from dataclasses import dataclass

import cv2
import poselib
import pycolmap
import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.spatial.transform
from scipy.spatial.transform import Rotation



DATASET_DIR = Path(__file__).parent / 'datasets'



def load_dataset(dataset: str):
  match dataset.replace('/', '-'):
    case 'household' | 'fr3-long-office-household':
      return Dataset(DATASET_DIR / 'rgbd_dataset_freiburg3_long_office_household')
    case _:
      raise Exception(f'unknown dataset {dataset}')



class Dataset:
  CAMERA = {
    # from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    'fx': 535.4,
    'fy': 539.2,
    'cx': 320.1,
    'cy': 247.6,
  }
  
  def __init__(self, path: str | Path):
    self.path = Path(path)
    images = pd.read_csv(self.path / 'rgb.txt', sep=' ', comment='#', names=['timestamp', 'filename'])
    images = images.set_index('timestamp')['filename']
    assert images.index.is_monotonic_increasing and images.index.is_unique

    gt = pd.read_csv(self.path / 'groundtruth.txt', sep=' ', comment='#', names='timestamp,tx,ty,tz,qx,qy,qz,qw'.split(','))
    gt = gt.set_index('timestamp')
    assert gt.index.is_monotonic_increasing and gt.index.is_unique

    images = images.loc[(gt.index[0] < images.index) & (images.index < gt.index[-1])]
    index_union = images.index.union(gt.index, sort=True)
    pos = gt[['tx', 'ty', 'tz']]
    pos = pos.reindex(index_union).interpolate(method='index').loc[images.index]
    rot = Rotation.from_quat(gt[['qx', 'qy', 'qz', 'qw']], scalar_first=False)
    rot = scipy.spatial.transform.Slerp(gt.index, rot)(images.index)

    gt = np.zeros((len(images.index), 4, 4))
    gt[:, 3, 3] = 1
    gt[:, :3, :3] = rot.as_matrix()
    gt[:, :3, 3] = pos.values

    self.images, self.gt = images, gt
    self.len = len(images)
    self.K = make_K(**self.CAMERA)
    self.height, self.width = self.load_image(0).shape[:2]

  def load_image(self, i: int) -> np.ndarray:
    return cv2.imread(str(self.path / self.images.iloc[i]), cv2.IMREAD_COLOR_RGB)

  def __iter__(self):
    for i in range(self.len):
      yield Frame(i, self.images.index[i], self.load_image(i), self.gt[i])



def make_K(fx, fy, cx, cy):     # poses and K are float64 because we invert them
  return np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1],
  ], dtype=np.float64)



def inv_K(K: np.ndarray):
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  return np.array([
    [1/fx,    0, -cx/fx],
    [   0, 1/fy, -cy/fy],
    [   0,    0,      1],
  ])



def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
  assert R.shape == (3, 3)
  assert t.shape == (3,)
  pose = np.eye(4, dtype=np.float64)
  pose[:3, :3] = R
  pose[:3, 3] = t
  return pose



def inv_pose(pose: np.ndarray) -> np.ndarray:
  assert pose.shape == (4, 4)
  assert (pose[3] == [0, 0, 0, 1]).all()
  R = pose[:3, :3]
  t = pose[:3, 3]
  return make_pose(R.T, -R.T @ t)



def project(K: np.ndarray, T_cw: np.ndarray, pw: np.ndarray):
  pc = pw @ T_cw[:3, :3].T + T_cw[:3, 3]
  z = pc[:, 2]
  uv = (pc @ K.T)[:, :2] / z[:, None]
  return uv.astype(np.float32), z



def points2d_to_homogenous(pts: np.ndarray) -> np.ndarray:
  n = len(pts)
  assert pts.shape == (n, 2)
  return np.hstack((pts, np.ones((n, 1), dtype=pts.dtype)))



def normalize_points2d(K_inv: np.ndarray, pts: np.ndarray) -> np.ndarray:
  n = (K_inv @ points2d_to_homogenous(pts).T).T
  assert (n[:, 2] == 1).all()
  return n[:, :2]



def triangulate(K, T_cw_cur, T_cw_ref, pts_cur, pts_ref, *, min_depth: float, max_reproj_error_px: float):
  K_inv = inv_K(K)
  pts_cur_n = normalize_points2d(K_inv, pts_cur)
  pts_ref_n = normalize_points2d(K_inv, pts_ref)

  points_4d = cv2.triangulatePoints(T_cw_cur[:3], T_cw_ref[:3], pts_cur_n.T, pts_ref_n.T)
  bad = np.abs(points_4d[3]) < 1e-6 * np.linalg.norm(points_4d[:3], axis=0)
  points_3d = (points_4d[:3] / points_4d[3]).T

  uv_cur, z_cur = project(K, T_cw_cur, points_3d)
  uv_ref, z_ref = project(K, T_cw_ref, points_3d)

  bad |= z_cur <= min_depth
  bad |= z_ref <= min_depth
  bad |= np.linalg.norm(uv_cur - pts_cur, axis=1) > max_reproj_error_px
  bad |= np.linalg.norm(uv_ref - pts_ref, axis=1) > max_reproj_error_px
  return points_3d, bad



def klt(img_ref, img_cur, pts_ref, pts_cur=None, *, win_size, max_iter, eps, max_level, max_error, max_fb_dist):
  win_size = (win_size, win_size)
  criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, max_iter, eps)

  pts_cur, ok_forward, error = cv2.calcOpticalFlowPyrLK(img_ref, img_cur, pts_ref, pts_cur,
                                                        winSize=win_size, maxLevel=max_level, criteria=criteria,
                                                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW if pts_cur is not None else None)
  pts_back, ok_back, _ = cv2.calcOpticalFlowPyrLK(img_cur, img_ref, pts_cur, pts_ref.copy(),
                                                  winSize=win_size, maxLevel=max_level, criteria=criteria,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
  distance = np.linalg.norm(pts_ref - pts_back, axis=1).flatten()
  ok = (ok_forward.flatten() == 1) & (ok_back.flatten() == 1) & (error[:, 0] < max_error) & (distance < max_fb_dist)
  return pts_cur, ok



draw_points_colormap = matplotlib.colormaps['cool']

def draw_points(img: np.ndarray, pts: list['Point'], radius=3, thickness=-1):
  shift = 8
  scale = 2 ** shift

  img = img.copy()
  for pt in pts:
    u, v = pt.uv
    if pt.is_3d:
      color = (255, 0, 0)
    else:
      r, g, b, a = draw_points_colormap(np.clip(pt.age, 0, 100) / 100)
      color = (r * 255, g * 255, b * 255)
    img = cv2.circle(img, (round(u * scale), round(v * scale)), radius * scale, color, shift=shift, thickness=thickness)
  return img



def kabsch_umeyama(A, B):
  # from: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
  # licence: CC0
  '''
    Returns rotation R, scale s, translation t

    can be used to align B to A using:

    t + s * (R @ B.T).T
  '''
  assert len(A) > 0
  assert A.shape == B.shape
  assert A.shape[1] in {2, 3}   # not required, but sanity check that we're passing in points in rows
  n, m = A.shape

  EA = np.mean(A, axis=0)
  EB = np.mean(B, axis=0)
  VarB = np.mean(np.linalg.norm(B - EB, axis=1) ** 2)

  H = ((A - EA).T @ (B - EB)) / n
  U, D, VT = np.linalg.svd(H)
  d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
  S = np.eye(m)   # S can also be a vector (then `trace` below is just a sum)
  S[-1, -1] = d

  R = U @ S @ VT
  s = np.trace(np.diag(D) @ S) / VarB
  t = EA - s * R @ EB

  return R, s, t



def optimize_setup(K, max_iter: int, loss='soft_l1'):
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  reconstruction = pycolmap.Reconstruction()
  reconstruction.add_camera(pycolmap.Camera(model='PINHOLE', camera_id=0, width=1, height=1, params=[fx, fy, cx, cy]))
  sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, id=0)
  rig = pycolmap.Rig(rig_id=0)
  rig.add_ref_sensor(sensor)
  reconstruction.add_rig(rig)

  options = pycolmap.BundleAdjustmentOptions()
  options.refine_focal_length = False
  options.refine_principal_point = False
  options.refine_extra_params = False
  options.refine_sensor_from_rig = False
  options.print_summary = False
  options.ceres.loss_function_type = {
    'soft_l1': pycolmap.LossFunctionType.SOFT_L1,
    'cauchy': pycolmap.LossFunctionType.CAUCHY,
    'huber': pycolmap.LossFunctionType.HUBER,
    'trivial': pycolmap.LossFunctionType.TRIVIAL
  }[loss]
  options.ceres.loss_function_scale = 1.
  options.ceres.max_num_images_direct_dense_cpu_solver = 0
  options.ceres.solver_options.max_num_iterations = max_iter
  options.ceres.solver_options.num_threads = 1
  options.ceres.solver_options.function_tolerance = 1e-6
  options.ceres.solver_options.gradient_tolerance = 1e-8
  options.ceres.solver_options.parameter_tolerance = 1e-8

  config = pycolmap.BundleAdjustmentConfig()
  # fix first camera pose, and one translation dimension of second camera to fix scale
  config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

  return reconstruction, options, config, sensor



def optimize(K, frame_poses, frame_fixed, point_positions, frame_indices, point_indices, uvs, *, max_iter, loss, max_error):
  assert len(frame_poses) == len(frame_fixed)
  assert len(frame_indices) == len(point_indices) == uvs.shape[0]
  reconstruction, options, config, sensor = optimize_setup(K, max_iter, loss)

  num_frames = len(frame_poses)
  num_points = len(point_positions)
  num_obs = uvs.shape[0]

  tracks = [pycolmap.Track() for _ in range(num_points)]
  observation_indices = np.zeros_like(frame_indices)
  for frame_index in range(num_frames):
    frame = pycolmap.Frame(frame_id=frame_index, rig_id=0, rig_from_world=pycolmap.Rigid3d(frame_poses[frame_index, :3]))
    frame.add_data_id(pycolmap.data_t(sensor_id=sensor, id=frame_index))
    reconstruction.add_frame(frame)

    observations = frame_indices == frame_index
    image = pycolmap.Image(name=str(frame_index), keypoints=uvs[observations], camera_id=0, image_id=frame_index)
    image.frame_id = frame_index
    reconstruction.add_image(image)
    reconstruction.register_frame(frame_index)
    config.add_image(frame_index)
    if frame_fixed[frame_index]:
      config.set_constant_rig_from_world_pose(frame_index)

    for obs_index, point_index in enumerate(point_indices[observations]):
      tracks[point_index].add_element(frame_index, obs_index)
    observation_indices[observations] = range(np.sum(observations))

  point_ids = [reconstruction.add_point3D(point_positions[i], tracks[i]) for i in range(num_points)]

  summary = pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()

  def results():
    frame_poses[:, :3] = [reconstruction.image(frame_index).cam_from_world().matrix() for frame_index in range(num_frames)]
    point_positions[:] = [reconstruction.point3D(point_id).xyz if reconstruction.exists_point3D(point_id) else point_positions[i]
                          for i, point_id in enumerate(point_ids)]

    zs = np.zeros(num_obs, dtype=uvs.dtype)
    uvs_proj = np.zeros_like(uvs)
    for i in range(num_obs):    # einsum would be much faster, but less readable
      uvs_proj[i], z = project(K, frame_poses[frame_indices[i]], point_positions[point_indices[i]][None, :])
      zs[i] = z[0]

    error = np.linalg.norm(uvs_proj - uvs, axis=1)
    ok = np.isfinite(error) & (zs > 0) & (error <= max_error)
    return frame_poses, point_positions, ok

  frame_poses, point_positions, ok = results()
  # second round: remove outliers, rerun with L2 loss
  for i in np.flatnonzero(~ok):
    if reconstruction.image(frame_indices[i]).point2D(observation_indices[i]).has_point3D():
      reconstruction.delete_observation(frame_indices[i], observation_indices[i])
  for point_id in point_ids:
    if reconstruction.exists_point3D(point_id) and reconstruction.point3D(point_id).track.length() < 2:
      reconstruction.delete_point3D(point_id)
  options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
  summary = pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()

  frame_poses, point_positions, ok = results()
  ok_points = np.array([reconstruction.exists_point3D(point_id) for point_id in point_ids])
  ok &= ok_points[point_indices]
  success = summary.is_solution_usable() and np.isfinite(frame_poses).all() and np.isfinite(point_positions).all()
  return success, frame_poses, point_positions, ok



@dataclass(eq=False)    # compare using object identity (the `is` operator)
class Frame:
  id: int
  ts: float
  img: np.ndarray
  gt: np.ndarray | None = None



@dataclass(eq=False)
class KeyFrame:
  points: list['Point']
  frame: Frame
  pose: np.ndarray

  @property
  def id(self):
    return self.frame.id



@dataclass(eq=False)
class Point:
  keyframe: KeyFrame
  last_frame: Frame
  uv: np.ndarray
  uvs: dict
  color: tuple
  xyz: np.ndarray | None = None

  @property
  def age(self):
    return self.last_frame.id - self.keyframe.frame.id

  @property
  def is_3d(self):
    return self.xyz is not None



class Config:
  lk_win_size = 21
  lk_max_iter = 100
  lk_eps = 0.1
  lk_max_level = 5
  lk_max_error = 30
  lk_max_fb_dist = 0.5

  rel_threshold_px = 1.0
  rel_prob = 0.9999
  rel_min_iter = 10
  rel_max_iter = 1000

  init_min_parallax = 15
  init_min_depth = 0
  init_max_reproj_error_px = 3

  abs_max_reproj_error = 3.0

  reloc_window = 8
  reloc_min_points = 100
  reloc_border = 1
  reloc_max_iter = 100
  reloc_eps = 0.1
  reloc_max_level = 5
  reloc_win_size = 21
  reloc_max_error = 30
  reloc_max_fb_dist = 0.5

  kf_min_gap = 3
  kf_min_3d_pts = 80
  kf_min_3d_pts_ratio = 0.5
  kf_min_pts = 50

  tri_min_depth = 0
  tri_max_reproj_error_px = 3

  feat_mask_radius = 20
  feat_num = 10_000
  feat_min_quality = 0.01
  feat_min_dist = 5
  feat_block_size = 3
  feat_subpix = True
  feat_subpix_win_size = 3
  feat_subpix_max_iter = 30
  feat_subpix_eps = 0.01

  ba_window = 10
  ba_max_iter = 20
  ba_max_error = 2.0
  ba_loss = 'soft_l1'



class SLAM:
  frames: list[Frame]
  points: list[Point]
  keyframe: KeyFrame | None
  keyframes: dict[int, KeyFrame]


  def __init__(self, config: Config, K: np.ndarray):
    self.K = K
    self.config = config

    self.pose = None
    self.initialized = False
    self.frames = []
    self.points = []
    self.keyframes = {}
    self.keyframe = None

  
  def track_lk(self):
    config = self.config
    frame = self.frames[-1]
    img_cur = cv2.cvtColor(frame.img, cv2.COLOR_RGB2GRAY)
    img_ref = cv2.cvtColor(self.frames[-2].img, cv2.COLOR_RGB2GRAY)

    uv_ref = np.array([pt.uv for pt in self.points])
    uv_cur, ok = klt(img_ref, img_cur, uv_ref,
                      win_size=config.lk_win_size, max_iter=config.lk_max_iter, eps=config.lk_eps,
                      max_level=config.lk_max_level, max_error=config.lk_max_error, max_fb_dist=config.lk_max_fb_dist)

    new_points = []
    for i in range(len(uv_ref)):
      if ok[i]:
        pt = self.points[i]
        pt.uv = pt.uvs[frame.id] = uv_cur[i]
        pt.last_frame = frame
        new_points.append(pt)
    self.points = new_points


  def get_relative_pose(self):    # relative pose to previous keyframe - we only use at initialization
    config, K = self.config, self.K
    kf_id = self.keyframe.id
    uv_cur = np.array([pt.uv for pt in self.points])
    uv_ref = np.array([pt.uvs[kf_id] for pt in self.points])

    parallax = np.linalg.norm(uv_cur - uv_ref, axis=1)    # TODO: rotation-compensated

    height, width = self.keyframe.frame.img.shape[:2]
    camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
    ransac_options = {'max_epipolar_error': config.rel_threshold_px, 'success_prob': config.rel_prob,
                      'min_iterations': config.rel_min_iter, 'max_iterations': config.rel_max_iter}
    bundle_options = {'loss_type': 'CAUCHY'}
    # for more options see: https://github.com/PoseLib/PoseLib#robust-estimation-and-non-linear-refinement
    pose, info = poselib.estimate_relative_pose(uv_cur, uv_ref, camera, camera, ransac_options, bundle_options)
    R, t = pose.R, pose.t
    inliers = np.array(info['inliers'], dtype=bool)
    self.points = [pt for i, pt in enumerate(self.points) if inliers[i]]
    return R, t, parallax[inliers].mean()


  def try_init(self, R, t, parallax):
    config = self.config
    if parallax < config.init_min_parallax:
      return

    kf_id = self.keyframe.id
    uv_cur = np.array([pt.uv for pt in self.points])
    uv_ref = np.array([pt.uvs[kf_id] for pt in self.points])

    T_cw_cur = inv_pose(make_pose(R, t))
    T_cw_ref = np.eye(4)
    points_3d, bad = triangulate(self.K, T_cw_cur, T_cw_ref, uv_cur, uv_ref,
                                 min_depth=config.init_min_depth, max_reproj_error_px=config.init_max_reproj_error_px)

    for i, pt in enumerate(self.points):
      assert not pt.is_3d
      if not bad[i]:
        pt.xyz = points_3d[i]

    self.initialized = True
    self.metrics['init_points'] = sum(bad == 0)
    self.keyframe.pose = T_cw_ref
    self.pose = T_cw_cur


  def get_absolute_pose(self):
    config, K = self.config, self.K
    pts_3d = [pt for pt in self.points if pt.is_3d]
    uv = [pt.uv for pt in pts_3d]
    xyz = [pt.xyz for pt in pts_3d]

    height, width = self.keyframe.frame.img.shape[:2]
    camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
    ransac_options = {'max_reproj_error': config.abs_max_reproj_error}
    bundle_options = {}
    pose, info = poselib.estimate_absolute_pose(uv, xyz, camera, ransac_options, bundle_options)
    R, t = pose.R, pose.t
    inliers = np.array(info['inliers'], dtype=bool)
    self.pose = make_pose(R, t)

    outliers = {pt for pt, inlier in zip(pts_3d, inliers) if not inlier}
    for pt in outliers:
      pt.uvs.pop(self.frames[-1].id)
    self.points = [pt for pt in self.points if pt not in outliers]


  def relocalize(self):
    config, K = self.config, self.K

    frame_cur = self.frames[-1]
    img_cur = cv2.cvtColor(frame_cur.img, cv2.COLOR_RGB2GRAY)
    height, width = img_cur.shape

    count = 0
    window = list(self.keyframes.values())[-config.reloc_window-1:-1]
    for kf in window:
      if sum(pt.is_3d for pt in self.points) > config.reloc_min_points: break
      img_ref = cv2.cvtColor(kf.frame.img, cv2.COLOR_RGB2GRAY)

      pts = [pt for pt in kf.points if pt.is_3d and frame_cur.id not in pt.uvs]
      if not pts: continue
      uv_cur, z = project(K, self.pose, np.array([pt.xyz for pt in pts]))

      w, h, u, v, b = width, height, uv_cur[:, 0], uv_cur[:, 1], config.reloc_border
      good = (z > 0) & (b <= u) & (u <= w - b) & (b <= v) & (v <= h - b)

      if not any(good): continue
      pts = [pt for pt, ok in zip(pts, good) if ok]
      uv_ref = np.array([pt.uvs[kf.id] for pt in pts])

      uv_cur, ok = klt(img_ref, img_cur, uv_ref, uv_cur[good],
                        win_size=config.reloc_win_size, max_iter=config.reloc_max_iter, eps=config.reloc_eps,
                        max_level=config.reloc_max_level, max_error=config.reloc_max_error, max_fb_dist=config.reloc_max_fb_dist)
      count += ok.sum()

      for i in range(len(pts)):
        if ok[i]:
          pt = pts[i]
          pt.uv = pt.uvs[frame_cur.id] = uv_cur[i]
          pt.last_frame = frame_cur
          self.points.append(pt)

    self.metrics['relocalize'] = int(count)


  def need_keyframe(self):
    self.metrics['pts'] = len(self.points)
    self.metrics['pts3d'] = sum(pt.is_3d for pt in self.points)

    if self.keyframe is None or self.initialized and len(self.keyframes) <= 1:
      return True
    if self.frames[-1].id <= self.keyframe.id + self.config.kf_min_gap:
      return False
    if self.initialized:
      n3d = sum(pt.is_3d for pt in self.points)
      n3d_kf = sum(pt.is_3d for pt in self.keyframe.points)
      if n3d < self.config.kf_min_3d_pts:
        return True
      if n3d < n3d_kf * self.config.kf_min_3d_pts_ratio:
        return True
    return len(self.points) < self.config.kf_min_pts


  def create_keyframe(self):
    keyframe = KeyFrame(self.points.copy(), self.frames[-1], self.pose)
    self.keyframe = keyframe
    self.keyframes[keyframe.id] = keyframe
    self.metrics['kf'] = True


  def create_3d_points(self):
    by_keyframe = collections.defaultdict(list)
    for pt in self.points:
      if pt.is_3d: continue
      by_keyframe[pt.keyframe].append(pt)

    T_cw_cur = self.pose
    for kf, pts in by_keyframe.items():
      uv_cur = np.array([pt.uv for pt in pts])
      uv_ref = np.array([pt.uvs[kf.id] for pt in pts])
      T_cw_ref = kf.pose
      points_3d, bad = triangulate(self.K, T_cw_cur, T_cw_ref, uv_cur, uv_ref,
                                   min_depth=self.config.tri_min_depth, max_reproj_error_px=self.config.tri_max_reproj_error_px)

      for i, pt in enumerate(pts):
        assert not pt.is_3d
        if not bad[i]:
          pt.xyz = points_3d[i]


  def detect_keypoints(self):
    config = self.config
    keyframe = self.keyframe
    img = cv2.cvtColor(keyframe.frame.img, cv2.COLOR_RGB2GRAY)
    mask = np.ones_like(img, dtype=np.uint8)
    for pt in keyframe.points:
      cv2.circle(mask, pt.uv.round().astype(int), radius=config.feat_mask_radius, color=0, thickness=-1)

    # TODO: support other detectors
    detector = cv2.GFTTDetector_create(config.feat_num, config.feat_min_quality, config.feat_min_dist, config.feat_block_size)
    kps = detector.detect(img, mask)
    uvs = np.array([kp.pt for kp in kps], dtype=np.float32)
    if config.feat_subpix:
      win_size = (config.feat_subpix_win_size, config.feat_subpix_win_size)
      criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, config.feat_subpix_max_iter, config.feat_subpix_eps)
      uvs = cv2.cornerSubPix(img, uvs, win_size, (-1, -1), criteria)
    
    for uv in uvs:
      u, v = uv
      color = keyframe.frame.img[int(v), int(u)]
      pt = Point(keyframe, last_frame=keyframe.frame, uv=uv, uvs={keyframe.id: uv}, color=color)
      keyframe.points.append(pt)
    self.points = list(keyframe.points)


  def local_bundle_adjustment(self, global_ba=False, remove_outliers=True):
    self.metrics['kf_pts'] = len(self.points)
    self.metrics['kf_pts3d'] = sum(pt.is_3d for pt in self.points)
    if len(self.keyframes) < 3: return
    config = self.config

    all_keyframes = list(self.keyframes.values())
    local_keyframes = all_keyframes if global_ba else all_keyframes[-config.ba_window:]
    pts = list(dict.fromkeys(pt for kf in local_keyframes for pt in kf.points if pt.is_3d))

    keyframes = {self.keyframes[frame_id] for pt in pts for frame_id in pt.uvs if frame_id in self.keyframes}
    keyframes = sorted(keyframes, key=lambda kf: kf.id)    # sort so that oldest keyframe is fixed
    keyframe_map = {kf.id: i for i, kf in enumerate(keyframes)}

    keyframe_indices = []
    point_indices = []
    uvs = []
    for point_index, pt in enumerate(pts):
      for frame_id, uv in pt.uvs.items():
        if frame_id in self.keyframes:
          keyframe_indices.append(keyframe_map[frame_id])
          point_indices.append(point_index)
          uvs.append(uv)

    keyframe_poses = np.array([kf.pose for kf in keyframes])
    fixed = np.array([kf not in local_keyframes for kf in keyframes])
    point_positions = np.array([pt.xyz for pt in pts])
    success, poses, positions, ok = optimize(self.K, keyframe_poses, fixed, point_positions,
                                             np.array(keyframe_indices), np.array(point_indices), np.array(uvs),
                                             max_iter=config.ba_max_iter, loss=config.ba_loss, max_error=config.ba_max_error)

    assert success, 'local bundle adjustment failed'

    for i in range(len(keyframes)):
      keyframes[i].pose = poses[i]
    for i in range(len(pts)):
      pts[i].xyz = positions[i]
    self.pose = self.keyframe.pose
    self.metrics['kf_pts'] = len(self.points)
    self.metrics['kf_pts3d'] = sum(pt.is_3d for pt in self.points)


    if remove_outliers:
      for i in range(len(uvs)):
        if not ok[i]:
          pt = pts[point_indices[i]]
          kf = keyframes[keyframe_indices[i]]
          pt.uvs.pop(kf.id)
          kf.points.remove(pt)
          if kf is self.keyframe:
            self.points.remove(pt)

      points_to_remove = {pt for pt in pts if sum(frame_id in self.keyframes for frame_id in pt.uvs) < 2}
      if points_to_remove:
        self.points = [pt for pt in self.points if pt not in points_to_remove]
        for kf in keyframes:
          kf.points = [pt for pt in kf.points if pt not in points_to_remove]
        for pt in points_to_remove:
          pt.xyz = None


  def on_frame(self, frame: Frame):
    self.frames.append(frame)
    self.metrics = {'i': frame.id}

    if self.points:
      self.track_lk()
      if not self.initialized:
        R, t, parallax = self.get_relative_pose()
        self.try_init(R, t, parallax)
      else:
        self.get_absolute_pose()
        self.relocalize()

    if self.need_keyframe():
      self.create_keyframe()
      self.create_3d_points()   # not needed immediately after we've initialized
      self.detect_keypoints()
      self.local_bundle_adjustment()

    age3d = np.median([pt.age for pt in self.points if pt.is_3d])
    self.metrics['age3d'] = int(age3d) if np.isfinite(age3d) else None



def main():
  parser = argparse.ArgumentParser('nanoslam')
  parser.add_argument('dataset')
  parser.add_argument('--rerun', action='store_true')
  args = parser.parse_args()

  config = Config()
  dataset = load_dataset(args.dataset)
  slam = SLAM(config, dataset.K)

  if args.rerun:
    import rerun
    rerun.init('nanoslam')
    rerun.set_time('time', sequence=0)

    rerun.log('ground_truth', rerun.LineStrips3D(dataset.gt[:, :3, 3]), static=True)
    K = dataset.K
    fx, fy, cx, cy = map(float, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    width, height = dataset.width, dataset.height
    rerun.log('camera/image',
              rerun.Pinhole(resolution=[width, height], focal_length=[fx, fy], principal_point=[cx, cy], image_plane_distance=0.5),
              static=True)
    rerun.spawn()

  slam_path = []
  gt_path = []
  elapsed = []
  for frame in tqdm(dataset, total=dataset.len):
    start_t = time.perf_counter()
    slam.on_frame(frame)
    elapsed.append(time.perf_counter() - start_t)
    if slam.pose is not None:
      slam_path.append(inv_pose(slam.pose)[:3, 3])
      gt_path.append(dataset.gt[frame.id, :3, 3])

    img = frame.img
    if slam.points:
      img = draw_points(img, slam.points, radius=1, thickness=-1)

    np_slam_path = np.array(slam_path)
    if len(slam_path) > 2:
      if frame.id < 300:   # TODO: only works for 'household', generalize to other datasets
        R_, s_, t_ = kabsch_umeyama(np.array(gt_path), np_slam_path)
      np_slam_path = s_ * (R_ @ np_slam_path.T).T + t_
      slam.metrics['error'] = round(float(np.linalg.norm(np_slam_path[-1] - gt_path[-1])), 3)

    print(slam.metrics)

    if args.rerun:
      rerun.set_time('time', sequence=frame.id)
      rerun.log('slam', rerun.LineStrips3D(np_slam_path))
      rerun.log('camera', rerun.Transform3D(translation=frame.gt[:3, 3], mat3x3=frame.gt[:3, :3]))
      rerun.log('camera/image/image', rerun.Image(img, color_model='rgb').compress(99))

  print(f'frame latency: average={np.mean(elapsed):0.3f} max={np.max(elapsed):0.3f}')



if __name__ == '__main__':
  main()
