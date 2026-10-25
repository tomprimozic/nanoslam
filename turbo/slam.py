'''
nanoslam turbo: monocular SLAM, geometry only.

The map (map.py) is a set of flat numpy tables: keyframes, points (candidate / mapped) and observations.

Per frame: KLT (map points seeded with their predicted projection), PnP, relocalization by projection when weak.
Per keyframe: re-measure tracked points against their reference keyframe (no template drift), search by projection
for all other map points (warped patches, fusion of duplicates), triangulation, local BA, loop detection by
projection of old points (Sim3 pose graph + global BA), point and keyframe culling, detection in empty cells.
'''
import time
import threading
import collections

import cv2
import numpy as np
import poselib
import scipy.spatial
from scipy.spatial.transform import Rotation

import lk
import optim
from map import Map, Table, CANDIDATE, MAPPED, DEAD, group_starts
from geometry import (make_pose, inv_pose, orthonormalize, transform, camera_center, project, unproject, sim3,
                      sim3_inv, sim3_to_se3, rotation_angle)


class Config:
  focal_scale = 1.0                 # diagnostics: simulated calibration error
  clahe = 3.0                       # CLAHE clip limit (0 = off)

  # frame-to-frame KLT
  klt_win = 9
  klt_levels = 3
  klt_iters = 30
  klt_fb = 0.5                      # px, forward-backward check
  motion_model = 'constant'         # pose prediction: 'none', 'constant' (velocity), 'accel' (constant acceleration)
  phase_prior = True                # global image shift by phase correlation corrects the KLT initial guesses
  phase_min_response = 0.05         # phase correlation peak needed to trust the shift
  phase_min_correction = 2.0        # px, only correct the prediction by larger differences

  # detection: the strongest min-eigenvalue corner of every empty grid cell
  cells = 20                        # across the image width
  min_quality = 0.001               # relative to the strongest corner
  detect_blur = True                # 3x3 blur before the corner score (OV2SLAM)

  # initialization
  init_mode = 'window'              # 'window': batch BA over the frames since the anchor keyframe; 'twoview'
  init_parallax = 2.5               # deg, median (not rotation compensated), twoview
  init_window = 20                  # frames in the window BA (subsampled)
  init_check_parallax = 0.5         # deg, median parallax before a window init is attempted
  init_ratio = 2.0                  # translation established: likelihood ratio full vs rotation-only > this x extra params
  rotation_only_poses = True        # before translation is established: emit rotation-only poses (position held)
  init_min_points = 50
  init_min_ratio = 0.5              # fraction of inliers with parallax (rejects rotation-dominated motion)

  # tracking
  pnp_px = 3.0
  # adaptive thresholds: the live noise level = robust scale of the PnP residuals / noise_ref
  adaptive_pnp = False              # PnP inliers re-decided at the live noise level
  adaptive_map = False              # keyframes keep their noise level: BA weights, triangulation / epipolar thresholds
  noise_ref = 0.67                  # px
  pnp_min_inliers = 15
  drop_pnp_outliers = True          # False: keep tracking them (they are excluded from this frame's pose only)
  epipolar_px = 3.0
  track_local_map = False           # every frame: re-find untracked points of the local keyframes (neutral, slower)
  local_map_radius = 3.0            # px
  reloc_min_3d = 50                 # relocalize (search the map with a wide window) below this many 3D tracks
  reloc_keyframes = 8               # keyframe poses tried per frame while lost
  reinit = False                    # start a new map segment when relocalization keeps failing
  reinit_after = 10                 # frames lost before that

  # triangulation
  tri_min_angle = 1.0               # deg
  tri_max_px = 3.0

  # keyframes
  kf_parallax = 1.75                # deg, rotation-compensated median
  kf_tracked_ratio = 0.7
  kf_min_3d = 40

  # local BA
  ba_window = 10
  ba_min_covisible = 15
  ba_iters = 10
  ba_loss = 'huber'                 # or 'cauchy'

  # patch search (search by projection, re-measurement)
  patch_half = 4                    # 9x9
  patch_iters = 20
  search_max_angle = 35.0           # deg, max viewing angle difference to the source keyframe
  search_max_scale = 1.8            # max viewing distance ratio
  search_radius = 4.0               # px
  search_min_ncc = 0.75
  remeasure = True                  # re-measure tracked points from their reference keyframe
  remeasure_radius = 2.0
  remeasure_drop_slipped = False    # drop tracks whose re-measurement is further than remeasure_radius
  remeasure_min_ncc = 0.8
  fuse_px = 2.0                     # merge points re-found within this distance of another track

  # loops (old points searched with a wide window)
  loop = True
  loop_age = 20                     # keyframes since a point was last seen
  loop_top_level = 2
  loop_search_px = 6                # brute-force search on the top level (x 2^top_level px)
  loop_min_matches = 30
  loop_min_inliers = 25
  loop_every = 3                    # search old points at every n-th keyframe
  loop_fuse_ba = True               # local BA right after fusing old points (else: at the next keyframe)
  loop_min_correction = 5.0         # px, mean disagreement of the old points needed for a pose graph correction
  essential_edges = 5               # covisibility edges per keyframe in the pose graph
  global_ba_iters = 3
  global_ba_delay = 60              # frames: global BA after a loop runs in the background, merged this much later (0 = sync)
  final_ba_iters = 20

  # culling
  cull_points = True
  cull_min_ratio = 0.25             # found / visible
  cull_keyframes = True
  cull_kf_shared = 0.8              # near-duplicate keyframes: fraction of shared points,
  cull_kf_angle = 10.0              # deg, and orientation difference (and closer than the keyframe parallax)
  cull_old_redundant = False        # also drop old keyframes whose points are (mostly) seen by >= 3 others
  cull_old_age = 20                 # keyframes
  cull_old_redundancy = 0.9
  ransac_seed = 0


class SLAM:
  def __init__(self, config: Config, K, width, height):
    self.c = c = config
    K = K.copy()
    K[0, 0] *= c.focal_scale
    K[1, 1] *= c.focal_scale
    self.K = K
    self.f = np.sqrt(K[0, 0] * K[1, 1])
    self.width, self.height = width, height
    self.camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
    self.clahe = cv2.createCLAHE(c.clahe, (max(width // 50, 1), max(height // 50, 1))) if c.clahe else None
    self.map = Map()
    self.tr_uv = np.zeros((0, 2), np.float32)    # tracks: current pixel and point id
    self.tr_pt = np.zeros(0, np.int64)
    self.tr_kf_uv = np.zeros((0, 2))             # pixel at the last keyframe (for parallax), NaN if found since
    self.initialized = False
    self.T_cw = np.eye(4)
    self.T_good = np.eye(4)           # last successfully tracked pose
    self.noise = 1.0                  # live noise level: residual scale / noise_ref
    self.velocity = np.eye(4)
    self.acceleration = np.eye(4)
    self.counts = collections.Counter()   # diagnostics
    self.last_kf = -1
    self.segment = 0                  # map segment (a new one starts when tracking is lost and we re-initialize)
    self.initializing = True          # waiting for two-view initialization (of the first map or a new segment)
    self.good_depth = 1.0             # median depth of the tracked points at the last good frame
    self.window = []                  # while initializing: (frame, point ids, pixels) of every frame since the anchor
    self.pre_kfs = []                 # keyframes made before translation was established (rotation-only poses)
    self.anchor = -1                  # the one the translation test is relative to
    self.init_n_tracks = 0            # tracks right after the last of them
    self.lost = 0                     # frames since tracking was lost
    self.kf_n3d = 0                   # tracked map points right after the last keyframe
    self.segment_cursor = 0           # round-robin position of the loop search across segments
    self.frame = -1
    self.pending_ba = None            # global BA running in the background (c.global_ba_delay)
    self.img = self.pyr = None
    self.hann = None
    # tracked frames: pose relative to a reference keyframe (so it follows map corrections), and as estimated online
    self.traj = Table(frame=(np.int64, (), -1), ref=(np.int64, (), -1), T=(np.float64, (4, 4), 0),
                      online=(np.float64, (4, 4), np.nan))
    # per frame diagnostics (intrinsic: no ground truth involved)
    self.stats = Table(frame=(np.int64, (), -1), tracks=(np.int32, (), 0), mapped=(np.int32, (), 0),
                       far=(np.int32, (), 0), inliers=(np.int32, (), 0), sigma=(np.float32, (), np.nan),
                       phase_response=(np.float32, (), np.nan), init_ratio=(np.float32, (), np.nan),
                       flow=(np.float32, (), np.nan), blur=(np.float32, (), np.nan), lost=(np.int32, (), 0),
                       keyframe=(bool, (), False), loop_search=(bool, (), False), loop_closed=(bool, (), False),
                       keyframes=(np.int32, (), 0), points=(np.int32, (), 0),
                       ms=(np.float32, (), np.nan))
    self.diag = {}
    self.events = []
    self.timing = collections.defaultdict(float)

  # ------------------------------------------------------------------------------------------------- helpers

  def in_image(self, uv, border=0):
    return ((uv[:, 0] >= border) & (uv[:, 0] < self.width - 1 - border)
            & (uv[:, 1] >= border) & (uv[:, 1] < self.height - 1 - border))

  def px(self, deg):
    return self.f * np.tan(np.deg2rad(deg))

  def mapped_tracks(self):
    return self.map.pts.state[self.tr_pt] == MAPPED

  def keep_tracks(self, keep):
    self.tr_uv, self.tr_pt, self.tr_kf_uv = self.tr_uv[keep], self.tr_pt[keep], self.tr_kf_uv[keep]

  def add_tracks(self, pt, uv, kf_uv):
    self.tr_pt = np.r_[self.tr_pt, pt]
    self.tr_uv = np.r_[self.tr_uv, uv.astype(np.float32)]
    self.tr_kf_uv = np.r_[self.tr_kf_uv, kf_uv]

  def timed(self, name):
    slam = self

    class Timer:
      def __enter__(self):
        self.t = time.perf_counter()

      def __exit__(self, *args):
        slam.timing[name] += time.perf_counter() - self.t
    return Timer()

  # ------------------------------------------------------------------------------------------------- KLT

  def klt(self, img_a, img_b, pts, guess=None, levels=None):
    c = self.c
    if len(pts) == 0:
      return np.zeros((0, 2), np.float32), np.zeros(0, bool)
    pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, c.klt_iters, 0.01)
    flags, nxt = 0, None
    if guess is not None:
      nxt = np.ascontiguousarray(guess, np.float32).reshape(-1, 1, 2).copy()
      flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, pts, nxt, winSize=(c.klt_win, c.klt_win),
                                          maxLevel=c.klt_levels if levels is None else levels,
                                          criteria=criteria, flags=flags)
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(img_b, img_a, nxt, pts.copy(), winSize=(c.klt_win, c.klt_win),
                                             maxLevel=0, criteria=criteria, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    nxt = nxt.reshape(-1, 2)
    fb = np.linalg.norm(back.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb <= c.klt_fb) & self.in_image(nxt, 2)
    return nxt, ok

  def global_shift(self, pyr):
    '''image shift from the previous frame (phase correlation on pyramid level 2, in level-0 pixels) or None'''
    a, b = self.pyr[2].astype(np.float32), pyr[2].astype(np.float32)
    if self.hann is None:
      self.hann = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, self.hann)
    self.diag['phase_response'] = float(response)
    return np.array([dx, dy]) * 4 if response >= self.c.phase_min_response else None

  def track(self, img, T_pred, pyr):
    '''two passes: mapped points from their predicted projection (1 level), then everything else'''
    c = self.c
    n = len(self.tr_pt)
    new = np.zeros((n, 2), np.float32)
    ok = np.zeros(n, bool)
    m = np.flatnonzero(self.mapped_tracks())
    shift = self.global_shift(pyr) if c.phase_prior else None
    if len(m):
      guess, z = project(self.K, T_pred, self.map.pts.X[self.tr_pt[m]])
      bad = (z <= 0) | ~np.isfinite(guess).all(axis=1)
      guess[bad] = self.tr_uv[m[bad]]
      if shift is not None:   # the image moved differently than predicted: correct the prediction by the difference
        correction = shift - np.median(guess - self.tr_uv[m], axis=0)
        if np.linalg.norm(correction) > c.phase_min_correction:
          guess = guess + correction
      new[m], ok[m] = self.klt(self.img, img, self.tr_uv[m], guess, levels=1)
    rest = np.flatnonzero(~ok)
    rest_guess = self.tr_uv[rest] + shift if shift is not None else None
    new[rest], ok[rest] = self.klt(self.img, img, self.tr_uv[rest], rest_guess)
    if ok.any():
      self.diag['flow'] = float(np.median(np.linalg.norm(new[ok] - self.tr_uv[ok], axis=1)))
    self.tr_uv = new
    self.keep_tracks(ok)

  # ------------------------------------------------------------------------------------------------- detection

  def detect(self, kf):
    c = self.c
    img = self.img
    h, w = img.shape
    cell = int(np.ceil(w / c.cells))
    ny, nx = -(-h // cell), -(-w // cell)
    occupied = np.zeros((ny, nx), bool)
    uv = self.tr_uv.astype(int)
    occupied[np.clip(uv[:, 1] // cell, 0, ny - 1), np.clip(uv[:, 0] // cell, 0, nx - 1)] = True
    score = cv2.cornerMinEigenVal(cv2.GaussianBlur(img, (3, 3), 0) if c.detect_blur else img, 3, 3)
    b = 8
    score[:b], score[-b:], score[:, :b], score[:, -b:] = 0, 0, 0, 0
    padded = np.zeros((ny * cell, nx * cell), np.float32)
    padded[:h, :w] = score
    blocks = padded.reshape(ny, cell, nx, cell).transpose(0, 2, 1, 3).reshape(ny, nx, cell * cell)
    best = blocks.argmax(axis=2)
    best_score = np.take_along_axis(blocks, best[..., None], axis=2)[..., 0]
    cy, cx = np.nonzero(~occupied & (best_score > c.min_quality * score.max()))
    if len(cy) == 0:
      return
    by, bx = np.divmod(best[cy, cx], cell)
    pts = np.stack([cx * cell + bx, cy * cell + by], axis=1).astype(np.float32)
    pts = cv2.cornerSubPix(img, pts.reshape(-1, 1, 2), (3, 3), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)).reshape(-1, 2)
    ids = self.map.add_points(kf, pts.astype(np.float64))
    self.add_tracks(ids, pts, pts.astype(np.float64))

  # ------------------------------------------------------------------------------------------------- keyframes

  def add_keyframe(self, T_cw):
    m = self.map
    kf = m.kfs.add(1, T=T_cw, frame=self.frame, parent=self.last_kf, segment=self.segment, noise=self.noise)[0]
    if m.images is None:
      m.images = lk.ImageStore(self.pyr)
    m.kfs.slot[kf] = m.images.add(self.pyr)
    m.add_obs(kf, self.tr_pt, self.tr_uv.astype(np.float64))
    self.tr_kf_uv = self.tr_uv.astype(np.float64)
    self.last_kf = kf
    return kf

  def need_keyframe(self):
    c = self.c
    if self.frame - self.map.kfs.frame[self.last_kf] < 1 or self.lost:
      return False
    n3d = int(self.mapped_tracks().sum())
    if n3d < c.kf_min_3d:
      return True
    parallax = self.parallax()
    return parallax >= self.px(c.kf_parallax) or (n3d < c.kf_tracked_ratio * self.kf_n3d
                                                   and parallax >= self.px(c.kf_parallax) / 3)

  def parallax(self):
    '''rotation-compensated median parallax (px) of the tracks w.r.t. the last keyframe'''
    has = np.isfinite(self.tr_kf_uv).all(axis=1)
    if not has.any():
      return 0.0
    R = self.T_cw[:3, :3] @ self.map.kfs.T[self.last_kf, :3, :3].T
    b = unproject(self.K, self.tr_kf_uv[has]) @ R.T
    uv = b[:, :2] / b[:, 2:] * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
    return float(np.median(np.linalg.norm(uv - self.tr_uv[has], axis=1)))

  # ------------------------------------------------------------------------------------------------- initialization

  def try_initialize(self):
    self.window.append((self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64)))
    if self.c.init_mode == 'window':
      return self.try_initialize_window()
    return self.try_initialize_twoview()

  def rotation_only(self, uv_ref, uv, iterations=3):
    '''
    rotation R (x = R x_ref) that best explains the bearings of a pure rotation (Wahba, IRLS with a Huber weight);
    returns R and the re-projection residuals [px] of the rotation-only model
    '''
    b0 = unproject(self.K, uv_ref)
    b0 /= np.linalg.norm(b0, axis=1, keepdims=True)
    b1 = unproject(self.K, uv)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    w = np.ones(len(b0))
    for _ in range(iterations):
      U, _, VT = np.linalg.svd((b1 * w[:, None]).T @ b0)
      R = U @ np.diag([1, 1, np.linalg.det(U @ VT)]) @ VT
      p = b0 @ R.T
      res = np.linalg.norm(p[:, :2] / p[:, 2:] * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2] - uv, axis=1)
      w = np.where(res <= optim.HUBER, 1.0, optim.HUBER / np.maximum(res, 1e-9))
    return R, res

  def try_initialize_window(self):
    '''
    batch BA over the frames since the anchor keyframe (subsampled), started from the rotation-only rotations and the
    two-view translation direction; initialized once the rotation-only model clearly explains the tracks worse than the
    full one (residual scale ratio, a likelihood ratio without the noise level) and enough points have parallax
    '''
    c, m = self.c, self.map
    kf0, sub, ref = self.init_anchor()
    if sub.sum() < c.init_min_points:
      return
    tr_pt = self.tr_pt[sub]
    cur = self.tr_uv[sub].astype(np.float64)
    if np.median(np.linalg.norm(cur - ref, axis=1)) < self.px(c.init_check_parallax):
      return
    pose, info = poselib.estimate_relative_pose(ref, cur, self.camera, self.camera,
                                                {'max_epipolar_error': 1.0, 'success_prob': 0.9999,
                                                 'seed': c.ransac_seed}, {})
    inl = np.array(info['inliers'], bool)
    pts = tr_pt[inl]
    n = len(pts)
    if n < c.init_min_points:
      return
    T_last = make_pose(pose.R, pose.t)
    row_all = np.full(len(m.pts), -1)
    row_all[tr_pt] = np.arange(len(tr_pt))
    ref_all = ref
    X, _, _, depth = optim.triangulate(self.K, np.array([np.eye(4), T_last]), np.r_[np.zeros(n, int), np.ones(n, int)],
                                       np.r_[np.arange(n), np.arange(n)], np.r_[ref[inl], cur[inl]], n)
    ok = np.isfinite(X).all(axis=1) & (depth > 0)
    if ok.sum() < c.init_min_points:
      return
    # robust depths: points behind / too far are placed at a large depth along their ray
    med = np.median(X[ok, 2])
    rays = unproject(self.K, ref[inl])
    z = np.clip(np.where(ok, X[:, 2], 100 * med), 0.05 * med, 100 * med)
    X = rays * z[:, None]
    # frames of the window: evenly spread in parallax (a long still start collapses to a few frames)
    par_all = np.array([np.median(np.linalg.norm(xy[k >= 0] - ref_all[k[k >= 0]], axis=1)) if (k := row_all[ids]).max() >= 0
                        else 0.0 for _, ids, xy in self.window])
    targets = np.linspace(0, par_all[-1], c.init_window)
    sel = np.unique(np.r_[0, np.searchsorted(np.maximum.accumulate(par_all), targets), len(self.window) - 1])
    sel = sel[sel < len(self.window)]
    frames = [self.window[i] for i in sel]
    row = np.full(len(m.pts), -1)
    row[pts] = np.arange(n)
    cam, pt, uv = [], [], []
    for j, (_, ids, xy) in enumerate(frames):
      k = row[ids]
      have = k >= 0
      cam.append(np.full(have.sum(), j)); pt.append(k[have]); uv.append(xy[have])
    cam, pt, uv = np.concatenate(cam), np.concatenate(pt), np.concatenate(uv)
    # initial poses: rotation-only rotations, translations along the two-view baseline in proportion to the parallax
    ref_of = ref[inl][pt]
    T = np.tile(np.eye(4), (len(frames), 1, 1))
    res_rot = []
    par = []
    for j in range(len(frames)):
      o = cam == j
      R, res = self.rotation_only(ref_of[o], uv[o])
      T[j, :3, :3] = R
      res_rot.append(res)
      par.append(np.median(np.linalg.norm(uv[o] - ref_of[o], axis=1)))
    par = np.array(par)
    T[:, :3, 3] = T_last[:3, 3] * (par / max(par[-1], 1e-9))[:, None]
    T[-1] = T_last
    fixed = np.zeros(len(frames), bool)
    fixed[0] = True
    baseline = T_last[:3, :3] @ camera_center(T_last)
    with self.timed('ba'):
      T, X, e2 = optim.bundle_adjust(self.K, T, fixed, X, cam, pt, uv, 15,
                                     scale_fix=(len(frames) - 1, int(np.argmax(np.abs(baseline)))))
    # translation test: likelihood ratio of the full model vs the rotation-only one (robust costs in units of the
    # full model's residual scale), against the parameters the full model has in addition (translations + depths)
    sigma = max(1.4826 * np.median(np.sqrt(e2[np.isfinite(e2)])), 0.1)
    rr = np.concatenate(res_rot) / sigma
    rf = np.sqrt(np.minimum(e2, 1e6)) / sigma
    huber = lambda r: np.where(r <= optim.HUBER, 0.5 * r ** 2, optim.HUBER * (r - 0.5 * optim.HUBER))
    dF = float(huber(rr).sum() - huber(rf).sum())
    extra = 3 * (len(frames) - 1) + n
    # per point: inlier in the first and the last frame (a glitch in an intermediate frame is just a lost observation),
    # in front, parallax
    bad = np.zeros(n, bool)
    ends = (cam == 0) | (cam == len(frames) - 1)
    np.logical_or.at(bad, pt[ends], ~(e2[ends] <= optim.CHI2_2D))
    centers = camera_center(T)
    angle = np.zeros(n)
    for j in range(1, len(frames)):
      a, b = X - centers[0], X - centers[j]
      cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
      angle = np.maximum(angle, np.arccos(np.clip(cos, -1, 1)))
    good = ~bad & (X[:, 2] > 0)
    good_angle = good & (angle >= np.deg2rad(c.tri_min_angle))
    self.diag['init_ratio'] = dF / extra
    if (dF < c.init_ratio * extra or good_angle.sum() < c.init_min_points
        or good_angle.sum() < c.init_min_ratio * good.sum()):
      return
    s = self.good_depth / np.median(X[good_angle, 2])
    T[:, :3, 3] *= s
    X = X * s
    T0 = m.kfs.T[kf0]
    self.T_cw = self.T_good = T[-1] @ T0
    self.velocity = np.eye(4)
    keep = np.zeros(len(m.pts), bool)
    keep[pts[good]] = True
    self.keep_tracks(keep[self.tr_pt])
    m.kill(np.setdiff1d(m.points_of(kf0, CANDIDATE), self.tr_pt))
    kf1 = self.add_keyframe(self.T_cw)
    good_pts = pts[good_angle]
    m.pts.X[good_pts] = transform(inv_pose(T0), X[good_angle])
    m.pts.state[good_pts] = MAPPED
    # hindsight: the frames since the anchor get poses from the window solution (refined on its points)
    Xw = X[good_angle]
    for f, ids, xy in self.window[1:-1]:
      k = row[ids]
      have = (k >= 0) & good_angle[np.maximum(k, 0)]
      if have.sum() < c.pnp_min_inliers:
        continue
      j = np.searchsorted(np.array([fr[0] for fr in frames]), f)
      T_init = T[min(j, len(frames) - 1)]
      T_f, _ = optim.refine_pose(self.K, T_init, Xw[np.searchsorted(np.flatnonzero(good_angle), k[have])], xy[have])
      i = np.flatnonzero(self.traj.frame == f)
      if len(i):
        self.traj.T[i] = T_f @ T0 @ inv_pose(m.kfs.T[self.traj.ref[i]])
    self.window = []
    self.drop_pre_keyframes(kf0, kf1)
    self.initialized, self.initializing, self.lost = True, False, 0
    self.local_ba(kf1)
    self.detect(kf1)
    self.kf_n3d = int(self.mapped_tracks().sum())
    self.events.append({'type': 'init', 'frame': self.frame, 'segment': self.segment, 'points': int(len(good_pts)),
                        'frames': len(frames), 'ratio': dF / extra})

  def try_initialize_twoview(self):
    '''
    two-view initialization from the tracks of the anchor keyframe (see init_anchor; the first keyframe of a new segment
    is at the last good pose): relative pose, triangulation, scale = median depth 1 (first map) / the depth
    seen before tracking was lost (new segment)
    '''
    c, m = self.c, self.map
    kf0, sub, ref = self.init_anchor()
    if sub.sum() < c.init_min_points:
      return
    tr_pt = self.tr_pt[sub]
    cur = self.tr_uv[sub].astype(np.float64)
    if np.median(np.linalg.norm(cur - ref, axis=1)) < self.px(c.init_parallax):
      return
    pose, info = poselib.estimate_relative_pose(ref, cur, self.camera, self.camera,
                                                {'max_epipolar_error': 1.0, 'success_prob': 0.9999,
                                                 'seed': c.ransac_seed}, {})
    inliers = np.array(info['inliers'], bool)
    T_10 = make_pose(pose.R, pose.t)
    n = len(ref)
    X, angle, err, depth = optim.triangulate(self.K, np.array([np.eye(4), T_10]), np.r_[np.zeros(n, int), np.ones(n, int)],
                                             np.r_[np.arange(n), np.arange(n)], np.r_[ref, cur], n)
    good = inliers & (depth > 0) & (err <= c.tri_max_px)
    good_angle = good & (angle >= np.deg2rad(c.tri_min_angle))
    if good_angle.sum() < c.init_min_points or good_angle.sum() < c.init_min_ratio * inliers.sum():
      return
    s = self.good_depth / np.median(X[good_angle, 2])
    T_10[:3, 3] *= s
    T0 = m.kfs.T[kf0]
    self.T_cw = self.T_good = T_10 @ T0
    self.velocity = np.eye(4)
    keep = np.zeros(len(self.tr_pt), bool)
    keep[np.flatnonzero(sub)[inliers]] = True
    self.keep_tracks(keep)
    m.kill(np.setdiff1d(m.points_of(kf0, CANDIDATE), self.tr_pt))
    kf1 = self.add_keyframe(self.T_cw)
    good_pts = tr_pt[good_angle]
    m.pts.X[good_pts] = transform(inv_pose(T0), X[good_angle] * s)
    m.pts.state[good_pts] = MAPPED
    self.drop_pre_keyframes(kf0, kf1)
    self.initialized, self.initializing, self.lost = True, False, 0
    self.local_ba(kf1)
    self.detect(kf1)
    self.kf_n3d = int(self.mapped_tracks().sum())
    self.events.append({'type': 'init', 'frame': self.frame, 'segment': self.segment, 'points': int(len(good_pts))})

  def reset(self):
    self.map = Map()
    self.tr_uv, self.tr_pt, self.tr_kf_uv = np.zeros((0, 2), np.float32), np.zeros(0, np.int64), np.zeros((0, 2))
    self.last_kf = -1
    kf = self.add_keyframe(np.eye(4))
    self.detect(kf)
    self.window = [(self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64))]
    self.pre_kfs, self.anchor, self.init_n_tracks = [kf], kf, len(self.tr_pt)

  def pre_keyframe(self):
    '''
    before translation is established: a keyframe at the rotation-only pose (position held) with new corners, so that
    tracking goes on through rotations
    '''
    kf = self.add_keyframe(self.T_cw)
    self.detect(kf)
    self.window[-1] = (self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64))
    self.pre_kfs.append(kf)
    self.init_n_tracks = len(self.tr_pt)
    self.events.append({'type': 'pre_keyframe', 'frame': self.frame})

  def init_anchor(self):
    '''
    the keyframe the translation test is relative to (the window starts there): the first pre-translation keyframe
    until fewer than init_min_points of its points are tracked, then the newest one (fresh points); returns it, the mask
    of the tracks it observed and their pixels in it
    '''
    c, m = self.c, self.map
    if len(self.tr_pt) < max(c.init_min_points, c.kf_tracked_ratio * self.init_n_tracks):
      if (T := self.rotation_only_pose()) is not None:
        self.T_cw = T
      self.pre_keyframe()
    def observed(kf):
      o = m.obs_of(kfs=[kf])
      uv = np.full((len(m.pts), 2), np.nan)
      uv[m.obs.pt[o]] = m.obs.uv[o]
      return uv[self.tr_pt]
    uv = observed(self.anchor)
    if np.isfinite(uv[:, 0]).sum() < c.init_min_points and self.anchor != self.last_kf:
      self.anchor = self.last_kf
      self.window = [w for w in self.window if w[0] >= m.kfs.frame[self.anchor]]
      uv = observed(self.anchor)
    sub = np.isfinite(uv[:, 0])
    return self.anchor, sub, uv[sub]

  def drop_pre_keyframes(self, kf0, kf1):
    '''translation established between kf0 and kf1: the other pre-translation keyframes (whose positions were only
    assumed) leave the map; their frames keep their rotation-only poses'''
    m = self.map
    for k in self.pre_kfs:
      if k != kf0 and m.kfs.alive[k]:
        m.remove_obs(m.obs_of(kfs=[k]))
        m.kfs.alive[k] = False
        m.images.release(m.kfs.slot[k])
    m.kfs.parent[kf0] = m.kfs.parent[self.pre_kfs[0]]
    m.kfs.parent[kf1] = kf0
    self.pre_kfs = []

  def rotation_only_pose(self):
    '''the current pose if the camera only rotated since the last keyframe (None: too few tracks)'''
    if len(self.tr_pt) < 8:
      return None
    R, _ = self.rotation_only(self.tr_kf_uv, self.tr_uv.astype(np.float64))
    return make_pose(R, np.zeros(3)) @ self.map.kfs.T[self.last_kf]

  def start_segment(self, new=True):
    '''tracking is lost for good: start a new map segment at the last good pose (joined to the map by a loop later)'''
    self.keep_tracks(np.zeros(len(self.tr_pt), bool))
    if new:
      self.segment = int(self.map.kfs.segment.max()) + 1
      self.events.append({'type': 'segment', 'frame': self.frame, 'segment': self.segment})
    self.initializing = True
    kf = self.add_keyframe(self.T_good)
    self.detect(kf)
    self.window = [(self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64))]
    self.pre_kfs, self.anchor, self.init_n_tracks = [kf], kf, len(self.tr_pt)

  def abort_segment(self):
    '''relocalized on the map while initializing a new segment: drop it'''
    m = self.map
    for k in self.pre_kfs[::-1]:
      self.last_kf = int(m.kfs.parent[k])
      self.remove_keyframe(k)
    self.pre_kfs = []
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)
    self.segment = int(m.kfs.segment[self.last_kf])
    self.initializing = False

  # ------------------------------------------------------------------------------------------------- tracking

  def pnp(self, uv, X, T_guess=None, threshold=None):
    '''RANSAC PnP (+ Huber refinement); returns T_cw, inlier mask'''
    c = self.c
    initial = None
    if T_guess is not None:
      initial = poselib.CameraPose()
      initial.R, initial.t = T_guess[:3, :3], T_guess[:3, 3]
    pose, info = poselib.estimate_absolute_pose(np.asarray(uv, np.float64), X, self.camera,
                                                {'max_reproj_error': threshold or self.px_threshold(c.pnp_px, c.adaptive_pnp),
                                                 'seed': c.ransac_seed, 'min_iterations': 50},
                                                {'loss_type': 'HUBER', 'loss_scale': 1.0}, initial)
    return make_pose(pose.R, pose.t), np.array(info['inliers'], bool)

  def px_threshold(self, px, adaptive):
    '''a pixel threshold, at the current noise level if adaptive'''
    return px * self.noise if adaptive else px

  def estimate_pose(self, T_pred):
    c = self.c
    m3 = np.flatnonzero(self.mapped_tracks())
    if len(m3) < c.pnp_min_inliers:
      return False
    X, uv = self.map.pts.X[self.tr_pt[m3]], self.tr_uv[m3]
    T, inliers = self.pnp(uv, X, T_pred, threshold=c.pnp_px * max(self.noise, 1) if c.adaptive_pnp else None)
    if inliers.sum() < c.pnp_min_inliers:
      return False
    # robust scale of the reprojection residuals of all tracked map points (MAD per axis)
    proj, _ = project(self.K, T, X)
    sigma = float(1.4826 * np.median(np.abs(proj - uv)))
    self.diag['sigma'] = sigma
    self.noise = float(np.clip(sigma / c.noise_ref, 0.5, 3.0))
    if c.adaptive_pnp:   # re-decide the inliers at the measured noise level and refine
      inliers = np.linalg.norm(proj - uv, axis=1) <= self.px_threshold(c.pnp_px, True)
      if inliers.sum() < c.pnp_min_inliers:
        return False
      T, _ = optim.refine_pose(self.K, T, X[inliers], uv[inliers], delta=optim.HUBER * self.noise)
    self.T_cw = T
    self.diag['inliers'] = int(inliers.sum())
    if c.drop_pnp_outliers:
      keep = np.ones(len(self.tr_pt), bool)
      keep[m3[~inliers]] = False
      self.keep_tracks(keep)
    return True

  def relocalize(self, T_guess):
    '''
    find map points with a wide window around their projection, then PnP. When tracking is lost (e.g. fast rotation
    with motion blur) the prediction can be far off: also try the last good pose rotated by +-15 / 30 deg (pan, tilt)
    and, round-robin, the poses of all keyframes (nearest first), each with its own points and an even wider window
    '''
    c, m = self.c, self.map
    local = self.local_keyframes(self.last_kf, 2 * c.ba_window)
    hypotheses = [(T_guess, local)]
    top_level, search_px = c.loop_top_level, c.loop_search_px
    if self.lost:
      top_level, search_px = 3, 8
      alive = np.flatnonzero(m.kfs.alive)
      near = alive[np.argsort(np.linalg.norm(camera_center(m.kfs.T[alive]) - camera_center(self.T_good), axis=1))]
      if self.lost <= 3:
        for axis in ([0, 1, 0], [1, 0, 0]):
          for deg in (15, -15, 30, -30):
            R = Rotation.from_rotvec(np.deg2rad(deg) * np.array(axis)).as_matrix()
            hypotheses.append((make_pose(R, np.zeros(3)) @ self.T_good, near[:c.ba_window]))
      for k in near[(self.lost * c.reloc_keyframes + np.arange(c.reloc_keyframes)) % len(near)]:
        hypotheses.append((m.kfs.T[k], [k]))
    m3 = np.flatnonzero(self.mapped_tracks())
    best = (0, None, None, None, None)
    for T, kfs in hypotheses:
      candidates = np.setdiff1d(np.unique(m.obs.pt[m.obs_of(kfs=kfs)]), self.tr_pt)
      pt, uv = self.search(T, self.pyr, candidates, top_level, search_px, np.inf)
      if len(pt) + len(m3) < c.reloc_min_3d // 2:
        continue
      T_new, inliers = self.pnp(np.r_[self.tr_uv[m3], uv], m.pts.X[np.r_[self.tr_pt[m3], pt]], T)
      if inliers.sum() > best[0]:
        best = (inliers.sum(), T_new, inliers, pt, uv)
      if best[0] >= 2 * c.reloc_min_3d:
        break
    n, T_new, inliers, pt, uv = best
    if n < c.reloc_min_3d // 2:
      return False
    self.T_cw = T_new
    keep = np.ones(len(self.tr_pt), bool)
    keep[m3[~inliers[:len(m3)]]] = False
    self.keep_tracks(keep)
    found = inliers[len(m3):]
    self.add_tracks(pt[found], uv[found], np.full((found.sum(), 2), np.nan))
    self.events.append({'type': 'reloc', 'frame': self.frame, 'lost': self.lost, 'found': int(found.sum()),
                        'n3d': int(self.mapped_tracks().sum())})
    return True

  def track_local_map(self):
    '''re-find untracked points of the local keyframes in the current frame (longer tracks), then refine the pose'''
    c, m = self.c, self.map
    local = self.local_keyframes(self.last_kf, c.ba_window)
    candidates = np.setdiff1d(np.unique(m.obs.pt[m.obs_of(kfs=local)]), self.tr_pt)
    pt, uv = self.search(self.T_cw, self.pyr, candidates, 0, 0, c.local_map_radius)
    self.add_tracks(pt, uv, np.full((len(pt), 2), np.nan))
    m3 = np.flatnonzero(self.mapped_tracks())
    T, e2 = optim.refine_pose(self.K, self.T_cw, m.pts.X[self.tr_pt[m3]], self.tr_uv[m3])
    keep = np.ones(len(self.tr_pt), bool)
    keep[m3[~(e2 <= optim.CHI2_2D)]] = False
    if keep[m3].sum() >= c.pnp_min_inliers:
      self.T_cw = T
      self.keep_tracks(keep)

  def predict(self, T_prev):
    '''pose prediction for the new frame (motion model, in the camera frame)'''
    c = self.c
    if c.motion_model == 'none':
      return T_prev
    if c.motion_model == 'accel':     # constant acceleration: the last change of velocity again
      return orthonormalize(self.acceleration @ self.velocity @ T_prev)
    return orthonormalize(self.velocity @ T_prev)

  def local_keyframes(self, kf, n):
    '''kf and its n most covisible keyframes'''
    covis = self.map.covisibility(kf)
    order = np.argsort(-covis)[:n]
    return np.r_[kf, order[covis[order] > 0]]

  def epipolar_filter(self):
    '''drop candidate tracks far from the epipolar line of their first observation'''
    m = self.map
    cand = np.flatnonzero(m.pts.state[self.tr_pt] == CANDIDATE)
    if len(cand) == 0:
      return
    ref = m.pts.ref[self.tr_pt[cand]]
    keep = np.ones(len(self.tr_pt), bool)
    for k in np.unique(ref):
      sel = cand[ref == k]
      T = self.T_cw @ inv_pose(m.kfs.T[k])
      t = T[:3, 3]
      if np.linalg.norm(t) < 1e-9:
        continue
      E = np.cross(np.eye(3), t / np.linalg.norm(t)) @ T[:3, :3]
      Kinv = np.linalg.inv(self.K)
      F = Kinv.T @ E @ Kinv
      lines = np.c_[m.pts.ref_uv[self.tr_pt[sel]], np.ones(len(sel))] @ F.T
      d = np.abs(np.sum(lines[:, :2] * self.tr_uv[sel], axis=1) + lines[:, 2]) / np.linalg.norm(lines[:, :2], axis=1)
      keep[sel[d > self.px_threshold(self.c.epipolar_px, self.c.adaptive_map)]] = False
    self.keep_tracks(keep)

  # ------------------------------------------------------------------------------------------------- search

  def choose_sources(self, pts, X, T_cw):
    '''per point the observing keyframe with the most similar viewpoint (-1 if none is similar enough)'''
    c, m = self.c, self.map
    o = m.obs_of(pts=pts)
    o = o[m.kfs.alive[m.obs.kf[o]]]
    row = np.full(len(m.pts), -1)
    row[pts] = np.arange(len(pts))
    r = row[m.obs.pt[o]]
    k = m.obs.kf[o]
    centers = camera_center(m.kfs.T)
    ray_cur = X[r] - camera_center(T_cw)
    ray_kf = X[r] - centers[k]
    d_cur, d_kf = np.linalg.norm(ray_cur, axis=1), np.linalg.norm(ray_kf, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.sum(ray_cur * ray_kf, axis=1) / (d_cur * d_kf), -1, 1)))
    ratio = d_kf / d_cur
    good = (angle <= c.search_max_angle) & (ratio <= c.search_max_scale) & (ratio >= 1 / c.search_max_scale)
    o, r, angle = o[good], r[good], angle[good]
    order = np.lexsort((angle, r))
    o, r = o[order], r[order]
    first = group_starts(r)
    src_kf = np.full(len(pts), -1)
    src_uv = np.zeros((len(pts), 2))
    src_kf[r[first]] = m.obs.kf[o[first]]
    src_uv[r[first]] = m.obs.uv[o[first]]
    return src_kf, src_uv

  def align(self, src_kf, src_uv, X, T_cw, pyr, guess, top_level, search_px):
    '''patch alignment of points from their source keyframes into an image with pose T_cw; returns uv, ncc'''
    c, m = self.c, self.map
    uv = guess.copy()
    ncc = np.full(len(X), -1.0)
    sel = np.flatnonzero(src_kf >= 0)
    if len(sel) == 0:
      return uv, ncc
    A, _, good = lk.affine_warps(self.K, m.kfs.T[src_kf[sel]], T_cw, X[sel], src_uv[sel])
    sel, A = sel[good], A[good]
    uv[sel], ncc[sel] = lk.align_patches(m.images.levels, m.kfs.slot[src_kf[sel]], pyr, src_uv[sel], A, guess[sel],
                                         top_level, search_px, c.patch_half, c.patch_iters)
    return uv, ncc

  def search(self, T_cw, pyr, pts, top_level=0, search_px=0, radius=None, count=False):
    '''find map points `pts` in an image (pyramid `pyr`) with pose T_cw; returns found point ids, pixels'''
    c, m = self.c, self.map
    radius = c.search_radius if radius is None else radius
    pts = pts[m.pts.state[pts] == MAPPED]
    X = m.pts.X[pts]
    uv_pred, z = project(self.K, T_cw, X)
    vis = (z > 0) & self.in_image(uv_pred, 8)
    pts, X, uv_pred = pts[vis], X[vis], uv_pred[vis]
    src_kf, src_uv = self.choose_sources(pts, X, T_cw)
    has = src_kf >= 0
    pts, X, uv_pred, src_kf, src_uv = pts[has], X[has], uv_pred[has], src_kf[has], src_uv[has]
    uv, ncc = self.align(src_kf, src_uv, X, T_cw, pyr, uv_pred, top_level, search_px)
    found = (ncc >= c.search_min_ncc) & (np.linalg.norm(uv - uv_pred, axis=1) <= radius)
    if count:
      m.pts.visible[pts] += 1
      m.pts.found[pts[found]] += 1
    return pts[found], uv[found]

  def remeasure(self, kf):
    '''re-measure tracked map points in the new keyframe with patches from their reference keyframe'''
    c, m = self.c, self.map
    T_cw = m.kfs.T[kf]
    tr = np.flatnonzero(self.mapped_tracks() & (m.pts.ref[self.tr_pt] != kf))
    if len(tr) == 0:
      return
    pts = self.tr_pt[tr]
    X = m.pts.X[pts]
    src_kf, src_uv = m.pts.ref[pts].copy(), m.pts.ref_uv[pts]
    # only if the reference view is similar enough
    ray_cur = X - camera_center(T_cw)
    ray_ref = X - camera_center(m.kfs.T[src_kf])
    cos = np.sum(ray_cur * ray_ref, axis=1) / np.linalg.norm(ray_cur, axis=1) / np.linalg.norm(ray_ref, axis=1)
    src_kf[cos < np.cos(np.deg2rad(c.search_max_angle))] = -1
    guess = self.tr_uv[tr].astype(np.float64)
    uv, ncc = self.align(src_kf, src_uv, X, T_cw, self.pyr, guess, 0, 0)
    ok = (ncc >= c.remeasure_min_ncc) & (np.linalg.norm(uv - guess, axis=1) <= c.remeasure_radius)
    self.counts['remeasure_tracks'] += len(tr)
    self.counts['remeasure_no_source'] += int((src_kf < 0).sum())
    self.counts['remeasure_failed'] += int(((src_kf >= 0) & (ncc < c.remeasure_min_ncc)).sum())
    self.counts['remeasure_moved_too_far'] += int(((ncc >= c.remeasure_min_ncc) & ~ok).sum())
    self.counts['remeasure_shift_px_sum'] += float(np.linalg.norm(uv - guess, axis=1)[ok].sum())
    self.tr_uv[tr[ok]] = uv[ok]
    self.tr_kf_uv[tr[ok]] = uv[ok]
    o = m.obs_of(kfs=[kf])
    row = np.full(len(m.pts), -1)
    row[m.obs.pt[o]] = o
    m.obs.uv[row[pts[ok]]] = uv[ok]
    if c.remeasure_drop_slipped:      # the patch is found, but far from the KLT position: the track slipped
      slipped = tr[(ncc >= c.remeasure_min_ncc) & ~ok]
      m.remove_obs(m.obs_of(kfs=[kf], pts=self.tr_pt[slipped]))
      keep = np.ones(len(self.tr_pt), bool)
      keep[slipped] = False
      self.keep_tracks(keep)

  def fuse(self, kf, pts, uv):
    '''add found points to the keyframe: merge with a track at the same place, else start a new track'''
    m = self.map
    near, j = np.zeros(len(pts), bool), np.zeros(len(pts), int)
    if len(self.tr_pt) and len(pts):
      d, j = scipy.spatial.cKDTree(self.tr_uv).query(uv)
      near = d <= self.c.fuse_px
    # at most one found point per track
    near_idx = np.flatnonzero(near)
    _, first = np.unique(j[near_idx], return_index=True)
    take = near_idx[first]
    jt, q, p, u = j[take], self.tr_pt[j[take]], pts[take], uv[take]
    # on a candidate track: the candidate is replaced by the found point
    cand = m.pts.state[q] == CANDIDATE
    m.kill(q[cand])
    m.add_obs(kf, p[cand], u[cand])
    self.tr_pt[jt[cand]] = p[cand]
    self.tr_uv[jt[cand]] = u[cand]
    # on a mapped track: the two are the same point, the one with more observations survives
    a, b = p[~cand], q[~cand]
    keep_a = m.pts.n_obs[a] > m.pts.n_obs[b]
    keep = np.where(keep_a, a, b)
    m.merge(keep, np.where(keep_a, b, a))
    self.tr_pt[jt[~cand]] = keep
    # everything else becomes a new track
    new = ~near
    m.add_obs(kf, pts[new], uv[new])
    self.add_tracks(pts[new], uv[new], uv[new])
    return int(new.sum() + len(take))

  # ------------------------------------------------------------------------------------------------- mapping

  def triangulate(self, kf):
    c, m = self.c, self.map
    cand = np.flatnonzero((m.pts.state == CANDIDATE) & (m.pts.n_obs >= 2))
    tracked = np.zeros(len(m.pts), bool)
    tracked[self.tr_pt] = True
    if len(cand):
      o = m.obs_of(pts=cand)
      row = np.full(len(m.pts), -1)
      row[cand] = np.arange(len(cand))
      ks = np.unique(m.obs.kf[o])
      kf_row = np.full(len(m.kfs), -1)
      kf_row[ks] = np.arange(len(ks))
      X, angle, err, depth = optim.triangulate(self.K, m.kfs.T[ks], kf_row[m.obs.kf[o]], row[m.obs.pt[o]],
                                               m.obs.uv[o], len(cand))
      ok = np.isfinite(X).all(axis=1) & (depth > 0) & (err <= self.px_threshold(c.tri_max_px, c.adaptive_map))
      enough = ok & (angle >= np.deg2rad(c.tri_min_angle))
      m.pts.X[cand[enough]] = X[enough]
      m.pts.state[cand[enough]] = MAPPED
      m.kill(cand[~ok])
    # candidates that are not tracked anymore can't collect more parallax
    lost = np.flatnonzero((m.pts.state == CANDIDATE) & ~tracked)
    m.kill(lost)
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)

  def local_ba(self, kf):
    '''the new keyframe and its most covisible ones; all other keyframes that see their points are held fixed'''
    c, m = self.c, self.map
    covis = m.covisibility(kf)
    order = np.argsort(-covis)
    window = np.r_[kf, order[:c.ba_window - 1][covis[order[:c.ba_window - 1]] >= c.ba_min_covisible]]
    pts = np.unique(m.obs.pt[m.obs_of(kfs=window)])
    pts = pts[m.pts.state[pts] == MAPPED]
    o = m.obs_of(pts=pts)
    kfs = np.unique(m.obs.kf[o])
    fixed = ~np.isin(kfs, window)
    self.bundle_adjust(kfs, fixed, pts, o, c.ba_iters)

  def global_ba(self, iters, delay=0):
    '''
    all keyframes and points. With delay > 0 it runs in a background thread on a snapshot of the map and its result is
    merged `delay` frames later (deterministic: never "when it is done"), see merge_global_ba
    '''
    m = self.map
    pts = np.flatnonzero(m.pts.state == MAPPED)
    o = m.obs_of(pts=pts)
    kfs = np.unique(m.obs.kf[o])
    if not delay:
      return self.bundle_adjust(kfs, np.zeros(len(kfs), bool), pts, o, iters)
    self.merge_global_ba()
    args = self.ba_problem(kfs, np.zeros(len(kfs), bool), pts, o, iters)
    result = {}
    thread = threading.Thread(target=lambda: result.update(out=optim.bundle_adjust(*args[0], **args[1])))
    thread.start()
    self.pending_ba = dict(thread=thread, result=result, kfs=kfs, pts=pts, T=m.kfs.T[kfs].copy(),
                           obs=(m.obs.kf[o].copy(), m.obs.pt[o].copy()), frame=self.frame + delay)

  def merge_global_ba(self, force=True):
    '''
    merge a background global BA (ORB-SLAM style): keyframes and points in the snapshot get the optimized values;
    keyframes created since move with their nearest ancestor in the snapshot, points with their reference keyframe
    '''
    p = self.pending_ba
    if p is None or (not force and self.frame < p['frame']):
      return
    with self.timed('global_ba_wait'):
      p['thread'].join()
    self.pending_ba = None
    m = self.map
    T_ba, X_ba, e2 = p['result']['out']
    T_now = m.kfs.T.copy()
    # correction of every keyframe: T_new = C @ T_now, for snapshot keyframes directly, else through its ancestors
    C = np.tile(np.eye(4), (len(m.kfs), 1, 1))
    in_ba = np.zeros(len(m.kfs), bool)
    in_ba[p['kfs']] = True
    C[p['kfs']] = T_ba @ inv_pose(T_now[p['kfs']])
    for k in range(p['kfs'].max() + 1, len(m.kfs)):     # newer keyframes: their parent was already corrected
      parent = m.kfs.parent[k]
      while parent >= 0 and not in_ba[parent] and parent > p['kfs'].max() and not m.kfs.alive[parent]:
        parent = m.kfs.parent[parent]
      if parent >= 0:
        C[k] = C[parent]
    alive = m.kfs.alive.copy()
    m.kfs.T[alive] = C[alive] @ T_now[alive]
    # points: optimized ones get their new position, the others move with their reference keyframe
    mapped = m.pts.state == MAPPED
    moved = np.flatnonzero(mapped)
    ref = m.pts.ref[moved]
    M = inv_pose(m.kfs.T[ref]) @ T_now[ref]
    m.pts.X[moved] = np.einsum('nij,nj->ni', M[:, :3, :3], m.pts.X[moved]) + M[:, :3, 3]
    still = p['pts'][mapped[p['pts']]]
    m.pts.X[still] = X_ba[mapped[p['pts']]]
    # outlier observations (looked up again: the observation table may have been compacted)
    bad = ~(e2 <= optim.CHI2_2D)
    key = m.obs.pt * len(m.kfs) + m.obs.kf
    bad_key = p['obs'][1][bad] * len(m.kfs) + p['obs'][0][bad]
    m.remove_obs(np.flatnonzero(np.isin(key, bad_key) & m.obs.alive))
    self.T_cw = C[self.last_kf] @ self.T_cw
    self.T_good = C[self.last_kf] @ self.T_good
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)

  def ba_problem(self, kfs, fixed, pts, o, iters):
    '''
    arguments of optim.bundle_adjust (copies). Monocular gauge (7 DoF): if fewer than two keyframes are held fixed,
    fix the oldest one and the scale, with the translation component of the next one along their baseline
    '''
    m = self.map
    scale_fix = None
    if fixed.sum() < 2:
      free = np.flatnonzero(~fixed)
      if fixed.sum() == 0:
        fixed[free[0]] = True
        free = free[1:]
      if len(free) == 0:
        return None
      a, b = np.flatnonzero(fixed)[0], free[0]
      baseline = m.kfs.T[kfs[b], :3, :3] @ (camera_center(m.kfs.T[kfs[b]]) - camera_center(m.kfs.T[kfs[a]]))
      scale_fix = (b, int(np.argmax(np.abs(baseline))))
    kf_row = np.full(len(m.kfs), -1)
    kf_row[kfs] = np.arange(len(kfs))
    pt_row = np.full(len(m.pts), -1)
    pt_row[pts] = np.arange(len(pts))
    return ((self.K, m.kfs.T[kfs].copy(), fixed, m.pts.X[pts].copy(), kf_row[m.obs.kf[o]], pt_row[m.obs.pt[o]],
             m.obs.uv[o].copy(), iters),
            dict(scale_fix=scale_fix, noise=m.kfs.noise[m.obs.kf[o]] if self.c.adaptive_map else None,
                 loss=self.c.ba_loss))

  def bundle_adjust(self, kfs, fixed, pts, o, iters):
    m = self.map
    args = self.ba_problem(kfs, fixed, pts, o, iters)
    if args is None:
      return
    with self.timed('ba'):
      T, X, e2 = optim.bundle_adjust(*args[0], **args[1])
    m.kfs.T[kfs] = T
    m.pts.X[pts] = X
    m.remove_obs(o[~(e2 <= optim.CHI2_2D)])
    self.T_cw = m.kfs.T[self.last_kf].copy()
    # tracks lose their point if it died or its observation in the current keyframe was removed
    o = m.obs_of(kfs=[self.last_kf])
    has = np.zeros(len(m.pts), bool)
    has[m.obs.pt[o]] = True
    self.keep_tracks(has[self.tr_pt] & (m.pts.state[self.tr_pt] != DEAD))

  def cull_points(self, kf):
    c, m = self.c, self.map
    p = np.flatnonzero(m.pts.state == MAPPED)
    bad = (m.pts.visible[p] >= 5) & (m.pts.found[p] < c.cull_min_ratio * m.pts.visible[p])
    bad |= (m.pts.first_kf[p] <= kf - 3) & (m.pts.n_obs[p] <= 2)
    m.kill(p[bad])
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)

  def cull_keyframes(self, kf, candidates):
    '''
    drop keyframes that are near-duplicates of an older one (revisits, hovering): about the same place (closer than
    the keyframe parallax threshold at the scene depth), orientation and points
    '''
    c, m = self.c, self.map
    candidates = [k for k in candidates if 1 < k < kf - 2 and m.kfs.alive[k]]
    if not candidates:
      return
    C = m.covisibility_matrix()
    n_points = C.diagonal()
    for k in candidates:
      if c.cull_old_redundant and k < kf - c.cull_old_age:
        # old keyframes thin out (ORB-SLAM's rule, but only outside the recent past): most of their points are
        # seen by >= 3 other keyframes
        p = m.points_of(k)
        if len(p) and np.mean(m.pts.n_obs[p] >= 4) >= c.cull_old_redundancy:
          self.remove_keyframe(k)
          continue
      j, shared = C.indices[C.indptr[k]:C.indptr[k + 1]], C.data[C.indptr[k]:C.indptr[k + 1]]
      older = j[(j < k) & (shared >= c.cull_kf_shared * n_points[k]) & m.kfs.alive[j]]
      if n_points[k] < c.kf_min_3d or len(older) == 0:
        continue
      depth = np.median(transform(m.kfs.T[k], m.pts.X[m.points_of(k)])[:, 2])
      dist = np.linalg.norm(camera_center(m.kfs.T[older]) - camera_center(m.kfs.T[k]), axis=1)
      angle = np.degrees(rotation_angle(m.kfs.T[older, :3, :3] @ m.kfs.T[k, :3, :3].T))
      if np.any((dist < np.tan(np.deg2rad(c.kf_parallax)) * depth) & (angle < c.cull_kf_angle)):
        self.remove_keyframe(k)

  def remove_keyframe(self, k):
    m = self.map
    parent = m.kfs.parent[k]
    while not m.kfs.alive[parent]:
      parent = m.kfs.parent[parent]
    m.kfs.alive[k] = False
    m.images.release(m.kfs.slot[k])
    m.kfs.parent[m.kfs.parent == k] = parent
    m.remove_obs(m.obs_of(kfs=[k]))     # (moves the reference of k's points)
    # frames referenced to k are re-expressed relative to the parent
    i = np.flatnonzero(self.traj.ref == k)
    self.traj.T[i] = self.traj.T[i] @ m.kfs.T[k] @ inv_pose(m.kfs.T[parent])
    self.traj.ref[i] = parent
    self.events.append({'type': 'cull_kf', 'frame': self.frame, 'kf': int(k)})

  def create_keyframe(self):
    c, m = self.c, self.map
    kf = self.add_keyframe(self.T_cw)
    if c.remeasure:
      with self.timed('remeasure'):
        self.remeasure(kf)
    with self.timed('search'):
      tracked = np.zeros(len(m.pts), bool)
      tracked[self.tr_pt] = True
      untracked = np.flatnonzero((m.pts.state == MAPPED) & ~tracked)
      m.pts.visible[self.tr_pt[self.mapped_tracks()]] += 1
      m.pts.found[self.tr_pt[self.mapped_tracks()]] += 1
      old = m.pts.last_kf[untracked] < kf - c.loop_age      # (searched with a wide window by the loop detection)
      pts, uv = self.search(m.kfs.T[kf], self.pyr, untracked[~old], count=True)
      self.fuse(kf, pts, uv)
    with self.timed('triangulate'):
      self.triangulate(kf)
    with self.timed('local_ba'):
      self.local_ba(kf)
    if c.loop and kf % c.loop_every == 0:
      self.diag['loop_search'] = True
      with self.timed('loop'):
        if old.any():
          self.detect_loop(kf, untracked[old])
        else:
          self.detect_segment_loop(kf)
    if c.cull_points:
      self.cull_points(kf)
    if c.cull_keyframes:
      self.cull_keyframes(kf, self.local_keyframes(kf, c.ba_window)[1:])
    m.compact_obs()
    with self.timed('detect'):
      self.detect(kf)
    self.kf_n3d = int(self.mapped_tracks().sum())

  # ------------------------------------------------------------------------------------------------- loops

  def detect_loop(self, kf, old):
    '''old points re-found with a wide window: if they agree on a different pose, correct the drift'''
    c, m = self.c, self.map
    T = m.kfs.T[kf]
    pts, uv = self.search(T, self.pyr, old, c.loop_top_level, c.loop_search_px, np.inf)
    T_loop, inl = self.pnp(uv, m.pts.X[pts]) if len(pts) >= c.loop_min_matches else (None, np.zeros(0, bool))
    if inl.sum() < c.loop_min_inliers:
      self.detect_segment_loop(kf)
      return
    pts, uv = pts[inl], uv[inl]
    proj, _ = project(self.K, T, m.pts.X[pts])
    disagreement = float(np.mean(np.linalg.norm(proj - uv, axis=1)))
    if disagreement < c.loop_min_correction:
      self.fuse(kf, pts, uv)     # small drift: the old points are just observations, local BA takes care of it
      if c.loop_fuse_ba:
        self.local_ba(kf)
      return
    self.close_loop(kf, T_loop, pts, uv, disagreement)

  def detect_segment_loop(self, kf):
    '''
    in a new segment the pose of the old map is unknown: try the poses of keyframes of other segments (round-robin)
    as hypotheses for where we are in the old map
    '''
    c, m = self.c, self.map
    others = np.flatnonzero(m.kfs.alive & (m.kfs.segment != self.segment))
    if len(others) == 0:
      return
    self.segment_cursor += c.reloc_keyframes
    for k in others[(self.segment_cursor + np.arange(c.reloc_keyframes)) % len(others)]:
      pts, uv = self.search(m.kfs.T[k], self.pyr, m.points_of(k), 3, 8, np.inf)
      if len(pts) < 2 * c.loop_min_matches:
        continue
      T_loop, inl = self.pnp(uv, m.pts.X[pts], m.kfs.T[k])
      if inl.sum() >= 2 * c.loop_min_inliers:
        self.close_loop(kf, T_loop, pts[inl], uv[inl], np.inf)
        return

  def close_loop(self, kf, T_loop, pts, uv, disagreement):
    c, m = self.c, self.map
    self.merge_global_ba()
    # 1. scale drift: depth of the old points (loop pose) vs nearby current points (current pose)
    cur = m.points_of(kf)
    o = m.obs_of(kfs=[kf], pts=cur)
    cur_uv, cur_depth = m.obs.uv[o], transform(m.kfs.T[kf], m.pts.X[m.obs.pt[o]])[:, 2]
    old_depth = transform(T_loop, m.pts.X[pts])[:, 2]
    d, i = scipy.spatial.cKDTree(cur_uv).query(uv)
    close = d < 10
    if close.sum() < 5:
      return
    s = float(np.median(old_depth[close] / cur_depth[i[close]]))
    # 2. Sim3 pose graph over the alive keyframes: spanning tree + covisibility + loop edges
    alive = np.flatnonzero(m.kfs.alive)
    row = np.full(len(m.kfs), -1)
    row[alive] = np.arange(len(alive))
    S = m.kfs.T[alive].copy()
    # essential graph: spanning tree + the strongest covisibility links of every keyframe
    C = m.covisibility_matrix()
    pairs = {}
    for k in alive:
      j, n = C.indices[C.indptr[k]:C.indptr[k + 1]], C.data[C.indptr[k]:C.indptr[k + 1]]
      sel = (j < k) & (n >= c.ba_min_covisible)
      for i in np.argsort(-n[sel])[:c.essential_edges]:
        pairs[(int(k), int(j[sel][i]))] = n[sel][i]
      if m.kfs.parent[k] >= 0:
        pairs.setdefault((int(k), int(m.kfs.parent[k])), c.ba_min_covisible)
    # (the edge to the first keyframe of a segment is a guess: it gets almost no weight)
    edges = [(row[k], row[j], S[row[k]] @ sim3_inv(S[row[j]]),
              max(n, c.ba_min_covisible) / 100 * (1 if m.kfs.segment[k] == m.kfs.segment[j] else 1e-4))
             for (k, j), n in pairs.items() if m.kfs.alive[k] and m.kfs.alive[j]]
    src = collections.Counter(m.pts.ref[pts].tolist()).most_common(1)[0][0]
    if not m.kfs.alive[src]:
      return
    # all other edges hold the current relative poses: only the new loop edge has an error
    S_kf = sim3(1 / s, T_loop[:3, :3], T_loop[:3, 3] / s)
    edges.append((row[kf], row[src], S_kf @ sim3_inv(S[row[src]]), 10 * max(len(pts), 15) / 100))
    fixed = np.zeros(len(alive), bool)
    fixed[row[src]] = True
    with self.timed('pose_graph'):
      S_new = optim.pose_graph(S, edges, fixed)
    # points move with their reference keyframe
    p = np.flatnonzero(m.pts.state == MAPPED)
    ref = row[m.pts.ref[p]]
    M = np.einsum('nij,njk->nik', np.array([sim3_inv(Sn) for Sn in S_new]), S)[ref]
    m.pts.X[p] = np.einsum('nij,nj->ni', M[:, :3, :3], m.pts.X[p]) + M[:, :3, 3]
    moved = np.linalg.norm(camera_center(sim3_to_se3(S_new[row[kf]])) - camera_center(m.kfs.T[kf]))
    m.kfs.T[alive] = np.array([sim3_to_se3(Sn) for Sn in S_new])
    self.T_cw = m.kfs.T[kf].copy()
    if m.kfs.segment[src] != m.kfs.segment[kf]:     # the new segment joins the old one
      m.kfs.segment[m.kfs.segment == m.kfs.segment[kf]] = m.kfs.segment[src]
      self.segment = int(m.kfs.segment[src])
    self.diag['loop_closed'] = True
    self.events.append({'type': 'loop', 'frame': self.frame, 'kf': int(kf), 'loop_kf': int(src),
                        'matches': int(len(pts)), 'scale': s, 'disagreement': disagreement, 'moved': float(moved)})
    # 3. re-find the old points with the corrected pose, fuse, global BA
    tracked = np.zeros(len(m.pts), bool)
    tracked[self.tr_pt] = True
    pts, uv = self.search(self.T_cw, self.pyr, np.flatnonzero((m.pts.state == MAPPED) & ~tracked))
    self.fuse(kf, pts, uv)
    if c.global_ba_iters:
      with self.timed('global_ba'):
        self.global_ba(c.global_ba_iters, c.global_ba_delay)
      if c.global_ba_delay:
        self.local_ba(kf)
    else:
      self.local_ba(kf)

  # ------------------------------------------------------------------------------------------------- main loop

  def on_frame(self, index, img):
    c = self.c
    t_start = time.perf_counter()
    self.frame = index
    if self.clahe is not None:
      img = self.clahe.apply(img)
    pyr = lk.pyramid(img)
    half = pyr[1].astype(np.float32)
    self.diag['blur'] = float(cv2.Laplacian(half, cv2.CV_32F).var() / max(half.var(), 1.0))
    if self.img is None:
      self.img, self.pyr = img, pyr
      self.reset()
      if c.rotation_only_poses:
        self.traj.add(1, frame=index, ref=self.last_kf, T=np.eye(4), online=np.eye(4))
    else:
      self.merge_global_ba(force=False)
      T_prev = self.T_cw
      T_pred = self.predict(T_prev)
      with self.timed('klt'):
        self.track(img, T_pred, pyr)
      self.img, self.pyr = img, pyr
      if self.initializing:
        with self.timed('init'):
          if self.initialized and self.relocalize(self.T_good):    # back on the map: no new segment needed
            self.abort_segment()
            self.lost, self.T_good = 0, self.T_cw.copy()
          else:
            self.lost += self.initialized
            self.try_initialize()
            if self.initializing and not self.initialized:
              # no translation yet: the best pose is the rotation-only one (position held)
              if (T := self.rotation_only_pose()) is not None:
                self.T_cw = T
              if c.rotation_only_poses:
                self.traj.add(1, frame=index, ref=self.last_kf, T=self.T_cw @ inv_pose(self.map.kfs.T[self.last_kf]),
                              online=inv_pose(self.T_cw))
      else:
        with self.timed('pose'):
          ok = self.estimate_pose(T_pred)
        if not ok or self.mapped_tracks().sum() < c.reloc_min_3d:
          with self.timed('relocalize'):
            ok = self.relocalize(self.T_cw if ok else T_pred) or ok
        if ok and c.track_local_map:
          with self.timed('local_map'):
            self.track_local_map()
        if ok:
          velocity = orthonormalize(self.T_cw @ inv_pose(T_prev)) if not self.lost else np.eye(4)
          self.acceleration = velocity @ inv_pose(self.velocity)
          self.velocity = velocity
          self.lost = 0
          self.T_good = self.T_cw.copy()
          m3 = self.mapped_tracks()
          self.good_depth = float(np.median(transform(self.T_cw, self.map.pts.X[self.tr_pt[m3]])[:, 2]))
        else:
          self.lost += 1
          self.T_cw = T_pred if self.lost <= 3 else self.T_good
          if self.lost > 3:
            self.velocity = np.eye(4)
          if c.reinit and self.lost > c.reinit_after:
            self.start_segment()
        self.epipolar_filter()
        if not self.initializing and self.need_keyframe():
          with self.timed('keyframe'):
            self.create_keyframe()
          self.diag['keyframe'] = True

    m = self.map
    if self.initialized and not self.lost and not self.initializing:
      self.traj.add(1, frame=index, ref=self.last_kf, T=self.T_cw @ inv_pose(m.kfs.T[self.last_kf]),
                    online=inv_pose(self.T_cw))
    self.stats.add(1, frame=index, tracks=len(self.tr_pt), mapped=int(self.mapped_tracks().sum()),
                   far=int((m.pts.state[self.tr_pt] == CANDIDATE).sum()), lost=self.lost, keyframes=int(m.kfs.alive.sum()),
                   points=int((m.pts.state == MAPPED).sum()), ms=1e3 * (time.perf_counter() - t_start), **self.diag)
    self.diag = {}

  def finish(self):
    self.merge_global_ba()
    if self.initialized and self.c.final_ba_iters:
      with self.timed('final_ba'):
        self.global_ba(self.c.final_ba_iters)

  def poses(self, n_frames, online=True):
    '''
    T_wc of all frames (NaN = not tracked): as estimated live (online) or from the final keyframe poses (after later
    local BAs, loop closures and the final BA, i.e. with hindsight)
    '''
    T = np.full((n_frames, 4, 4), np.nan)
    T[self.traj.frame] = self.traj.online if online else inv_pose(self.traj.T @ self.map.kfs.T[self.traj.ref])
    return T

  def trajectory(self, n_frames):
    '''camera centres of all frames, final and online'''
    return self.poses(n_frames, online=False)[:, :3, 3], self.poses(n_frames)[:, :3, 3]
