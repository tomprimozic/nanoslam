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
                      sim3_inv, sim3_scale, sim3_to_se3, rotation_angle)


class Config:
  focal_scale = 1.0                 # diagnostics: simulated calibration error
  clahe = 3.0                       # CLAHE clip limit (0 = off)

  # frame-to-frame KLT
  klt_win = 9
  klt_levels = 3
  klt_iters = 30
  klt_fb = 0.5                      # px, forward-backward check
  phase_prior = True                # global image shift by phase correlation corrects the KLT initial guesses
  phase_min_response = 0.05         # phase correlation peak needed to trust the shift
  phase_min_correction = 2.0        # px, only correct the prediction by larger differences

  # detection: the strongest min-eigenvalue corner of every empty grid cell
  cells = 20                        # across the image width
  min_quality = 0.001               # relative to the strongest corner
  detect_blur = True                # 3x3 blur before the corner score (OV2SLAM)

  # initialization
  # (batch BA over the frames since the anchor keyframe, see try_initialize)
  init_window = 20                  # frames in the window BA (subsampled)
  init_check_parallax = 0.0         # deg, median parallax before a window test is attempted (0: every frame)
  init_test = 'marginal'            # translation test: 'marginal' (slamfold: depths integrated out, the null is
                                    # 1.5 x 3(W-1)/2 nats) or 'map' (MAP depths, null = the extra parameters)
  init_ratio = 2.0                  # translation established: likelihood ratio full vs rotation-only > this x the null
  init_seed = 'angle'               # map seeding: 'angle' (>= tri_min_angle of parallax, init_min_points and
                                    # init_min_ratio) or 'resolved' depths (posterior std < 20% of the mean, marginal
                                    # test only: maps from 1-5 mm baselines on real data, early error 2x)
  rotation_only_poses = True        # before translation is established: emit rotation-only poses (position held)
  window_poses = True               # translation established but too little parallax for the map yet: emit the window BA's pose
  init_min_points = 50
  init_min_ratio = 0.5              # fraction of inliers with parallax (rejects rotation-dominated motion)

  # tracking
  pnp_px = 3.0
  pnp_min_inliers = 15
  epipolar_px = 3.0
  reloc_min_3d = 50                 # relocalize (search the map with a wide window) below this many 3D tracks
  reloc_keyframes = 8               # keyframe poses tried per frame while lost
  reloc_max_hypotheses = 0          # cap on the pose hypotheses relocalization tries per frame (0: no cap; ~3 x faster
                                    # lost / mapless frames at 4, at the cost of slower recovery)
  mapless = True                    # lost: track on like before the first initialization (rotation-only / window BA poses)
                                    # from the last keyframe, until a new part of the map is made or relocalization works
  resume_scale_points = 10          # map points next to the new ones that make the scale of a new part observable; with
                                    # fewer it is provisional (scale guessed) until the old map is found again

  # triangulation
  tri_min_angle = 1.0               # deg
  tri_max_px = 3.0
  tri_ray_gate = True               # a candidate whose solution fails is only dropped when its rays have the parallax
                                    # (else: a noisy far point, it keeps collecting observations)

  # keyframes
  kf_parallax = 1.75                # deg, rotation-compensated median
  kf_tracked_ratio = 0.7
  kf_min_3d = 40

  # local BA
  ba_window = 10
  ba_min_covisible = 15
  ba_iters = 10

  # patch search (search by projection, re-measurement)
  patch_half = 4                    # 9x9
  patch_iters = 20
  search_max_angle = 35.0           # deg, max viewing angle difference to the source keyframe
  search_max_scale = 1.8            # max viewing distance ratio
  search_radius = 4.0               # px
  search_min_ncc = 0.75
  remeasure = True                  # re-measure tracked points from their reference keyframe
  remeasure_radius = 2.0
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
    self.velocity = np.eye(4)
    self.counts = collections.Counter()   # diagnostics
    self.last_kf = -1
    self.initializing = True          # no map tracking: before the first initialization, or lost (see start_mapless)
    self.good_depth = 1.0             # median depth of the tracked points at the last good frame
    self.window = []                  # while initializing: (frame, point ids, pixels) of every frame since the anchor
    self.pre_kfs = []                 # keyframes made before translation was established (rotation-only poses)
    self.anchor = -1                  # the one the translation test is relative to
    self.window_pose = None           # T_cw of the current frame from the window BA (translation established)
    self.provisional = None           # a map part of unknown scale since a junction keyframe: (junction, first keyframe,
                                      # ids of the points mapped before)
    self.init_n_tracks = 0            # tracks right after the last of them
    self.lost = 0                     # frames since tracking was lost
    self.kf_n3d = 0                   # tracked map points right after the last keyframe
    self.cursor = 0                   # round-robin position of the search for the old map (provisional part)
    self.reloc_cursor = 0             # round-robin position of the keyframe hypotheses of relocalization
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
    shift = self.shift = self.global_shift(pyr) if c.phase_prior else None
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
    pts = self.corners(self.img, self.tr_uv)
    if len(pts):
      ids = self.map.add_points(kf, pts.astype(np.float64))
      self.add_tracks(ids, pts, pts.astype(np.float64))

  def corners(self, img, tracked):
    '''the strongest min-eigenvalue corner of every grid cell without a track'''
    c = self.c
    h, w = img.shape
    cell = int(np.ceil(w / c.cells))
    ny, nx = -(-h // cell), -(-w // cell)
    occupied = np.zeros((ny, nx), bool)
    uv = tracked.astype(int)
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
      return np.zeros((0, 2), np.float32)
    by, bx = np.divmod(best[cy, cx], cell)
    pts = np.stack([cx * cell + bx, cy * cell + by], axis=1).astype(np.float32)
    return cv2.cornerSubPix(img, pts.reshape(-1, 1, 2), (3, 3), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)).reshape(-1, 2)

  # ------------------------------------------------------------------------------------------------- keyframes

  def add_keyframe(self, T_cw):
    m = self.map
    kf = m.kfs.add(1, T=T_cw, frame=self.frame, parent=self.last_kf)[0]
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


  def translation_test(self, T, R_rot, X, cam, pt, uv, host_uv, res_rot):
    '''
    slamfold's test: likelihood ratio of the full model (window BA rotations and translation directions, every
    landmark's inverse depth integrated out under a flat prior, its host bearing integrated by Laplace; the
    translation magnitude - the prior's support in the BA's gauge - at its optimum) vs the nested rotation-only model,
    in units of the noise level (from the rotation-only residuals); the null is 1.5 x half a nat per extra
    pose parameter. Returns the ratio, the null and which landmarks have a resolved depth (posterior std < 20% of the
    mean)
    '''
    n, W = len(X), len(T)
    # noise level from the rotation-only residuals (host and frame noise: variance 2 sigma^2; median |r| = 1.1774 x
    # the per-axis sigma in 2D): unbiased without translation, larger with it (the test is then conservative)
    sigma = max(np.median(res_rot) / 1.1774 / np.sqrt(2), 0.02)
    b = unproject(self.K, host_uv)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    B = np.zeros((n, 3, 2))
    a = np.where(np.abs(b[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    B[:, :, 0] = np.cross(b, a)
    B[:, :, 0] /= np.linalg.norm(B[:, :, 0], axis=1, keepdims=True)
    B[:, :, 1] = np.cross(b, B[:, :, 0])
    order = np.argsort(pt, kind='stable')
    start = np.searchsorted(pt[order], np.arange(n + 1)).astype(np.int64)
    cam_s, uv_s = cam[order].astype(np.int64), np.ascontiguousarray(uv[order], dtype=np.float64)
    rho_map = np.where(X[:, 2] > 0, 1 / np.linalg.norm(X, axis=1), 0.0)
    rho_ref = max(float(np.median(rho_map[rho_map > 0])) if (rho_map > 0).any() else 1.0, 1e-9)
    log_int, mean, std = optim.depth_integrated(T[:, :3, :3], T[:, :3, 3], b, B, cam_s, uv_s, start, self.K, sigma,
                                                rho_ref)
    f, c = np.array([self.K[0, 0], self.K[1, 1]]), np.array([self.K[0, 2], self.K[1, 2]])
    E_rot = optim.landmark_costs(np.ascontiguousarray(R_rot), np.zeros((W, 3)), b, B, cam_s, uv_s, start, f, c, sigma,
                                 optim.HUBER, np.zeros((n, 1)))[:, 0]
    dF = float(np.sum(E_rot + log_int))
    return dF, 1.5 * 3 * (W - 1) / 2, (std < 0.2 * mean) & (mean > 0)

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

  def try_initialize(self):
    '''
    batch BA over the frames since the anchor keyframe (subsampled), started from the rotation-only rotations and the
    two-view translation direction; initialized once the rotation-only model clearly explains the tracks worse than the
    full one (residual scale ratio, a likelihood ratio without the noise level) and enough points have parallax.
    (An approximation of slamfold's test: the full model's depths are MAP estimates in a robust cost, not integrated out)
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
    X, _, _, depth, _ = optim.triangulate(self.K, np.array([np.eye(4), T_last]), np.r_[np.zeros(n, int), np.ones(n, int)],
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
    R_rot = np.zeros((len(frames), 3, 3))
    for j in range(len(frames)):
      o = cam == j
      R, res = self.rotation_only(ref_of[o], uv[o])
      T[j, :3, :3] = R_rot[j] = R
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
    W = len(frames)
    if c.init_test == 'map':
      sigma = max(1.4826 * np.median(np.sqrt(e2[np.isfinite(e2)])), 0.1)
      rr = np.concatenate(res_rot) / sigma
      rf = np.sqrt(np.minimum(e2, 1e6)) / sigma
      huber = lambda r: np.where(r <= optim.HUBER, 0.5 * r ** 2, optim.HUBER * (r - 0.5 * optim.HUBER))
      dF = float(huber(rr).sum() - huber(rf).sum())
      null = 3 * (W - 1) + n
    else:
      dF, null, resolved = self.translation_test(T, R_rot, X, cam, pt, uv, ref[inl], np.concatenate(res_rot)[cam > 0])
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
    if c.init_test == 'map' or c.init_seed == 'angle':
      good_angle = good & (angle >= np.deg2rad(c.tri_min_angle))
      enough = good_angle.sum() >= max(c.init_min_points, c.init_min_ratio * good.sum())
    else:     # the map is seeded with the landmarks whose depth is resolved
      good_angle = good & resolved
      enough = good_angle.sum() >= c.init_min_points
    self.diag['init_ratio'] = dF / null
    # scale: the depths of the anchor's map points next to the new points in its image (the same point, or a
    # neighbour within 10 px), else the depth seen before. A heuristic (depth is assumed locally smooth; the number of
    # pairs does not make the scale reliable, only less often far off)
    s, n_scale = self.good_depth / np.median(X[good, 2]), 0
    o = m.obs_of(kfs=[kf0])
    o = o[m.pts.state[m.obs.pt[o]] == MAPPED]
    if len(o) >= 5:
      d, i = scipy.spatial.cKDTree(m.obs.uv[o]).query(ref[inl][good])
      close = d < 10
      if close.sum() >= 5:
        depth_old = transform(m.kfs.T[kf0], m.pts.X[m.obs.pt[o[i[close]]]])[:, 2]
        s, n_scale = float(np.median(depth_old / X[good][close, 2])), int(close.sum())
    T[:, :3, 3] *= s
    X = X * s
    T0 = m.kfs.T[kf0]
    if dF >= c.init_ratio * null and c.window_poses:
      self.window_pose = T[-1] @ T0     # translation is established (the map still waits for parallax)
    if dF < c.init_ratio * null or not enough:
      return
    self.T_cw = self.T_good = T[-1] @ T0
    self.velocity = np.eye(4)
    keep = np.zeros(len(m.pts), bool)
    keep[pts[good]] = True
    self.keep_tracks(keep[self.tr_pt])
    if not self.initialized:
      m.kill(np.setdiff1d(m.points_of(kf0, CANDIDATE), self.tr_pt))
    kf1 = self.add_keyframe(self.T_cw)
    new = good_angle & (m.pts.state[pts] == CANDIDATE)     # (map points keep their positions)
    good_pts = pts[new]
    new_mask = m.pts.state != MAPPED
    new_mask[good_pts] = True
    m.pts.X[good_pts] = transform(inv_pose(T0), X[new])
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
    self.drop_pre_keyframes(kf0, kf1)
    resumed = self.initialized
    if resumed and n_scale < c.resume_scale_points and self.provisional is None:
      self.provisional = (kf0, kf1, np.flatnonzero(~new_mask & (m.pts.state == MAPPED)))
    self.initialized, self.initializing, self.lost = True, False, 0
    self.local_ba(kf1)
    self.detect(kf1)
    self.kf_n3d = int(self.mapped_tracks().sum())
    self.window = [(self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64))]
    self.events.append({'type': 'init' if not resumed else 'resume', 'frame': self.frame, 'points': int(len(good_pts)), 'frames': len(frames), 'ratio': dF / null, 'scale_from': n_scale})

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

  def start_mapless(self):
    '''
    tracking is lost: from here on the same as before the first initialization, with the last keyframe (known pose) as
    the anchor - rotation-only poses, keyframes with new corners, the translation test on the window since the anchor;
    a successful one continues the map (scale from the map points among the tracks), relocalization ends it
    '''
    self.initializing = True
    self.anchor, self.pre_kfs, self.init_n_tracks = self.last_kf, [], len(self.tr_pt)
    self.window = [w for w in self.window if w[0] >= self.map.kfs.frame[self.anchor]]
    if (T := self.rotation_only_pose()) is not None:
      self.T_cw = T
    self.events.append({'type': 'mapless', 'frame': self.frame})

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
        i = np.flatnonzero(self.traj.ref == k)
        self.traj.T[i] = self.traj.T[i] @ m.kfs.T[k] @ inv_pose(m.kfs.T[kf0])
        self.traj.ref[i] = kf0
    if kf0 in self.pre_kfs:
      m.kfs.parent[kf0] = m.kfs.parent[self.pre_kfs[0]]
    m.kfs.parent[kf1] = kf0
    self.pre_kfs = []

  def rotation_only_pose(self):
    '''the current pose if the camera only rotated since the last keyframe (None: too few tracks)'''
    has = np.isfinite(self.tr_kf_uv).all(axis=1)     # (tracks found again by relocalization have no pixel there)
    if has.sum() < 8:
      return None
    R, _ = self.rotation_only(self.tr_kf_uv[has], self.tr_uv[has].astype(np.float64))
    return make_pose(R, np.zeros(3)) @ self.map.kfs.T[self.last_kf]

  def abort_mapless(self):
    '''relocalized on the map while lost: the keyframes made since leave it'''
    m = self.map
    for k in self.pre_kfs[::-1]:
      self.last_kf = int(m.kfs.parent[k])
      self.remove_keyframe(k)
    self.pre_kfs = []
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)
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
                                                {'max_reproj_error': threshold or c.pnp_px,
                                                 'seed': c.ransac_seed, 'min_iterations': 50},
                                                {'loss_type': 'HUBER', 'loss_scale': 1.0}, initial)
    return make_pose(pose.R, pose.t), np.array(info['inliers'], bool)

  def estimate_pose(self, T_pred):
    c = self.c
    m3 = np.flatnonzero(self.mapped_tracks())
    if len(m3) < c.pnp_min_inliers:
      return False
    X, uv = self.map.pts.X[self.tr_pt[m3]], self.tr_uv[m3]
    T, inliers = self.pnp(uv, X, T_pred)
    if inliers.sum() < c.pnp_min_inliers:
      return False
    # robust scale of the reprojection residuals of all tracked map points (MAD per axis)
    proj, _ = project(self.K, T, X)
    sigma = float(1.4826 * np.median(np.abs(proj - uv)))
    self.diag['sigma'] = sigma
    self.T_cw = T
    self.diag['inliers'] = int(inliers.sum())
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
      cap = c.reloc_max_hypotheses or len(hypotheses) + c.reloc_keyframes
      hypotheses = hypotheses[:cap]
      n_kfs = min(c.reloc_keyframes, cap - len(hypotheses))
      for k in near[(self.reloc_cursor + np.arange(n_kfs)) % len(near)]:     # round-robin, nearest first
        hypotheses.append((m.kfs.T[k], [k]))
      self.reloc_cursor += n_kfs
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

  def predict(self, T_prev):
    '''pose prediction for the new frame (constant velocity, in the camera frame)'''
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
      keep[sel[d > self.c.epipolar_px]] = False
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

  def fuse(self, kf, pts, uv):
    '''add found points to the keyframe: merge with a track at the same place, else start a new track'''
    m = self.map
    near, j = np.zeros(len(pts), bool), np.zeros(len(pts), int)
    if len(self.tr_pt) and len(pts):
      d, j = scipy.spatial.cKDTree(self.tr_uv).query(uv)
      near = d <= self.c.fuse_px
    # at most one found point per track
    near &= pts != self.tr_pt[j] if len(self.tr_pt) else near     # (already the track's point: nothing to fuse)
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
    # everything else becomes a new track (unless it is tracked already)
    tracked = np.zeros(len(m.pts), bool)
    tracked[self.tr_pt] = True
    new = ~near & ~tracked[pts]
    m.add_obs(kf, pts[new], uv[new])
    self.add_tracks(pts[new], uv[new], uv[new])

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
      X, angle, err, depth, rays = optim.triangulate(self.K, m.kfs.T[ks], kf_row[m.obs.kf[o]], row[m.obs.pt[o]],
                                                     m.obs.uv[o], len(cand))
      ok = np.isfinite(X).all(axis=1) & (depth > 0) & (err <= c.tri_max_px)
      enough = ok & (angle >= np.deg2rad(c.tri_min_angle))
      m.pts.X[cand[enough]] = X[enough]
      m.pts.state[cand[enough]] = MAPPED
      # a failed solution: inconsistent observations if the rays have the parallax, else just noise on a far point
      m.kill(cand[~ok & (rays >= np.deg2rad(c.tri_min_angle))] if c.tri_ray_gate else cand[~ok])
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
    keyframes created since move with their nearest ancestor in the snapshot, points with their reference keyframe.
    Returns whether something was merged
    '''
    p = self.pending_ba
    if p is None or (not force and self.frame < p['frame']):
      return False
    with self.timed('global_ba_wait'):
      p['thread'].join()
    self.pending_ba = None
    m = self.map
    T_ba, X_ba, e2 = p['result']['out']
    T_now = m.kfs.T.copy()
    # correction of every keyframe in the world frame, T_new = T_now @ D (keeps the pose relative to the keyframe it
    # follows): for snapshot keyframes directly, else through its ancestors
    D = np.tile(np.eye(4), (len(m.kfs), 1, 1))
    in_ba = np.zeros(len(m.kfs), bool)
    in_ba[p['kfs']] = True
    D[p['kfs']] = inv_pose(T_now[p['kfs']]) @ T_ba
    for k in range(p['kfs'].max() + 1, len(m.kfs)):     # newer keyframes: their parent was already corrected
      parent = m.kfs.parent[k]
      while parent >= 0 and not in_ba[parent] and parent > p['kfs'].max() and not m.kfs.alive[parent]:
        parent = m.kfs.parent[parent]
      if parent >= 0:
        D[k] = D[parent]
    alive = m.kfs.alive.copy()
    m.kfs.T[alive] = T_now[alive] @ D[alive]
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
    self.T_cw = self.T_cw @ D[self.last_kf]
    self.T_good = self.T_good @ D[self.last_kf]
    self.keep_tracks(m.pts.state[self.tr_pt] != DEAD)
    return True

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
            dict(scale_fix=scale_fix))

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
    if m.kfs.frame[self.last_kf] == self.frame:     # (the current frame is that keyframe: its pose is the live one)
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
    keep = self.provisional[:2] if self.provisional is not None else ()     # (merge_provisional relies on them)
    candidates = [k for k in candidates if 1 < k < kf - 2 and m.kfs.alive[k] and k not in keep]
    if not candidates:
      return
    C = m.covisibility_matrix()
    n_points = C.diagonal()
    for k in candidates:
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
    if self.provisional is not None:
      with self.timed('loop'):
        self.refind_old_map(kf)
    if c.cull_points:
      self.cull_points(kf)
    if c.cull_keyframes:
      self.cull_keyframes(kf, self.local_keyframes(kf, c.ba_window)[1:])
    m.compact_obs()
    with self.timed('detect'):
      self.detect(kf)
    self.kf_n3d = int(self.mapped_tracks().sum())
    self.window = [(self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64))]

  # ------------------------------------------------------------------------------------------------- loops

  def detect_loop(self, kf, old):
    '''old points re-found with a wide window: if they agree on a different pose, correct the drift'''
    c, m = self.c, self.map
    T = m.kfs.T[kf]
    pts, uv = self.search(T, self.pyr, old, c.loop_top_level, c.loop_search_px, np.inf)
    T_loop, inl = self.pnp(uv, m.pts.X[pts]) if len(pts) >= c.loop_min_matches else (None, np.zeros(0, bool))
    if inl.sum() < c.loop_min_inliers:
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

  def refind_old_map(self, kf):
    '''
    a provisional map part (scale unknown): look for the map before it with the poses of its keyframes as hypotheses
    (round-robin, like relocalization); found, the part gets the old map's scale and pose - by the pose graph of a loop
    closure (or a Sim3 of the whole part without loop closing)
    '''
    c, m = self.c, self.map
    junction, first, old_pts = self.provisional
    old_kfs = np.flatnonzero(m.kfs.alive[:first])
    self.cursor += c.reloc_keyframes
    for k in old_kfs[(self.cursor + np.arange(c.reloc_keyframes)) % len(old_kfs)]:
      cand = np.setdiff1d(np.intersect1d(m.points_of(k), old_pts), self.tr_pt)     # (untracked old points)
      pts, uv = self.search(m.kfs.T[k], self.pyr, cand, 3, 8, np.inf)
      if len(pts) < 2 * c.loop_min_matches:
        continue
      T_old, inl = self.pnp(uv, m.pts.X[pts], m.kfs.T[k])
      if inl.sum() < 2 * c.loop_min_inliers:
        continue
      pts, uv = pts[inl], uv[inl]
      if c.loop:
        self.close_loop(kf, T_old, pts, uv, np.inf)
      else:
        self.merge_provisional(kf, T_old, pts, uv)
      return

  def loop_scale(self, kf, T_loop, pts, uv):
    '''scale drift at a loop: depth of the old points (loop pose) vs nearby current points (current pose), or None'''
    m = self.map
    cur = m.points_of(kf)
    o = m.obs_of(kfs=[kf], pts=cur)
    if len(o) == 0:
      return None
    cur_uv, cur_depth = m.obs.uv[o], transform(m.kfs.T[kf], m.pts.X[m.obs.pt[o]])[:, 2]
    old_depth = transform(T_loop, m.pts.X[pts])[:, 2]
    d, i = scipy.spatial.cKDTree(cur_uv).query(uv)
    close = d < 10
    return float(np.median(old_depth[close] / cur_depth[i[close]])) if close.sum() >= 5 else None

  def merge_provisional(self, kf, T_old, pts, uv):
    '''the provisional part (keyframes from its first one, the points mapped since) by the Sim3 that puts kf at T_old'''
    m = self.map
    junction, first, old_pts = self.provisional
    s = self.loop_scale(kf, T_old, pts, uv)
    if s is None:
      return
    R_S = T_old[:3, :3].T @ m.kfs.T[kf][:3, :3]
    t_S = camera_center(T_old) - s * R_S @ camera_center(m.kfs.T[kf])
    def move(T):      # T_cw in the provisional frame -> in the old one
      T = T.copy()
      T[..., :3, 3] = s * T[..., :3, 3] - (T[..., :3, :3] @ R_S.T) @ t_S
      T[..., :3, :3] = T[..., :3, :3] @ R_S.T
      return T
    part = np.flatnonzero(m.kfs.alive)
    part = part[part >= first]
    m.kfs.T[part] = move(m.kfs.T[part])
    p = np.setdiff1d(np.flatnonzero(m.pts.state == MAPPED), old_pts)
    m.pts.X[p] = s * m.pts.X[p] @ R_S.T + t_S
    i = np.flatnonzero(self.traj.ref >= first)
    self.traj.T[i, :3, 3] *= s
    self.T_cw = self.T_good = move(self.T_cw)
    self.velocity = np.eye(4)
    self.good_depth *= s
    self.provisional = None
    self.events.append({'type': 'merge', 'frame': self.frame, 'kf': int(kf), 'scale': s, 'matches': int(len(pts))})
    self.fuse(kf, pts, uv)
    self.local_ba(kf)

  def close_loop(self, kf, T_loop, pts, uv, disagreement):
    c, m = self.c, self.map
    if self.merge_global_ba():      # (the points moved: T_loop is stale)
      T_loop, inl = self.pnp(uv, m.pts.X[pts], T_loop)
      if inl.sum() < c.loop_min_inliers:
        return
      pts, uv = pts[inl], uv[inl]
    # 1. scale drift
    s = self.loop_scale(kf, T_loop, pts, uv)
    if s is None:
      return
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
    src = collections.Counter(m.pts.ref[pts].tolist()).most_common(1)[0][0]
    if not m.kfs.alive[src]:
      return
    if self.provisional is not None and src < self.provisional[1] <= kf:
      # the provisional part found the old map: its links to the old map (the junction edge, covisibility through the
      # junction's points) hold the guessed scale - drop them, the loop edge alone places the part (as a whole)
      first = self.provisional[1]
      pairs = {(k, j): n for (k, j), n in pairs.items() if (k >= first) == (j >= first)}
    edges = [(row[k], row[j], S[row[k]] @ sim3_inv(S[row[j]]), max(n, c.ba_min_covisible) / 100)
             for (k, j), n in pairs.items() if m.kfs.alive[k] and m.kfs.alive[j]]
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
    self.T_good = m.kfs.T[kf].copy()
    i = np.flatnonzero(row[self.traj.ref] >= 0)
    self.traj.T[i, :3, 3] /= np.array([sim3_scale(Sn) for Sn in S_new])[row[self.traj.ref[i]], None]
    self.T_cw = m.kfs.T[kf].copy()
    self.diag['loop_closed'] = True
    if self.provisional is not None and src < self.provisional[1]:   # (the provisional part is in the old map now)
      self.provisional = None
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
      self.window.append((self.frame, self.tr_pt.copy(), self.tr_uv.astype(np.float64)))
      if self.initializing:
        with self.timed('init'):
          if self.initialized and self.relocalize(self.T_cw):    # back on the map
            self.abort_mapless()
            self.lost, self.reloc_cursor, self.T_good = 0, 0, self.T_cw.copy()
          else:
            self.lost += self.initialized
            self.window_pose = None
            self.try_initialize()
            if self.initializing:
              # not initialized yet (or lost): the window BA's pose once translation is established, before that the
              # rotation-only one (position held)
              if self.window_pose is not None:
                self.T_cw = self.window_pose
              elif (T := self.rotation_only_pose()) is not None:
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
        if ok:
          velocity = orthonormalize(self.T_cw @ inv_pose(T_prev)) if not self.lost else np.eye(4)
          self.velocity = velocity
          self.lost = 0
          self.reloc_cursor = 0
          self.T_good = self.T_cw.copy()
          m3 = self.mapped_tracks()
          self.good_depth = float(np.median(transform(self.T_cw, self.map.pts.X[self.tr_pt[m3]])[:, 2]))
        else:
          self.lost += 1
          self.T_cw = T_pred if self.lost <= 3 else self.T_good
          if self.lost > 3:
            self.velocity = np.eye(4)
          if c.mapless:
            self.start_mapless()
        if ok:      # (not with a predicted / rotation-only pose: it would drop good candidates)
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
