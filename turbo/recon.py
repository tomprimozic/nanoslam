'''
3D reconstruction from a finished turbo map (map.npz written by run.py): per-keyframe inverse-depth maps from the
keyframe images and poses, evaluated against ground-truth depth.

  ../.venv/bin/python recon.py RUN_DIR --engine sparse          # RUN_DIR: a run.py --out directory (NAME or NAME@START)

Evaluation (per evaluated keyframe, over the pixels with ground truth): the estimate's scale comes from the keyframe's
sparse map points against the ground truth (so SLAM scale drift is not counted as reconstruction error, and the dense
estimate never fits its own scale). Metrics as in FLaME / Multi-Level Mapping: completeness (fraction of pixels with an
estimate), acc10 (fraction within 10% in inverse depth), RE (mean relative inverse-depth error of the estimates).
'''
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange

import datasets
from geometry import transform


class Recon:
  '''a finished map: keyframes (frame, T_cw), points, observations; keyframe images loaded on demand'''

  def __init__(self, run_dir):
    run_dir = Path(run_dir)
    d = np.load(run_dir / 'map.npz')
    self.kf_id, self.frame, self.T = d['kf_id'], d['kf_frame'], d['kf_T']
    self.pt_id, self.X = d['pt_id'], d['pt_X']
    self.K = d['K']
    row = np.full(max(self.kf_id.max(), d['obs_kf'].max()) + 1, -1)
    row[self.kf_id] = np.arange(len(self.kf_id))
    prow = np.full(max(self.pt_id.max(), d['obs_pt'].max()) + 1, -1)
    prow[self.pt_id] = np.arange(len(self.pt_id))
    ok = (row[d['obs_kf']] >= 0) & (prow[d['obs_pt']] >= 0)
    self.obs_kf, self.obs_pt, self.obs_uv = row[d['obs_kf'][ok]], prow[d['obs_pt'][ok]], d['obs_uv'][ok]
    self.name = run_dir.name.split('@')[0]
    self.data = datasets.load(self.name)
    self.images = {}
    # every frame's (hindsight) pose: frame index in the dataset -> T_cw (NaN where none)
    meta = json.loads((run_dir / 'metrics.json').read_text())
    T_wc = np.load(run_dir / 'trajectory.npz')['poses']
    self.frame_T = np.full((len(self.data), 4, 4), np.nan)
    start = meta.get('start', 0)
    ok = np.isfinite(T_wc[:, 0, 0])
    self.frame_T[start + np.flatnonzero(ok)] = np.linalg.inv(T_wc[ok])
    self.frames = {}
    self.limit = None     # online: only keyframes / frames up to this frame index exist

  def frame_image(self, f):
    if f not in self.frames:
      self.frames[f] = self.data.load(f)
    return self.frames[f]

  def __len__(self):
    return len(self.kf_id)

  def image(self, k):
    if k not in self.images:
      self.images[k] = self.data.load(self.frame[k])
    return self.images[k]

  def color(self, k):
    return self.data.load_color(self.frame[k])

  def display_frame(self):
    '''
    map -> display: the Sim3 that aligns the trajectory to the ground truth (metres, world frame), then the rotation that
    takes the world's up to +Z. Returns a function of points and the rotation (for camera orientations)
    '''
    import geometry
    G = self.data.gt
    C = np.array([np.linalg.inv(T)[:3, 3] if np.isfinite(T[0, 0]) else np.full(3, np.nan) for T in self.frame_T])
    ok = np.isfinite(C).all(axis=1) & np.isfinite(G[:, 0, 3])
    s, R, t = geometry.umeyama(G[ok, :3, 3], C[ok])
    up = self.data.up / np.linalg.norm(self.data.up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-9:
      R_up = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
      V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
      R_up = np.eye(3) + V + V @ V / (1 + c)
    def to_display(X):
      return (s * X @ R.T + t) @ R_up.T
    return to_display, R_up @ R, R_up

  def observations(self, k):
    '''pixels and map-point indices observed by keyframe k'''
    o = self.obs_kf == k
    return self.obs_uv[o], self.obs_pt[o]

  def covisible(self, k, n):
    '''the n keyframes sharing the most points with keyframe k'''
    pts = self.obs_pt[self.obs_kf == k]
    shared = np.bincount(self.obs_kf[np.isin(self.obs_pt, pts)], minlength=len(self))
    shared[k] = 0
    if self.limit is not None:
      shared[self.frame > self.limit] = 0
    order = np.argsort(-shared, kind='stable')
    return order[:n][shared[order[:n]] > 0]


# ------------------------------------------------------------------------------------------------- engines
# an engine maps (recon, keyframe) to an inverse-depth map in map units (NaN: no estimate) and its standard deviation

def sparse(rc, k):
  '''the map points observed by the keyframe, at their pixels'''
  h, w = rc.image(k).shape
  rho, sig = np.full((h, w), np.nan), np.full((h, w), np.nan)
  uv, pts = rc.observations(k)
  z = transform(rc.T[k], rc.X[pts])[:, 2]
  ui, vi = np.round(uv).astype(int).T
  ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
  rho[vi[ok], ui[ok]] = 1 / z[ok]
  sig[vi[ok], ui[ok]] = 0
  return rho, sig


def relative(rc, k, j):
  '''R, t of keyframe j relative to keyframe k (x_j = R x_k + t)'''
  T = rc.T[j] @ np.linalg.inv(rc.T[k])
  return T[:3, :3], T[:3, 3]


def triangulate_flow(K, R, t, uv, q):
  '''
  inverse depth (in keyframe k) of pixels uv seen at q in the other view: q = proj(K (R x / rho + t)), linear in rho after
  multiplying out; returns rho, the distance of q from the epipolar line [px] and |dq/drho| (px per unit inverse depth)
  '''
  x = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
  A = x @ (K @ R).T                       # K R x
  B = K @ t                               # K t
  # q (A_z + rho B_z) = A_xy + rho B_xy  ->  rho (q B_z - B_xy) = A_xy - q A_z
  c = q * B[2] - B[:2]
  a = A[:, :2] - q * A[:, 2:3]
  rho = np.sum(a * c, axis=1) / np.maximum(np.sum(c * c, axis=1), 1e-12)
  den = A[:, 2] + rho * B[2]
  proj = (A[:, :2] + rho[:, None] * B[:2]) / den[:, None]
  epi = np.linalg.norm(proj - q, axis=1)
  dq = (B[:2] * den[:, None] - (A[:, :2] + rho[:, None] * B[:2]) * B[2]) / den[:, None] ** 2
  return rho, epi, np.linalg.norm(dq, axis=1)


def fuse_hypotheses(rho, sig, min_views=2, chi2=5.99):
  '''
  per pixel, hypotheses (N, H, W) with NaN for none: the largest set compatible with one of them (seed) by a chi2 test,
  fused by inverse variance (Mur-Artal & Tardos 2015); NaN where fewer than min_views agree
  '''
  have = np.isfinite(rho)
  d2 = (rho[:, None] - rho[None]) ** 2 / (sig[:, None] ** 2 + sig[None] ** 2)
  compat = (d2 < chi2) & have[:, None] & have[None]
  count = compat.sum(axis=1)                              # (seed, H, W)
  seed = np.argmax(count, axis=0)
  best = np.take_along_axis(compat, seed[None, None], axis=0)[0]     # (N, H, W): members of the best set
  n = best.sum(axis=0)
  w = np.where(best, 1 / np.where(have, sig, 1) ** 2, 0)
  fused = np.sum(w * np.nan_to_num(rho), axis=0) / np.maximum(w.sum(axis=0), 1e-300)
  ok = n >= min_views
  return np.where(ok, fused, np.nan), np.where(ok, 1 / np.sqrt(np.maximum(w.sum(axis=0), 1e-300)), np.nan)


DIS = {}


def dis(rc, k, neighbours=6, epi_px=1.5, fb_px=1.0, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
  '''dense flow (DIS) to the most covisible keyframes, triangulated with the known poses, fused across them'''
  if preset not in DIS:
    DIS[preset] = cv2.DISOpticalFlow_create(preset)
  flow_engine = DIS[preset]
  img = rc.image(k)
  h, w = img.shape
  vv, uu = np.mgrid[0:h, 0:w]
  uv = np.c_[uu.ravel(), vv.ravel()].astype(np.float64)
  rhos, sigs = [], []
  for j in rc.covisible(k, neighbours):
    other = rc.image(j)
    f = flow_engine.calc(img, other, None)
    b = flow_engine.calc(other, img, None)
    q = uv + f.reshape(-1, 2)
    # forward-backward consistency
    back = cv2.remap(b, q[:, 0].reshape(h, w).astype(np.float32), q[:, 1].reshape(h, w).astype(np.float32), cv2.INTER_LINEAR,
                     borderValue=(1e6, 1e6)).reshape(-1, 2)
    fb = np.linalg.norm(f.reshape(-1, 2) + back, axis=1)
    R, t = relative(rc, k, j)
    rho, epi, slope = triangulate_flow(rc.K, R, t, uv, q)
    sigma_px = np.maximum(0.3, fb)
    ok = (fb < fb_px) & (epi < epi_px) & (rho > 0) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    rhos.append(np.where(ok, rho, np.nan).reshape(h, w))
    sigs.append(np.where(ok, sigma_px / np.maximum(slope, 1e-12), np.nan).reshape(h, w))
  if not rhos:
    return np.full((h, w), np.nan), np.full((h, w), np.nan)
  return fuse_hypotheses(np.array(rhos), np.array(sigs))


@njit(cache=True, parallel=True)
def plane_sweep(ref, others, K, R, t, lo, hi, S, half, truncate):
  '''
  cost (h, w, S) of inverse-depth samples rho for every reference pixel: mean over the other images that see it of the
  truncated 1 - ZNCC of a (2 half + 1)^2 patch (fronto-parallel: the patch is sampled at the projected centre, bilinear)
  '''
  h, w = ref.shape
  F = len(others)
  cost = np.zeros((h, w, S), np.float32)
  count = np.zeros((h, w, S), np.int16)
  n = (2 * half + 1) ** 2
  for v in prange(half, h - half):
    a = np.empty(n)
    b = np.empty(n)
    for u in range(half, w - half):
      # reference patch, normalized
      m = 0
      for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
          a[m] = ref[v + dv, u + du]
          m += 1
      a -= a.mean()
      na = np.sqrt(np.sum(a * a))
      if na < 1e-3:
        continue
      if not hi[v, u] > lo[v, u]:
        continue
      x0 = (u - K[0, 2]) / K[0, 0]
      x1 = (v - K[1, 2]) / K[1, 1]
      step = (hi[v, u] - lo[v, u]) / (S - 1)
      for f in range(F):
        # p(rho) = K (R x + rho t) / z: A + rho B
        A0 = K[0, 0] * (R[f, 0, 0] * x0 + R[f, 0, 1] * x1 + R[f, 0, 2]) + K[0, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A1 = K[1, 1] * (R[f, 1, 0] * x0 + R[f, 1, 1] * x1 + R[f, 1, 2]) + K[1, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A2 = R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2]
        B0 = K[0, 0] * t[f, 0] + K[0, 2] * t[f, 2]
        B1 = K[1, 1] * t[f, 1] + K[1, 2] * t[f, 2]
        B2 = t[f, 2]
        for s in range(S):
          rs = lo[v, u] + s * step
          z = A2 + rs * B2
          if z <= 1e-6:
            continue
          pu = (A0 + rs * B0) / z
          pv = (A1 + rs * B1) / z
          if pu < half or pv < half or pu >= w - half - 1 or pv >= h - half - 1:
            continue
          iu, iv = int(pu), int(pv)
          fu, fv = pu - iu, pv - iv
          m = 0
          for dv in range(-half, half + 1):
            for du in range(-half, half + 1):
              y, x = iv + dv, iu + du
              b[m] = ((1 - fu) * (1 - fv) * others[f, y, x] + fu * (1 - fv) * others[f, y, x + 1] +
                      (1 - fu) * fv * others[f, y + 1, x] + fu * fv * others[f, y + 1, x + 1])
              m += 1
          b -= b.mean()
          nb = np.sqrt(np.sum(b * b))
          c = 1.0 if nb < 1e-3 else 1.0 - np.sum(a * b) / (na * nb)
          cost[v, u, s] += min(c, truncate)
          count[v, u, s] += 1
  return cost, count


def sweep(rc, k, lo, hi, offsets=(1, 2, 4, 8, 16, 32), keyframes=6, samples=64, half=2, scale=2, min_views=3,
          uniqueness=0.8):
  '''
  multiple-baseline stereo (Okutomi & Kanade; the variable baselines of semi-dense VO): matching costs over frames at
  several baselines (frames at the offsets + the most covisible keyframes) summed as a function of inverse depth, in
  per-pixel ranges [lo, hi] (full-resolution arrays, NaN: skip). Small baselines give one wide valley, large ones a
  sharp minimum among false ones; their sum has one sharp valley. Half resolution; returned at full resolution
  '''
  f0 = rc.frame[k]
  T0 = rc.T[k]
  frames = [f0 + s * o for o in offsets for s in (-1, 1) if 0 <= f0 + s * o < len(rc.frame_T) and np.isfinite(rc.frame_T[f0 + s * o, 0, 0])
            and (rc.limit is None or f0 + s * o <= rc.limit)]
  frames += [int(rc.frame[j]) for j in rc.covisible(k, keyframes) if rc.frame[j] not in frames]
  full = rc.image(k)
  H, W = full.shape
  if not frames:
    return np.full((H, W), np.nan), np.full((H, W), np.nan)
  h, w = H // scale, W // scale
  ref = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
  others = np.array([cv2.resize(rc.frame_image(f), (w, h), interpolation=cv2.INTER_AREA) for f in frames], np.float32)
  Ks = rc.K.copy()
  Ks[:2] /= scale
  Ks[:2, 2] = (rc.K[:2, 2] + 0.5) / scale - 0.5
  rel = np.array([rc.frame_T[f] @ np.linalg.inv(T0) if f != f0 else np.eye(4) for f in frames])
  rel = np.array([rc.frame_T[f] @ np.linalg.inv(rc.frame_T[f0]) if np.isfinite(rc.frame_T[f0, 0, 0]) else rel[i]
                  for i, f in enumerate(frames)])
  lo_s = cv2.resize(lo, (w, h), interpolation=cv2.INTER_NEAREST)
  hi_s = cv2.resize(hi, (w, h), interpolation=cv2.INTER_NEAREST)
  cost, count = plane_sweep(ref, others, Ks, rel[:, :3, :3].copy(), rel[:, :3, 3].copy(), np.nan_to_num(lo_s, nan=1.0),
                            np.nan_to_num(hi_s, nan=0.0), samples, half, 1.0)
  mean = np.where(count >= min_views, cost / np.maximum(count, 1), np.inf)
  best = np.argmin(mean, axis=2)
  c0 = np.take_along_axis(mean, best[..., None], 2)[..., 0]
  far = np.abs(np.arange(samples)[None, None] - best[..., None]) > 2
  c1 = np.min(np.where(far, mean, np.inf), axis=2)
  bl, br = np.clip(best - 1, 0, samples - 1), np.clip(best + 1, 0, samples - 1)
  cl, cr = np.take_along_axis(mean, bl[..., None], 2)[..., 0], np.take_along_axis(mean, br[..., None], 2)[..., 0]
  curv = cl + cr - 2 * c0
  offset = np.where((curv > 1e-6) & (best > 0) & (best < samples - 1), 0.5 * (cl - cr) / np.maximum(curv, 1e-6), 0)
  step = (hi_s - lo_s) / (samples - 1)
  r = lo_s + (best + offset) * step
  ok = np.isfinite(c0) & (c0 < uniqueness * c1) & (best > 0) & (best < samples - 1) & (curv > 0) & np.isfinite(step)
  out = np.where(ok, r, np.nan).astype(np.float64)
  sig = np.where(ok, step / np.sqrt(np.maximum(curv, 1e-6)) * np.sqrt(np.maximum(c0, 1e-3)), np.nan)
  return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST), cv2.resize(sig, (W, H), interpolation=cv2.INTER_NEAREST)


def sparse_range(rc, k):
  uv, pts = rc.observations(k)
  z = transform(rc.T[k], rc.X[pts])[:, 2]
  inv = 1 / z[z > 0]
  return np.percentile(inv, 1) * 0.5, np.percentile(inv, 99) * 1.5


def multi_baseline(rc, k, **kw):
  '''the plane sweep over the keyframe's whole inverse-depth range (from its sparse points)'''
  H, W = rc.image(k).shape
  a, b = sparse_range(rc, k)
  return sweep(rc, k, np.full((H, W), a), np.full((H, W), b), samples=96, **kw)


def combined(rc, k, window=3.0, **kw):
  '''
  DIS gives the wide valley (a prior: rho +- window sigma), the multi-baseline sweep pinpoints the minimum in it;
  where DIS has nothing, the sweep searches the whole range
  '''
  rho, sig = dis(rc, k)
  H, W = rho.shape
  a, b = sparse_range(rc, k)
  have = np.isfinite(rho)
  half_width = np.maximum(window * sig, 0.05 * rho)
  lo = np.where(have, np.maximum(rho - half_width, a * 0.5), a)
  hi = np.where(have, rho + half_width, b)
  out, s = sweep(rc, k, lo, hi, samples=32 if kw.get('samples') is None else kw.pop('samples'), **kw)
  # inside a DIS window the prior disambiguates: keep DIS where the sweep found nothing
  out = np.where(np.isfinite(out), out, rho)
  s = np.where(np.isfinite(s), s, sig)
  return out, s


@njit(cache=True, parallel=True)
def filter_update(ref, gx, gy, img, K, R, t, rho, var, n_upd, n_out, half, sigma_i, min_grad, max_cost, max_samples):
  '''
  one frame's update of the per-pixel inverse-depth filter of a keyframe (in place): search rho +- 2 sigma along the
  epipolar line (3..max_samples samples, ~0.5 px apart), 1 - ZNCC of a patch, parabola at the best sample. Measurement
  noise as in semi-dense VO: the patch localizes along the epipolar line only as well as its intensity gradient along
  the line allows, sigma_px = sigma_i / rms(gradient . line direction) (the aperture problem: no update when it is
  below min_grad); sigma_rho = sigma_px / |d pixel / d rho|; chi2-gated inverse-variance update, else an outlier
  '''
  h, w = ref.shape
  n = (2 * half + 1) ** 2
  B0 = K[0, 0] * t[0] + K[0, 2] * t[2]
  B1 = K[1, 1] * t[1] + K[1, 2] * t[2]
  B2 = t[2]
  # the other camera's centre in the keyframe: c = -R^T t; its image (the epipole) defines the epipolar lines
  c0 = -(R[0, 0] * t[0] + R[1, 0] * t[1] + R[2, 0] * t[2])
  c1 = -(R[0, 1] * t[0] + R[1, 1] * t[1] + R[2, 1] * t[2])
  c2 = -(R[0, 2] * t[0] + R[1, 2] * t[1] + R[2, 2] * t[2])
  for v in prange(half, h - half):
    a = np.empty(n)
    b = np.empty(n)
    costs = np.empty(max_samples)
    for u in range(half, w - half):
      r0 = rho[v, u]
      if not (r0 > 0) or n_out[v, u] < 0:
        continue
      sd = np.sqrt(var[v, u])
      lo = max(r0 - 2 * sd, 1e-6)
      hi = r0 + 2 * sd
      # epipolar direction at the pixel in the keyframe, and the patch gradient along it
      if abs(c2) > 1e-9:
        lx, ly = u - (K[0, 0] * c0 / c2 + K[0, 2]), v - (K[1, 1] * c1 / c2 + K[1, 2])
      else:
        lx, ly = K[0, 0] * c0, K[1, 1] * c1
      ln = np.hypot(lx, ly)
      if ln < 1e-9:
        continue
      lx, ly = lx / ln, ly / ln
      g2 = 0.0
      for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
          gl = gx[v + dv, u + du] * lx + gy[v + dv, u + du] * ly
          g2 += gl * gl
      g_line = np.sqrt(g2 / n)
      if g_line < min_grad:
        continue
      px_sigma = sigma_i / g_line
      x0 = (u - K[0, 2]) / K[0, 0]
      x1 = (v - K[1, 2]) / K[1, 1]
      A0 = K[0, 0] * (R[0, 0] * x0 + R[0, 1] * x1 + R[0, 2]) + K[0, 2] * (R[2, 0] * x0 + R[2, 1] * x1 + R[2, 2])
      A1 = K[1, 1] * (R[1, 0] * x0 + R[1, 1] * x1 + R[1, 2]) + K[1, 2] * (R[2, 0] * x0 + R[2, 1] * x1 + R[2, 2])
      A2 = R[2, 0] * x0 + R[2, 1] * x1 + R[2, 2]
      zl, zh = A2 + lo * B2, A2 + hi * B2
      if zl <= 1e-6 or zh <= 1e-6:
        continue
      span = np.hypot((A0 + hi * B0) / zh - (A0 + lo * B0) / zl, (A1 + hi * B1) / zh - (A1 + lo * B1) / zl)
      S = max(int(span / 0.5) + 1, 3)
      if S > max_samples:
        continue      # (the window is too wide for this baseline: wait until smaller baselines shrank it)
      # reference patch
      m = 0
      for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
          a[m] = ref[v + dv, u + du]
          m += 1
      a -= a.mean()
      na = np.sqrt(np.sum(a * a))
      if na < 1e-3:
        continue
      best, ib = 1e9, -1
      for s in range(S):
        rs = lo + (hi - lo) * s / (S - 1)
        z = A2 + rs * B2
        pu = (A0 + rs * B0) / z
        pv = (A1 + rs * B1) / z
        costs[s] = 1e9
        if pu < half or pv < half or pu >= w - half - 1 or pv >= h - half - 1:
          continue
        iu, iv = int(pu), int(pv)
        fu, fv = pu - iu, pv - iv
        m = 0
        for dv in range(-half, half + 1):
          for du in range(-half, half + 1):
            y, x = iv + dv, iu + du
            b[m] = ((1 - fu) * (1 - fv) * img[y, x] + fu * (1 - fv) * img[y, x + 1] + (1 - fu) * fv * img[y + 1, x] +
                    fu * fv * img[y + 1, x + 1])
            m += 1
        b -= b.mean()
        nb = np.sqrt(np.sum(b * b))
        c = 1.0 if nb < 1e-3 else 1.0 - np.sum(a * b) / (na * nb)
        costs[s] = c
        if c < best:
          best, ib = c, s
      if ib < 0 or best > max_cost:
        continue
      off = 0.0
      if 0 < ib < S - 1 and costs[ib - 1] < 1e8 and costs[ib + 1] < 1e8:
        cu = costs[ib - 1] + costs[ib + 1] - 2 * best
        if cu > 1e-9:
          off = 0.5 * (costs[ib - 1] - costs[ib + 1]) / cu
      elif ib == 0 or ib == S - 1:
        continue      # (at the edge of the window: the minimum may be outside)
      rm = lo + (hi - lo) * (ib + off) / (S - 1)
      z = A2 + rm * B2
      du_dr = (B0 * z - (A0 + rm * B0) * B2) / (z * z)
      dv_dr = (B1 * z - (A1 + rm * B1) * B2) / (z * z)
      slope = np.hypot(du_dr, dv_dr)
      if slope < 1e-9:
        continue
      vm = (px_sigma / slope) ** 2
      if (rm - r0) ** 2 / (var[v, u] + vm) < 5.99:
        rho[v, u] = (r0 * vm + rm * var[v, u]) / (var[v, u] + vm)
        var[v, u] = var[v, u] * vm / (var[v, u] + vm)
        n_upd[v, u] += 1
      else:
        n_out[v, u] += 1
        if n_out[v, u] > 3 and n_out[v, u] > n_upd[v, u]:
          n_out[v, u] = -1      # (invalid)


def calibrate(rc, k, rho, sig):
  '''
  inflation of sigma so that the errors at the keyframe's sparse map points (inverse depth from BA, much more
  precise) have the median |z| of a Gaussian (0.6745); >= 1 (sigma is never shrunk)
  '''
  uv, pts = rc.observations(k)
  zs = transform(rc.T[k], rc.X[pts])[:, 2]
  ui, vi = np.round(uv).astype(int).T
  h, w = rho.shape
  ok = (zs > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
  r, s = rho[vi[ok], ui[ok]], sig[vi[ok], ui[ok]]
  have = np.isfinite(r) & (r > 0) & np.isfinite(s) & (s > 0)
  if have.sum() < 10:
    return 1.0
  z = np.abs(r[have] - 1 / zs[ok][have]) / s[have]
  return max(1.0, float(np.median(z)) / 0.6745)


def frame_parallax(rc, k, f, depth):
  '''baseline of frame f relative to keyframe k, as an angle at the median scene depth [deg]'''
  C0 = -rc.T[k][:3, :3].T @ rc.T[k][:3, 3]
  T = rc.frame_T[f]
  C = -T[:3, :3].T @ T[:3, 3]
  return np.degrees(np.linalg.norm(C - C0) / depth)


def sequential(rc, k, window=40, init_parallax=0.5, half=2, sigma_i=2.0, min_grad=2.0, max_cost=0.5, min_updates=3):
  '''
  the depth filter of semi-dense VO, dense and seeded by DIS: the nearest frames with init_parallax give the first
  (unambiguous) estimate, every further frame (increasing baseline) searches only rho +- 2 sigma and tightens it
  '''
  f0 = rc.frame[k]
  img = rc.image(k).astype(np.float32)
  H, W = img.shape
  uv, pts = rc.observations(k)
  zs = transform(rc.T[k], rc.X[pts])[:, 2]
  depth = float(np.median(zs[zs > 0]))
  T0 = rc.frame_T[f0] if np.isfinite(rc.frame_T[f0, 0, 0]) else rc.T[k]
  frames = [f for f in range(max(f0 - window, 0), min(f0 + window + 1, len(rc.frame_T)))
            if f != f0 and np.isfinite(rc.frame_T[f, 0, 0])]
  par = np.array([frame_parallax(rc, k, f, depth) for f in frames])
  order = np.argsort(par)
  # initialization: DIS to the nearest frame on each side with enough parallax
  init = []
  for side in (-1, 1):
    cand = [i for i in order if (frames[i] - f0) * side > 0 and par[i] >= init_parallax]
    if cand:
      init.append(frames[cand[0]])
  if not init:
    return np.full((H, W), np.nan), np.full((H, W), np.nan)
  if rc.frame_T is not None:
    pass
  vv, uu = np.mgrid[0:H, 0:W]
  uvf = np.c_[uu.ravel(), vv.ravel()].astype(np.float64)
  if cv2.DISOPTICAL_FLOW_PRESET_MEDIUM not in DIS:
    DIS[cv2.DISOPTICAL_FLOW_PRESET_MEDIUM] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
  engine = DIS[cv2.DISOPTICAL_FLOW_PRESET_MEDIUM]
  rhos, sigs = [], []
  img8 = rc.image(k)
  for f in init:
    other = rc.frame_image(f)
    fl = engine.calc(img8, other, None)
    bk = engine.calc(other, img8, None)
    q = uvf + fl.reshape(-1, 2)
    back = cv2.remap(bk, q[:, 0].reshape(H, W).astype(np.float32), q[:, 1].reshape(H, W).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=(1e6, 1e6)).reshape(-1, 2)
    fb = np.linalg.norm(fl.reshape(-1, 2) + back, axis=1)
    rel = rc.frame_T[f] @ np.linalg.inv(T0)
    r, epi, slope = triangulate_flow(rc.K, rel[:3, :3], rel[:3, 3], uvf, q)
    ok = (fb < 1.0) & (epi < 1.5) & (r > 0)
    rhos.append(np.where(ok, r, np.nan).reshape(H, W))
    sigs.append(np.where(ok, np.maximum(0.3, fb) / np.maximum(slope, 1e-12), np.nan).reshape(H, W))
  if len(rhos) == 2:
    rho, sig = fuse_hypotheses(np.array(rhos), np.array(sigs), min_views=1)
  else:
    rho, sig = rhos[0], sigs[0]
  sig = sig * calibrate(rc, k, rho, sig)
  rho = np.where(np.isfinite(rho), rho, -1.0)
  var = np.where(np.isfinite(sig), sig ** 2, 1.0)
  n_upd = np.zeros((H, W), np.int32)
  n_out = np.zeros((H, W), np.int32)
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8
  for i in order:
    f = frames[i]
    if f in init:
      continue
    rel = rc.frame_T[f] @ np.linalg.inv(T0)
    filter_update(img, gx, gy, rc.frame_image(f).astype(np.float32), rc.K, rel[:3, :3].copy(), rel[:3, 3].copy(), rho,
                  var, n_upd, n_out, half, sigma_i, min_grad, max_cost, 64)
  ok = (rho > 0) & (n_upd >= min_updates) & (n_out >= 0)
  rho, sig = np.where(ok, rho, np.nan), np.where(ok, np.sqrt(var), np.nan)
  return rho, sig * calibrate(rc, k, rho, sig)


def refined(rc, k, window=40, half=2, sigma_i=2.0, min_grad=2.0, max_cost=0.5):
  '''
  DIS to the covisible keyframes (BA poses, medium baselines) seeds the filter; the frames around the keyframe then
  refine it in order of increasing baseline, each searching only rho +- 2 sigma (a frame whose window is too wide for
  its baseline waits); pixels without updates keep the seed
  '''
  rho0, sig0 = dis(rc, k)
  sig0 = sig0 * calibrate(rc, k, rho0, sig0)
  img = rc.image(k).astype(np.float32)
  H, W = img.shape
  f0 = rc.frame[k]
  uv, pts = rc.observations(k)
  zs = transform(rc.T[k], rc.X[pts])[:, 2]
  depth = float(np.median(zs[zs > 0]))
  T0 = rc.frame_T[f0] if np.isfinite(rc.frame_T[f0, 0, 0]) else rc.T[k]
  frames = [f for f in range(max(f0 - window, 0), min(f0 + window + 1, len(rc.frame_T)))
            if f != f0 and np.isfinite(rc.frame_T[f, 0, 0])]
  frames += [int(rc.frame[j]) for j in rc.covisible(k, 6) if rc.frame[j] not in frames]
  par = np.array([frame_parallax(rc, k, f, depth) for f in frames])
  rho = np.where(np.isfinite(rho0), rho0, -1.0)
  var = np.where(np.isfinite(sig0), sig0 ** 2, 1.0)
  n_upd = np.zeros((H, W), np.int32)
  n_out = np.zeros((H, W), np.int32)
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8
  for i in np.argsort(par):
    f = frames[i]
    rel = rc.frame_T[f] @ np.linalg.inv(T0)
    filter_update(img, gx, gy, rc.frame_image(f).astype(np.float32), rc.K, rel[:3, :3].copy(), rel[:3, 3].copy(), rho,
                  var, n_upd, n_out, half, sigma_i, min_grad, max_cost, 64)
  ok = (rho > 0) & (n_out >= 0)
  out, sig = np.where(ok, rho, np.nan), np.where(ok, np.sqrt(var), np.nan)
  return out, sig * calibrate(rc, k, out, sig)


@njit(cache=True, nogil=True)
def pm_cost(ref, others, K, Kinv, R, t, u, v, z, n0, n1, n2, half, best_k, sigma_c=20.0):
  '''
  slanted-plane matching cost of reference pixel (u, v) with the plane through its point at depth z with normal n:
  every patch sample is intersected with the plane and projected into each view; per view 1 - ZNCC, bilaterally
  weighted (COLMAP: samples unlike the centre pixel count less, against foreground fattening); the mean of the best_k
  views (robust to occlusion)
  '''
  h, w = ref.shape
  F = len(others)
  x0 = Kinv[0, 0] * u + Kinv[0, 2]
  x1 = Kinv[1, 1] * v + Kinv[1, 2]
  nd = n0 * x0 * z + n1 * x1 * z + n2 * z      # n . X
  m = (2 * half + 1) ** 2
  a = np.empty(m)
  b = np.empty(m)
  wt = np.empty(m)
  costs = np.empty(F)
  c0 = ref[v, u]
  ss = 2.0 * half * half
  k = 0
  for dv in range(-half, half + 1):
    for du in range(-half, half + 1):
      a[k] = ref[v + dv, u + du]
      d = a[k] - c0
      wt[k] = np.exp(-d * d / (2 * sigma_c * sigma_c) - (du * du + dv * dv) / ss)
      k += 1
  sw = np.sum(wt)
  a -= np.sum(wt * a) / sw
  na = np.sqrt(np.sum(wt * a * a))
  if na < 1e-3:
    return 2.0
  for f in range(F):
    ok = True
    k = 0
    for dv in range(-half, half + 1):
      for du in range(-half, half + 1):
        q0 = Kinv[0, 0] * (u + du) + Kinv[0, 2]
        q1 = Kinv[1, 1] * (v + dv) + Kinv[1, 2]
        den = n0 * q0 + n1 * q1 + n2
        if abs(den) < 1e-9:
          ok = False
          break
        s = nd / den
        if s <= 0:
          ok = False
          break
        Y0 = R[f, 0, 0] * q0 * s + R[f, 0, 1] * q1 * s + R[f, 0, 2] * s + t[f, 0]
        Y1 = R[f, 1, 0] * q0 * s + R[f, 1, 1] * q1 * s + R[f, 1, 2] * s + t[f, 1]
        Y2 = R[f, 2, 0] * q0 * s + R[f, 2, 1] * q1 * s + R[f, 2, 2] * s + t[f, 2]
        if Y2 <= 1e-6:
          ok = False
          break
        pu = K[0, 0] * Y0 / Y2 + K[0, 2]
        pv = K[1, 1] * Y1 / Y2 + K[1, 2]
        if pu < 0 or pv < 0 or pu >= w - 1 or pv >= h - 1:
          ok = False
          break
        iu, iv = int(pu), int(pv)
        fu, fv = pu - iu, pv - iv
        b[k] = ((1 - fu) * (1 - fv) * others[f, iv, iu] + fu * (1 - fv) * others[f, iv, iu + 1] +
                (1 - fu) * fv * others[f, iv + 1, iu] + fu * fv * others[f, iv + 1, iu + 1])
        k += 1
      if not ok:
        break
    if not ok:
      costs[f] = 2.0
      continue
    b -= np.sum(wt * b) / sw
    nb = np.sqrt(np.sum(wt * b * b))
    costs[f] = 2.0 if nb < 1e-3 else 1.0 - np.sum(wt * a * b) / (na * nb)
  costs.sort()
  kk = min(best_k, F)
  return np.mean(costs[:kk])


@njit(cache=True, parallel=True)
def pm_iterate(ref, others, K, Kinv, R, t, z, n, cost, lo, hi, half, best_k, scale, seed):
  '''one PatchMatch iteration: red-black checkerboard; each pixel tries its neighbours' planes (near and far) and a
  random perturbation of its own; keeps the cheapest'''
  h, w = ref.shape
  for color in range(2):
    for v in prange(half, h - half):
      np.random.seed(seed + v * 7919 + color)
      for u in range(half + (v + color) % 2, w - half, 2):
        best = cost[v, u]
        bz, b0, b1, b2 = z[v, u], n[v, u, 0], n[v, u, 1], n[v, u, 2]
        for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1), (-5, 0), (5, 0), (0, -5), (0, 5)):
          y, x = v + dy, u + dx
          if y < half or y >= h - half or x < half or x >= w - half:
            continue
          # the neighbour's plane, evaluated at this pixel: depth where this pixel's ray meets it
          xn0 = Kinv[0, 0] * x + Kinv[0, 2]
          xn1 = Kinv[1, 1] * y + Kinv[1, 2]
          m0, m1, m2 = n[y, x, 0], n[y, x, 1], n[y, x, 2]
          ndn = (m0 * xn0 + m1 * xn1 + m2) * z[y, x]
          x0 = Kinv[0, 0] * u + Kinv[0, 2]
          x1 = Kinv[1, 1] * v + Kinv[1, 2]
          den = m0 * x0 + m1 * x1 + m2
          if abs(den) < 1e-9:
            continue
          zc = ndn / den
          if zc < lo or zc > hi:
            continue
          c = pm_cost(ref, others, K, Kinv, R, t, u, v, zc, m0, m1, m2, half, best_k)
          if c < best:
            best, bz, b0, b1, b2 = c, zc, m0, m1, m2
        # random refinement of the current best
        zr = bz * np.exp(np.random.normal() * 0.05 * scale)
        r0, r1, r2 = b0 + np.random.normal() * 0.15 * scale, b1 + np.random.normal() * 0.15 * scale, b2 + np.random.normal() * 0.15 * scale
        nr = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
        r0, r1, r2 = r0 / nr, r1 / nr, r2 / nr
        x0 = Kinv[0, 0] * u + Kinv[0, 2]
        x1 = Kinv[1, 1] * v + Kinv[1, 2]
        if r0 * x0 + r1 * x1 + r2 < 0 and lo <= zr <= hi:     # (facing the camera)
          c = pm_cost(ref, others, K, Kinv, R, t, u, v, zr, r0, r1, r2, half, best_k)
          if c < best:
            best, bz, b0, b1, b2 = c, zr, r0, r1, r2
        # a random normal (facing the camera) at the current depth, the current normal at a random depth
        r0, r1, r2 = np.random.normal(), np.random.normal(), np.random.normal()
        nr = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
        r0, r1, r2 = r0 / nr, r1 / nr, r2 / nr
        if r0 * x0 + r1 * x1 + r2 > 0:
          r0, r1, r2 = -r0, -r1, -r2
        if r0 * x0 + r1 * x1 + r2 < -0.1:
          c = pm_cost(ref, others, K, Kinv, R, t, u, v, bz, r0, r1, r2, half, best_k)
          if c < best:
            best, b0, b1, b2 = c, r0, r1, r2
        zr = 1.0 / (1.0 / hi + np.random.random() * (1.0 / lo - 1.0 / hi))
        c = pm_cost(ref, others, K, Kinv, R, t, u, v, zr, b0, b1, b2, half, best_k)
        if c < best:
          best, bz = c, zr
        cost[v, u] = best
        z[v, u] = bz
        n[v, u, 0], n[v, u, 1], n[v, u, 2] = b0, b1, b2


def patchmatch(rc, k, views=6, half=5, best_k=3, iterations=4, scale=2, max_cost=0.5, init='combined', planes=0,
               plane_tol=0.05, full=True):
  '''
  multi-view PatchMatch with slanted planes (Gipuma / COLMAP style, ZNCC instead of Google's learned HashMatch): per
  pixel a depth and a normal; patches are warped into the covisible keyframes through the plane, so surfaces seen at
  grazing angles (floors) still match; red-black propagation + random refinement; half resolution. With planes > 0
  the dominant planes of the confident pixels are offered to every pixel afterwards (a planar prior, after ACMP): a
  pixel takes a plane if it matches nearly as well (cost within plane_tol)
  '''
  img = rc.image(k)
  H, W = img.shape
  nbrs = rc.covisible(k, views)
  if len(nbrs) == 0:
    return np.full((H, W), np.nan), np.full((H, W), np.nan)
  h, w = H // scale, W // scale
  ref = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
  others = np.array([cv2.resize(rc.image(j), (w, h), interpolation=cv2.INTER_AREA) for j in nbrs], np.float32)
  Ks = rc.K.copy()
  Ks[:2] /= scale
  Ks[:2, 2] = (rc.K[:2, 2] + 0.5) / scale - 0.5
  Kinv = np.linalg.inv(Ks)
  rel = np.array([rc.T[j] @ np.linalg.inv(rc.T[k]) for j in nbrs])
  R, t = rel[:, :3, :3].copy(), rel[:, :3, 3].copy()
  a, b = sparse_range(rc, k)
  lo, hi = 1 / b, 1 / max(a, 1e-9)       # depth range
  rng = np.random.default_rng(k)
  z = 1 / rng.uniform(a, b, (h, w))
  if init == 'combined':
    r0, _ = combined(rc, k)
    r0 = cv2.resize(r0, (w, h), interpolation=cv2.INTER_NEAREST)
    z = np.where(np.isfinite(r0) & (r0 > 0), 1 / np.where(r0 > 0, r0, 1), z)
  z = np.clip(z, lo, hi)
  n = np.zeros((h, w, 3))
  n[..., 2] = -1.0
  cost = pm_init_costs(ref, others, Ks, Kinv, R, t, z, n, half, best_k)
  for it in range(iterations):
    pm_iterate(ref, others, Ks, Kinv, R, t, z, n, cost, lo, hi, half, best_k, 0.5 ** it, 1000 * it + k)
  vv, uu = np.mgrid[0:h, 0:w]
  ray = np.stack([Kinv[0, 0] * uu + Kinv[0, 2], Kinv[1, 1] * vv + Kinv[1, 2], np.ones((h, w))], -1)
  if planes:
    good = cost < 0.5 * max_cost
    for p_n, p_d in fit_planes(ray[good] * z[good][:, None], planes, rng):
      den = ray @ p_n
      zp = np.where(np.abs(den) > 1e-9, p_d / np.where(np.abs(den) > 1e-9, den, 1), -1)
      inside = (zp >= lo) & (zp <= hi)
      zp = np.where(inside, zp, z)
      cp = pm_init_costs(ref, others, Ks, Kinv, R, t, zp, np.broadcast_to(p_n, (h, w, 3)).copy(), half, best_k)
      take = inside & (cp < max_cost) & (cp < cost + plane_tol)
      z, cost = np.where(take, zp, z), np.where(take, np.minimum(cp, cost), cost)
      n[take] = p_n
  ok = cost < max_cost
  rho = np.where(ok, 1 / z, np.nan)
  sig = np.where(ok, 0.05 * rho * (0.5 + cost), np.nan)      # (scaled by calibrate())
  if not full:
    return rho, sig, n, cost
  return cv2.resize(rho, (W, H), interpolation=cv2.INTER_NEAREST), cv2.resize(sig, (W, H), interpolation=cv2.INTER_NEAREST)


def fit_planes(X, max_planes, rng, rel_tol=0.01, min_share=0.08, iterations=500):
  '''
  sequential RANSAC for the dominant planes n . X = d (camera frame, n towards the camera) of a point cloud; residuals
  relative to depth; a plane needs min_share of the points; refit by least squares on the inliers
  '''
  out = []
  left = X
  N0 = len(X)
  for _ in range(max_planes):
    if len(left) < 3 or len(left) < min_share * N0:
      break
    best, best_in = None, None
    for _ in range(iterations):
      P = left[rng.choice(len(left), 3, replace=False)]
      nrm = np.cross(P[1] - P[0], P[2] - P[0])
      if np.linalg.norm(nrm) < 1e-12:
        continue
      nrm /= np.linalg.norm(nrm)
      inl = np.abs(left @ nrm - nrm @ P[0]) < rel_tol * left[:, 2]
      if best_in is None or inl.sum() > best_in.sum():
        best, best_in = nrm, inl
    if best_in is None or best_in.sum() < min_share * N0:
      break
    Q = left[best_in]
    c = Q.mean(0)
    nrm = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    d = nrm @ c
    if d > 0:                              # (normal facing the camera: n . X < 0 at the plane)
      nrm, d = -nrm, -d
    out.append((nrm, d))
    left = left[~(np.abs(left @ nrm - d) < rel_tol * left[:, 2])]
  return out


@njit(cache=True, parallel=True)
def pm_init_costs(ref, others, K, Kinv, R, t, z, n, half, best_k):
  h, w = ref.shape
  cost = np.full((h, w), 2.0)
  for v in prange(half, h - half):
    for u in range(half, w - half):
      cost[v, u] = pm_cost(ref, others, K, Kinv, R, t, u, v, z[v, u], n[v, u, 0], n[v, u, 1], n[v, u, 2], half, best_k)
  return cost


# ------------------------------------------------------------------------------------------------- Depth from Motion
# Google's pipeline (Valentin et al. 2018, Depth from Motion for Smartphone AR): a learned binary matching cost
# (UltraStereo, Fanello et al. 2017), CRF inference by row-parallel PatchMatch / HashMatch passes, invalidation by
# CRF cost and connected components, densification by the planar bilateral solver. Multi-view (best views of the
# covisible keyframes) instead of one polar-rectified pair

def train_hash(images, half=5, bits=32, nonzero=4, samples=20000, proposals=100, seed=0):
  '''
  UltraStereo's unsupervised binary mapping: greedy splits of a tree (breadth first, bits nodes) on image patches
  (normalized: zero mean, unit norm; passive images instead of an active pattern), each a sparse hyperplane z
  (nonzero taps) and threshold theta, chosen among random proposals by the information gain under a Gaussian model
  of the patches (log-determinants of the covariances). Returns pixel offsets (bits, nonzero, 2), weights, thresholds
  '''
  rng = np.random.default_rng(seed)
  W = 2 * half + 1
  X = []
  for img in images:
    img = img.astype(np.float32)
    h, w = img.shape
    vv = rng.integers(half, h - half, samples // len(images))
    uu = rng.integers(half, w - half, samples // len(images))
    X.append(np.array([img[v - half:v + half + 1, u - half:u + half + 1].ravel() for v, u in zip(vv, uu)]))
  X = np.concatenate(X)
  X -= X.mean(1, keepdims=True)
  X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
  def logdet(Y):
    if len(Y) <= Y.shape[1]:
      return 0.0
    C = np.cov(Y, rowvar=False) + 1e-6 * np.eye(Y.shape[1])
    return np.linalg.slogdet(C)[1]
  offsets, weights, thetas = [], [], []
  queue = [np.arange(len(X))]
  while len(offsets) < bits:
    idx = queue.pop(0)
    S = X[idx]
    H = logdet(S)
    best = None
    for _ in range(proposals):
      taps = rng.choice(W * W, nonzero, replace=False)
      z = rng.normal(size=nonzero)
      p = S[:, taps] @ z
      th = np.quantile(p, rng.uniform(0.3, 0.7))
      L = p < th
      gain = H - (L.mean() * logdet(S[L]) + (~L).mean() * logdet(S[~L]))
      if best is None or gain > best[0]:
        best = (gain, taps, z, th, L)
    _, taps, z, th, L = best
    offsets.append(np.c_[taps // W - half, taps % W - half])
    weights.append(z)
    thetas.append(th)
    queue += [idx[L], idx[~L]]
  return np.array(offsets), np.array(weights), np.array(thetas)


def hash_codes(img, hsh, half=5):
  '''every pixel's 32-bit code (int64) of its normalized patch'''
  offsets, weights, thetas = hsh
  img = img.astype(np.float32)
  W = 2 * half + 1
  mu = cv2.boxFilter(img, -1, (W, W))
  norm = np.sqrt(np.maximum(cv2.boxFilter(img * img, -1, (W, W)) * W * W - mu * mu * W * W, 1e-6))
  pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
  h, w = img.shape
  code = np.zeros((h, w), np.int64)
  for b in range(len(thetas)):
    p = np.zeros((h, w), np.float32)
    for (dv, du), z in zip(offsets[b], weights[b]):
      p += z * (pad[half + dv:half + dv + h, half + du:half + du + w] - mu)
    code |= (p / norm >= thetas[b]).astype(np.int64) << b
  return code


@njit(cache=True, nogil=True)
def popcount(x):
  x = x - ((x >> 1) & 0x55555555)
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
  x = (x + (x >> 4)) & 0x0F0F0F0F
  return ((x * 0x01010101) & 0xFFFFFFFF) >> 24


@njit(cache=True, nogil=True)
def hash_unary(code0, codes, K, Kinv, R, t, u, v, rho, best_k):
  '''mean Hamming distance over the best_k views of pixel (u, v) at inverse depth rho; bilinear in the target pixel'''
  F, h, w = codes.shape
  c0 = code0[v, u]
  x0 = (Kinv[0, 0] * u + Kinv[0, 2]) / rho
  x1 = (Kinv[1, 1] * v + Kinv[1, 2]) / rho
  x2 = 1.0 / rho
  costs = np.empty(F)
  for f in range(F):
    Y0 = R[f, 0, 0] * x0 + R[f, 0, 1] * x1 + R[f, 0, 2] * x2 + t[f, 0]
    Y1 = R[f, 1, 0] * x0 + R[f, 1, 1] * x1 + R[f, 1, 2] * x2 + t[f, 1]
    Y2 = R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2] * x2 + t[f, 2]
    if Y2 <= 1e-9:
      costs[f] = 32.0
      continue
    pu = K[0, 0] * Y0 / Y2 + K[0, 2]
    pv = K[1, 1] * Y1 / Y2 + K[1, 2]
    if pu < 0 or pv < 0 or pu >= w - 1 or pv >= h - 1:
      costs[f] = 32.0
      continue
    iu, iv = int(pu), int(pv)
    fu, fv = pu - iu, pv - iv
    costs[f] = ((1 - fu) * (1 - fv) * popcount(c0 ^ codes[f, iv, iu]) + fu * (1 - fv) * popcount(c0 ^ codes[f, iv, iu + 1]) +
                (1 - fu) * fv * popcount(c0 ^ codes[f, iv + 1, iu]) + fu * fv * popcount(c0 ^ codes[f, iv + 1, iu + 1]))
  costs.sort()
  return np.mean(costs[:min(best_k, F)])


@njit(cache=True, nogil=True)
def crf_energy(code0, codes, K, Kinv, R, t, rho, u, v, r, best_k, lam, step, trunc):
  '''the pixel's local CRF energy for label r: unary + truncated linear pairwise to its 4 neighbours'''
  h, w = rho.shape
  e = hash_unary(code0, codes, K, Kinv, R, t, u, v, r, best_k)
  for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    y, x = v + dy, u + dx
    if 0 <= y < h and 0 <= x < w:
      e += lam * min(abs(r - rho[y, x]) / step, trunc)
  return e


@njit(cache=True, parallel=True)
def crf_passes(code0, codes, K, Kinv, R, t, rho, lo, hi, best_k, lam, step, trunc, passes, seed):
  '''
  DfM's hybrid PatchMatch / HashMatch inference: even passes go down the rows, each pixel trying the labels of the
  three pixels above it, odd passes go up (the three below); pixels of a row are independent (parallel). Plus a
  random perturbation per pixel (shrinking with the pass)
  '''
  h, w = rho.shape
  for p in range(passes):
    down = p % 2 == 0
    spread = (hi - lo) * 0.1 * 0.5 ** p
    for i in range(h):
      v = i if down else h - 1 - i
      dy = -1 if down else 1
      for u in prange(w):
        np.random.seed(seed + p * 1000003 + v * 1009 + u)
        best_r = rho[v, u]
        best_e = crf_energy(code0, codes, K, Kinv, R, t, rho, u, v, best_r, best_k, lam, step, trunc)
        y = v + dy
        for dx in (-1, 0, 1):
          x = u + dx
          if 0 <= y < h and 0 <= x < w:
            r = rho[y, x]
            e = crf_energy(code0, codes, K, Kinv, R, t, rho, u, v, r, best_k, lam, step, trunc)
            if e < best_e:
              best_e, best_r = e, r
        r = min(max(best_r + np.random.normal() * spread, lo), hi)
        e = crf_energy(code0, codes, K, Kinv, R, t, rho, u, v, r, best_k, lam, step, trunc)
        if e < best_e:
          best_e, best_r = e, r
        rho[v, u] = best_r


@njit(cache=True, parallel=True)
def crf_init(code0, codes, K, Kinv, R, t, lo, hi, best_k, particles, seed):
  '''the best of a few random inverse depths per pixel (unary only)'''
  h, w = code0.shape
  rho = np.empty((h, w))
  for v in prange(h):
    np.random.seed(seed + v)
    for u in range(w):
      best_e, best_r = 1e9, lo
      for _ in range(particles):
        r = lo + np.random.random() * (hi - lo)
        e = hash_unary(code0, codes, K, Kinv, R, t, u, v, r, best_k)
        if e < best_e:
          best_e, best_r = e, r
      rho[v, u] = best_r
  return rho


@njit(cache=True, parallel=True)
def crf_costs(code0, codes, K, Kinv, R, t, rho, best_k, lam, step, trunc):
  h, w = rho.shape
  L = np.empty((h, w))
  for v in prange(h):
    for u in range(w):
      L[v, u] = crf_energy(code0, codes, K, Kinv, R, t, rho, u, v, rho[v, u], best_k, lam, step, trunc)
  return L


class BilateralGrid:
  '''
  the bilateral grid of a reference image (Barron & Poole 2016): trilinear splat / slice S, the [1, 2, 1] blur B,
  bistochastization (m, n); solve() is the bilateral solver
  '''

  def __init__(self, ref, sigma_xy=16.0, sigma_r=16.0):
    import scipy.sparse as sp
    h, w = ref.shape
    vv, uu = np.mgrid[0:h, 0:w]
    g = np.stack([vv.ravel() / sigma_xy, uu.ravel() / sigma_xy, ref.ravel().astype(np.float64) / sigma_r], 1)
    base = np.floor(g).astype(np.int64)
    frac = g - base
    dims = base.max(0) + 2
    rows, cols, vals = [], [], []
    N = h * w
    for corner in range(8):
      o = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
      idx = ((base[:, 0] + o[0]) * dims[1] + base[:, 1] + o[1]) * dims[2] + base[:, 2] + o[2]
      wt = np.prod(np.where(o, frac, 1 - frac), axis=1)
      rows.append(idx)
      cols.append(np.arange(N))
      vals.append(wt)
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    nz = vals > 1e-9
    rows, cols, vals = rows[nz], cols[nz], vals[nz]
    used, rows = np.unique(rows, return_inverse=True)
    V = len(used)
    self.S = sp.csr_matrix((vals, (rows, cols)), shape=(V, N))
    # blur: neighbours along each grid axis (among the used vertices)
    lookup = {int(k): i for i, k in enumerate(used)}
    strides = [dims[1] * dims[2], dims[2], 1]
    br, bc = [], []
    for s in strides:
      for i, k in enumerate(used):
        j = lookup.get(int(k) + s)
        if j is not None:
          br += [i, j]
          bc += [j, i]
    self.B = sp.csr_matrix((np.ones(len(br)), (br, bc)), shape=(V, V)) + 6 * sp.eye(V)
    m = np.asarray(self.S.sum(1)).ravel()
    n = np.ones(V)
    for _ in range(20):
      n = np.sqrt(n * m / (self.B @ n))
    self.m, self.n, self.shape = m, n, (h, w)

  def solver(self, c, lam=4.0):
    '''the bilateral solver for confidence c (a linear filter): returns t -> argmin of eq. 7 of DfM'''
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    Sc = self.S @ c.ravel()
    A = lam * (sp.diags(self.m) - sp.diags(self.n) @ self.B @ sp.diags(self.n)) + sp.diags(Sc) + 1e-9 * sp.eye(len(Sc))
    lu = spla.splu(A.tocsc())
    def f(tgt):
      return (self.S.T @ lu.solve(self.S @ (c * tgt).ravel())).reshape(self.shape)
    return f


def planar_filter(Z, filt, eps=1e-3, scale=16.0):
  '''
  DfM's planar filter (Algorithm 1): a plane per pixel fitted by moving least squares, the weights given by a linear
  filter (the bilateral solver); returns the plane's value at the pixel. On inverse depth, planes are exact
  '''
  h, w = Z.shape
  vv, uu = np.mgrid[0:h, 0:w]
  x, y = uu / scale, vv / scale
  Z0 = np.nan_to_num(Z)
  F1, Fx, Fy, Fz, Fxx, Fxy, Fxz, Fyy, Fyz = (filt(a) for a in (np.ones_like(Z0), x, y, Z0, x * x, x * y, x * Z0, y * y, y * Z0))
  A = np.empty((h, w, 3, 3))
  A[..., 0, 0] = F1 * y * y - 2 * y * Fy + Fyy + eps ** 2
  A[..., 0, 1] = A[..., 1, 0] = F1 * y * x - x * Fy - y * Fx + Fxy
  A[..., 1, 1] = F1 * x * x - 2 * x * Fx + Fxx + eps ** 2
  A[..., 0, 2] = A[..., 2, 0] = F1 * y - Fy
  A[..., 1, 2] = A[..., 2, 1] = F1 * x - Fx
  A[..., 2, 2] = F1
  b = np.stack([Fz * y - Fyz, Fz * x - Fxz, Fz], -1)
  # (unknowns (Zy, Zx, Zz) of Z_j ~ Zz + Zy (y - y_j) + Zx (x - x_j); no Tikhonov term on the offset Zz)
  sol = np.linalg.solve(A.reshape(-1, 3, 3), b.reshape(-1, 3, 1)).reshape(h, w, 3)
  return sol[..., 2], F1


def google(rc, k, views=6, best_k=3, particles=5, passes=4, lam=1.0, levels=64, trunc=4.0, scale=2, max_cost=None,
           speckle=0.002, densify=True, sigma_xy=16.0, sigma_r=16.0, solver_lam=4.0):
  '''
  Depth from Motion on keyframe k: hashed patches matched at half resolution by the CRF passes, invalidated by the
  local CRF cost (default: the median plus a margin) and small connected components, then densified at full
  resolution by the planar bilateral solver (confidence 1 at the valid pixels)
  '''
  img = rc.image(k)
  H, W = img.shape
  nbrs = rc.covisible(k, views)
  if len(nbrs) == 0:
    return np.full((H, W), np.nan), np.full((H, W), np.nan)
  if getattr(rc, 'hash', None) is None:
    rc.hash = train_hash([rc.image(j) for j in range(0, len(rc), max(1, len(rc) // 20))])
  h, w = H // scale, W // scale
  small = lambda im: cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
  code0 = hash_codes(small(img), rc.hash)
  codes = np.array([hash_codes(small(rc.image(j)), rc.hash) for j in nbrs])
  Ks = rc.K.copy()
  Ks[:2] /= scale
  Ks[:2, 2] = (rc.K[:2, 2] + 0.5) / scale - 0.5
  Kinv = np.linalg.inv(Ks)
  rel = np.array([rc.T[j] @ np.linalg.inv(rc.T[k]) for j in nbrs])
  R, t = rel[:, :3, :3].copy(), rel[:, :3, 3].copy()
  lo, hi = sparse_range(rc, k)
  step = (hi - lo) / levels
  rho = crf_init(code0, codes, Ks, Kinv, R, t, lo, hi, best_k, particles, k)
  crf_passes(code0, codes, Ks, Kinv, R, t, rho, lo, hi, best_k, lam, step, trunc, passes, 7919 * k)
  L = crf_costs(code0, codes, Ks, Kinv, R, t, rho, best_k, lam, step, trunc)
  thr = np.median(L) if max_cost is None else max_cost
  ok = L < thr
  q = np.where(ok, np.round((rho - lo) / step * 4), -1).astype(np.int16)        # (quarter levels)
  cv2.filterSpeckles(q, -1, int(speckle * h * w), 4)
  ok &= q >= 0
  rho = np.where(ok, rho, np.nan)
  rho = cv2.resize(rho, (W, H), interpolation=cv2.INTER_NEAREST)
  if densify:
    valid = np.isfinite(rho)
    grid = BilateralGrid(img, sigma_xy, sigma_r)
    filt = grid.solver(valid.astype(np.float64), solver_lam)
    rho, _ = planar_filter(np.where(valid, rho, 0), filt)
    rho = np.where((rho >= lo) & (rho <= hi), rho, np.nan)
  sig = 0.05 * rho
  return rho, sig


def densified(rc, k, engine='combined', **kw):
  '''an engine's map densified by the planar bilateral solver (the confident pixels: sigma below the median)'''
  rho, sig = ENGINES[engine](rc, k)
  img = rc.image(k)
  valid = np.isfinite(rho) & (rho > 0)
  grid = BilateralGrid(img, kw.get('sigma_xy', 16.0), kw.get('sigma_r', 16.0))
  filt = grid.solver(valid.astype(np.float64), kw.get('solver_lam', 4.0))
  out, _ = planar_filter(np.where(valid, rho, 0), filt)
  lo, hi = sparse_range(rc, k)
  out = np.where((out >= lo) & (out <= hi), out, np.nan)
  return out, 0.05 * out


ENGINES = {'sparse': sparse, 'dis': dis, 'multi': multi_baseline, 'combined': combined, 'sequential': sequential, 'refined': refined, 'patchmatch': patchmatch, 'google': google, 'densified': densified}


# ------------------------------------------------------------------------------------------------- evaluation

def keyframe_scale(rc, k, gt):
  '''metres per map unit for keyframe k: median of ground truth / map depth over its sparse points'''
  uv, pts = rc.observations(k)
  z = transform(rc.T[k], rc.X[pts])[:, 2]
  ui, vi = np.round(uv).astype(int).T
  h, w = gt.shape
  ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
  g = gt[vi[ok], ui[ok]]
  ok2 = g > 0
  return float(np.median(g[ok2] / z[ok][ok2])) if ok2.sum() >= 5 else np.nan


def evaluate_keyframe(rho, gt, scale):
  valid = gt > 0
  est = valid & np.isfinite(rho) & (rho > 0)
  rho_gt = 1 / gt[est]
  err = np.abs(rho[est] / scale - rho_gt) / rho_gt      # rho in map units -> 1/m: divide by metres per unit
  n = max(valid.sum(), 1)
  return {'completeness': est.sum() / n, 'acc10': (err < 0.1).sum() / n, 're': float(np.mean(err)) if est.any() else np.nan,
          'precision': float(np.mean(err < 0.1)) if est.any() else np.nan,
          'outliers': float(np.mean(err > 0.25)) if est.any() else np.nan}


# ------------------------------------------------------------------------------------------------- filters

def discontinuities(rho, sig, k_sigma=3.0):
  '''drop pixels that differ from the median of their 3x3 neighbourhood by more than k sigma (flying pixels at depth
  edges, where the flow mixes foreground and background)'''
  r = np.where(np.isfinite(rho), rho, 0).astype(np.float32)
  med = cv2.medianBlur(r, 3)
  bad = np.abs(rho - med) > k_sigma * sig
  return np.where(bad, np.nan, rho), np.where(bad, np.nan, sig)


def consistency(rc, maps, k, neighbours=6, min_agree=2, chi2=3.84):
  '''
  inter-keyframe check (Mur-Artal & Tardos 2015): a pixel's point is projected into the covisible keyframes; it stays
  if at least min_agree of them have a compatible inverse depth (chi2, among the 4 pixels around the projection)
  '''
  rho, sig = maps[k]
  H, W = rho.shape
  vv, uu = np.nonzero(np.isfinite(rho))
  r = rho[vv, uu]
  x = np.c_[(uu - rc.K[0, 2]) / rc.K[0, 0], (vv - rc.K[1, 2]) / rc.K[1, 1], np.ones(len(uu))] / r[:, None]
  agree = np.zeros(len(uu), np.int32)
  for j in rc.covisible(k, neighbours):
    if j not in maps:
      continue
    rj, sj = maps[j]
    T = rc.T[j] @ np.linalg.inv(rc.T[k])
    Xj = x @ T[:3, :3].T + T[:3, 3]
    z = Xj[:, 2]
    pu = rc.K[0, 0] * Xj[:, 0] / np.maximum(z, 1e-9) + rc.K[0, 2]
    pv = rc.K[1, 1] * Xj[:, 1] / np.maximum(z, 1e-9) + rc.K[1, 2]
    ok_any = np.zeros(len(uu), bool)
    for du in (0, 1):
      for dv in (0, 1):
        iu, iv = np.floor(pu).astype(int) + du, np.floor(pv).astype(int) + dv
        inside = (z > 0) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        iu, iv = np.clip(iu, 0, W - 1), np.clip(iv, 0, H - 1)
        rn, sn = rj[iv, iu], sj[iv, iu]
        ok_any |= inside & np.isfinite(rn) & ((1 / np.maximum(z, 1e-9) - rn) ** 2 / np.maximum(sn, 1e-12) ** 2 < chi2)
    agree += ok_any
  keep = agree >= min_agree
  out, so = np.full_like(rho, np.nan), np.full_like(sig, np.nan)
  out[vv[keep], uu[keep]] = rho[vv[keep], uu[keep]]
  so[vv[keep], uu[keep]] = sig[vv[keep], uu[keep]]
  return out, so


# ------------------------------------------------------------------------------------------------- fusion

@njit(cache=True, parallel=True)
def tsdf_integrate(tsdf, weight, color, origin, voxel, K, T_cw, rho, sig, img, trunc, w_max, mode):
  '''
  one keyframe's inverse-depth map into the TSDF (voxel-centric, nearest pixel): signed distance along the optical axis,
  truncated at trunc, weighted by 1 / sigma_depth^2 (Sigma-Fusion); free space in front carves (sdf = +trunc)
  '''
  nx, ny, nz = tsdf.shape
  h, w = rho.shape
  for i in prange(nx):
    for j in range(ny):
      for k in range(nz):
        X = origin[0] + (i + 0.5) * voxel
        Y = origin[1] + (j + 0.5) * voxel
        Z = origin[2] + (k + 0.5) * voxel
        xc = T_cw[0, 0] * X + T_cw[0, 1] * Y + T_cw[0, 2] * Z + T_cw[0, 3]
        yc = T_cw[1, 0] * X + T_cw[1, 1] * Y + T_cw[1, 2] * Z + T_cw[1, 3]
        zc = T_cw[2, 0] * X + T_cw[2, 1] * Y + T_cw[2, 2] * Z + T_cw[2, 3]
        if zc <= 1e-6:
          continue
        u = int(K[0, 0] * xc / zc + K[0, 2] + 0.5)
        v = int(K[1, 1] * yc / zc + K[1, 2] + 0.5)
        if u < 0 or v < 0 or u >= w or v >= h:
          continue
        r = rho[v, u]
        if not (r > 0):
          continue
        d = 1.0 / r
        sd = sig[v, u] / (r * r)
        sdf = d - zc
        if sdf < -trunc:
          continue
        if sdf > trunc:
          sdf = trunc
        if mode == 0:
          wt = 1.0
        elif mode == 1:
          wt = 1.0 / max(sd, 1e-12)
        else:
          wt = 1.0 / max(sd * sd, 1e-12)
        w0 = weight[i, j, k]
        tsdf[i, j, k] = (w0 * tsdf[i, j, k] + wt * sdf / trunc) / (w0 + wt)
        if abs(sdf) < trunc:
          for ch in range(3):
            color[i, j, k, ch] = (w0 * color[i, j, k, ch] + wt * img[v, u, ch]) / (w0 + wt)
        weight[i, j, k] = min(w0 + wt, w_max)


def fuse(rc, maps, voxel_rel=0.01, trunc_voxels=4, w_max=1e12, upto=None, mode=2, max_voxels=120_000_000):
  '''TSDF over the extent of the depth maps; voxel = voxel_rel x the median keyframe depth (map units)'''
  depth = np.median([1 / np.nanmedian(r) for r, _ in maps.values() if np.isfinite(r).any()])
  voxel = voxel_rel * depth
  pts = np.concatenate([backproject(rc.K, rc.T[k], v[0], 16)[0] for k, v in maps.items()])
  lo, hi = np.percentile(pts, 0.5, axis=0) - 5 * voxel, np.percentile(pts, 99.5, axis=0) + 5 * voxel
  shape = np.ceil((hi - lo) / voxel).astype(int)
  if np.prod(shape) > max_voxels:     # (memory: coarsen)
    voxel *= (np.prod(shape) / max_voxels) ** (1 / 3)
    shape = np.ceil((hi - lo) / voxel).astype(int)
  tsdf = np.ones(shape, np.float32)
  weight = np.zeros(shape, np.float32)
  color = np.zeros(tuple(shape) + (3,), np.float32)
  for k, (rho, sig) in maps.items():
    if upto is not None and k > upto:
      continue
    tsdf_integrate(tsdf, weight, color, lo, voxel, rc.K, rc.T[k], np.nan_to_num(rho, nan=-1.0), np.nan_to_num(sig, nan=1.0),
                   rc.color(k).astype(np.float32), trunc_voxels * voxel, w_max, mode)
  return tsdf, weight, lo, voxel, color


def near_surface(points, tsdf, weight, origin, voxel, min_weight, band=0.5):
  '''points in voxels near a confident fused surface: enough weight, |tsdf| < band (of the truncation)'''
  idx = np.floor((points - origin) / voxel).astype(int)
  inside = np.all((idx >= 0) & (idx < np.array(tsdf.shape)), axis=1)
  keep = np.zeros(len(points), bool)
  i = idx[inside]
  keep[inside] = (weight[i[:, 0], i[:, 1], i[:, 2]] >= min_weight) & (np.abs(tsdf[i[:, 0], i[:, 1], i[:, 2]]) < band)
  return keep


def certain(weight, voxel, max_sigma_voxels=1.0):
  '''voxels whose fused depth is known within max_sigma_voxels (1/sigma^2 weights: sigma = 1/sqrt(W))'''
  return weight >= 1 / (max_sigma_voxels * voxel) ** 2


def mesh(tsdf, weight, origin, voxel, color, min_weight):
  '''marching cubes on the certain voxels; vertices in the map frame, a colour per vertex'''
  from skimage.measure import marching_cubes
  vol = np.where(weight >= min_weight, tsdf, np.nan)
  mask = weight >= min_weight
  verts, faces, normals, _ = marching_cubes(np.nan_to_num(vol, nan=1.0), 0.0, mask=mask)
  idx = np.clip(np.round(verts).astype(int), 0, np.array(tsdf.shape) - 1)
  return origin + (verts + 0.5) * voxel, faces, color[idx[:, 0], idx[:, 1], idx[:, 2]]


def surface_points(tsdf, weight, origin, voxel, min_weight):
  '''zero crossings of the TSDF between neighbouring voxels (both with enough weight), linearly interpolated'''
  pts = []
  for a in range(3):
    s0 = [slice(None)] * 3
    s1 = [slice(None)] * 3
    s0[a], s1[a] = slice(0, -1), slice(1, None)
    t0, t1 = tsdf[tuple(s0)], tsdf[tuple(s1)]
    w0, w1 = weight[tuple(s0)], weight[tuple(s1)]
    cross = (np.sign(t0) != np.sign(t1)) & (w0 >= min_weight) & (w1 >= min_weight) & (np.abs(t0 - t1) < 1.5)
    idx = np.argwhere(cross).astype(np.float64)
    f = t0[cross] / (t0[cross] - t1[cross])
    idx[:, a] += f
    pts.append(origin + (idx + 0.5) * voxel)
  return np.concatenate(pts)


def gt_cloud(rc, step=4):
  '''the ground-truth depth of every keyframe in the map frame (map units, the keyframe's scale)'''
  out = []
  for k in range(len(rc)):
    gt = rc.data.depth(rc.frame[k])
    if gt is None:
      continue
    sc = keyframe_scale(rc, k, gt)
    if not np.isfinite(sc):
      continue
    X, _ = backproject(rc.K, rc.T[k], np.where(gt > 0, sc / np.maximum(gt, 1e-6), np.nan), step)
    out.append(X)
  return np.concatenate(out)


def evaluate_cloud(points, gt_points, metres_per_unit, thresholds=(0.02, 0.05)):
  '''accuracy (reconstruction -> ground truth) and completeness (ground truth -> reconstruction), in metres'''
  import scipy.spatial
  d_acc = scipy.spatial.cKDTree(gt_points).query(points)[0] * metres_per_unit
  d_com = scipy.spatial.cKDTree(points).query(gt_points)[0] * metres_per_unit
  res = {'points': len(points), 'acc_median_cm': 100 * float(np.median(d_acc)), 'acc_mean_cm': 100 * float(np.mean(np.minimum(d_acc, 0.5)))}
  for t in thresholds:
    res[f'prec_{int(100 * t)}cm'] = float(np.mean(d_acc < t))
    res[f'compl_{int(100 * t)}cm'] = float(np.mean(d_com < t))
  return res


def online_maps(rc, engine, delay=32, cache=None):
  '''every keyframe's map as it could be computed delay frames after it: only keyframes / frames up to then'''
  if cache is not None and Path(cache).exists():
    d = np.load(cache)
    return {int(k): (d[f'rho{k}'], d[f'sig{k}']) for k in d['keys']}
  maps = {}
  for k in range(len(rc)):
    rc.limit = int(rc.frame[k]) + delay
    r, s = ENGINES[engine](rc, k)
    maps[k] = (r, s * calibrate(rc, k, r, s))
  rc.limit = None
  if cache is not None:
    np.savez_compressed(cache, keys=np.array(list(maps)), **{f'rho{k}': v[0].astype(np.float32) for k, v in maps.items()},
                        **{f'sig{k}': v[1].astype(np.float32) for k, v in maps.items()})
  return maps


def dense_points(rc, maps, tsdf, weight, origin, voxel, step=2, max_sigma_voxels=1.0, band=0.5):
  '''the keyframes' raw points near a confident fused surface (and their colours)'''
  P, G = [], []
  for k, (rho, _) in maps.items():
    X, g = backproject(rc.K, rc.T[k], rho, step, rc.color(k))
    keep = near_surface(X, tsdf, weight, origin, voxel, 1 / (max_sigma_voxels * voxel) ** 2, band)
    P.append(X[keep])
    G.append(g[keep])
  return np.concatenate(P), np.concatenate(G)


def surface_colors(points, color, origin, voxel):
  idx = np.clip(np.floor((points - origin) / voxel).astype(int), 0, np.array(color.shape[:3]) - 1)
  return color[idx[:, 0], idx[:, 1], idx[:, 2]]


def voxelize(P, C, size, min_points=3):
  '''the cells of a size grid (display frame, metres) holding at least min_points points, and their mean colours'''
  key = np.floor(P / size).astype(np.int64)
  cells, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  col = np.zeros((len(cells), 3))
  np.add.at(col, inv.ravel(), C)
  col /= cnt[:, None]
  keep = cnt >= min_points
  return cells[keep], col[keep]


def fill_floor(cells, colors, close=4):
  '''
  the floor layer (the most populated height in the lower part, +-1 cell) closed in 2D with a disk of radius close
  cells: holes in the untextured floor filled with the colour of the nearest floor cell
  '''
  import scipy.ndimage as ndi
  z = cells[:, 2]
  hist = np.bincount(z - z.min())
  low = max(1, len(hist) // 3)
  floor = z.min() + int(np.argmax(hist[:low]))
  on = np.abs(z - floor) <= 1
  xy = cells[on, :2]
  o = xy.min(0) - close
  grid = np.zeros(tuple(xy.max(0) - o + close + 1), bool)
  grid[xy[:, 0] - o[0], xy[:, 1] - o[1]] = True
  cg = np.zeros(grid.shape + (3,))
  cg[xy[:, 0] - o[0], xy[:, 1] - o[1]] = colors[on]
  yy, xx = np.mgrid[-close:close + 1, -close:close + 1]
  closed = ndi.binary_closing(grid, structure=yy ** 2 + xx ** 2 <= close ** 2)
  new = closed & ~grid
  _, (iy, ix) = ndi.distance_transform_edt(~grid, return_indices=True)
  ny, nx = np.nonzero(new)
  add = np.c_[ny + o[0], nx + o[1], np.full(len(ny), floor)]
  return np.r_[cells, add], np.r_[colors, cg[iy[ny, nx], ix[ny, nx]]], floor


def voxel_map(P, C, size=0.04, min_points=3, floor_close=4):
  '''
  a Matic-style voxel map of display-frame points (metres, +Z up): size cubes holding min_points points, coloured by
  their mean colour; the floor layer's holes closed. Returns cube centres and colours
  '''
  cells, cols = voxelize(P, C.astype(np.float64), size, min_points)
  if floor_close:
    cells, cols, _ = fill_floor(cells, cols, floor_close)
  return (cells + 0.5) * size, cols


def timelapse(rc, engines, path, fuse_every=10, delay=32, depth_step=6, dense_step=6, with_mesh=False, cube=0.0):
  '''
  rerun recording on a frame timeline, in metres with +Z up (the trajectory aligned to the ground truth): the ground
  truth path, the SLAM path and camera moving; per engine (name -> (refined maps, online maps); side by side views when
  several) every keyframe's depth as it could be computed delay frames after it (online), fused into its TSDF as it
  arrives; the fused surface points (optionally the mesh), the dense points (raw points near the fused surface) and
  optionally the voxel map (cube metres, floor holes closed) refreshed every fuse_every keyframes; at the end
  everything refined
  '''
  import rerun as rr
  import rerun.blueprint as rrb
  rr.init('nanoslam-recon-timelapse')
  rr.save(path)
  names = list(engines)
  if len(names) > 1:
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*[rrb.Spatial3DView(
      name=n, origin='/world', contents=['+ /world/**'] + [f'- /world/{m}/**' for m in names if m != n]) for n in names]),
      collapse_panels=True))
  rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  disp, R_disp, R_up = rc.display_frame()
  G = rc.data.gt
  okg = np.isfinite(G[:, 0, 3])
  rr.log('world/gt_path', rr.LineStrips3D([G[okg, :3, 3] @ R_up.T], radii=0.004, colors=[150, 150, 150]), static=True)
  def log_model(name, tsdf, weight, origin, voxel, color, maps):
    S = surface_points(tsdf, weight, origin, voxel, 1 / (0.5 * voxel) ** 2)
    rr.log(f'world/{name}/fused_surface', rr.Points3D(disp(S), colors=rgb(surface_colors(S, color, origin, voxel)), radii=0.004))
    if with_mesh:
      V, Fc, c = mesh(tsdf, weight, origin, voxel, color, 1 / (0.5 * voxel) ** 2)
      rr.log(f'world/{name}/fused_mesh', rr.Mesh3D(vertex_positions=disp(V), triangle_indices=Fc, vertex_colors=rgb(c)))
    P, C = dense_points(rc, maps, tsdf, weight, origin, voxel, step=dense_step)
    rr.log(f'world/{name}/dense', rr.Points3D(disp(P), colors=rgb(C), radii=0.002))
    if cube:
      X, c = voxel_map(np.r_[disp(S), disp(P)], np.r_[surface_colors(S, color, origin, voxel), C], cube)
      rr.log(f'world/{name}/voxels', rr.Boxes3D(centers=X, half_sizes=np.full((len(X), 3), cube / 2), colors=rgb(c), fill_mode='solid'))
  state = {}
  for n, (refined, online) in engines.items():
    t_ref = fuse(rc, refined, upto=-1)
    state[n] = [np.ones_like(t_ref[0]), np.zeros_like(t_ref[1]), t_ref[2], t_ref[3], np.zeros_like(t_ref[4]), []]
  img_h, img_w = rc.image(0).shape
  centers = np.array([np.linalg.inv(T)[:3, 3] if np.isfinite(T[0, 0]) else np.full(3, np.nan) for T in rc.frame_T])
  avail = {k: int(rc.frame[k]) + delay for k in range(len(rc))}
  last_frame = max(avail.values())
  for f in range(0, last_frame + 2, 2):
    rr.set_time('frame', sequence=f)
    if f < len(rc.frame_T) and np.isfinite(rc.frame_T[f, 0, 0]):
      T_wc = np.linalg.inv(rc.frame_T[f])
      rr.log('world/camera', rr.Transform3D(translation=disp(T_wc[:3, 3][None])[0], mat3x3=R_disp @ T_wc[:3, :3]))
      rr.log('world/camera', rr.Pinhole(image_from_camera=rc.K, width=img_w, height=img_h, image_plane_distance=0.15))
      seen = centers[:f + 1]
      rr.log('world/slam_path', rr.LineStrips3D([disp(seen[np.isfinite(seen).all(axis=1)])], radii=0.004, colors=[255, 200, 0]))
    for n, (refined, online) in engines.items():
      tsdf, weight, origin, voxel, color, done = state[n]
      for k in [k for k in online if avail[k] <= f and k not in done]:
        done.append(k)
        rho, sig = online[k]
        img_c = rc.color(k)
        X, c = backproject(rc.K, rc.T[k], rho, depth_step, img_c)
        rr.log(f'world/{n}/keyframe_depth/{k}', rr.Points3D(disp(X), colors=rgb(c), radii=0.002))
        tsdf_integrate(tsdf, weight, color, origin, voxel, rc.K, rc.T[k], np.nan_to_num(rho, nan=-1.0),
                       np.nan_to_num(sig, nan=1.0), img_c.astype(np.float32), 4 * voxel, 1e12, 2)
        if len(done) % fuse_every == 0:
          log_model(n, tsdf, weight, origin, voxel, color, {j: online[j] for j in done})
  # the end: refined
  rr.set_time('frame', sequence=last_frame + 4)
  for n, (refined, online) in engines.items():
    for k, (rho, _) in refined.items():
      X, c = backproject(rc.K, rc.T[k], rho, depth_step, rc.color(k))
      rr.log(f'world/{n}/keyframe_depth/{k}', rr.Points3D(disp(X), colors=rgb(c), radii=0.002))
    log_model(n, *fuse(rc, refined), refined)


def compute_maps(rc, engine, cache=None):
  '''every keyframe's (rho, sigma), cached as npz'''
  if cache is not None and Path(cache).exists():
    d = np.load(cache)
    maps = {int(k): (d[f'rho{k}'], d[f'sig{k}']) for k in d['keys']}
    return {k: (r, s * calibrate(rc, k, r, s)) for k, (r, s) in maps.items()}, None
  maps, times = {}, []
  for k in range(len(rc)):
    t = time.perf_counter()
    maps[k] = ENGINES[engine](rc, k)
    times.append(time.perf_counter() - t)
  maps = {k: (r, s * calibrate(rc, k, r, s)) for k, (r, s) in maps.items()}     # (sigma made honest at the sparse points)
  if cache is not None:
    np.savez_compressed(cache, keys=np.array(list(maps)), **{f'rho{k}': v[0].astype(np.float32) for k, v in maps.items()},
                        **{f'sig{k}': v[1].astype(np.float32) for k, v in maps.items()})
  return maps, float(np.mean(times))


def backproject(K, T_cw, rho, step=2, img=None):
  '''world points (and image values) of an inverse-depth map, every step-th pixel'''
  h, w = rho.shape
  vv, uu = np.mgrid[0:h:step, 0:w:step]
  r = rho[vv, uu]
  ok = np.isfinite(r) & (r > 0)
  uu, vv, z = uu[ok], vv[ok], 1 / r[ok]
  Xc = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
  T_wc = np.linalg.inv(T_cw)
  X = Xc @ T_wc[:3, :3].T + T_wc[:3, 3]
  return X, (img[vv, uu] if img is not None else None)


def rgb(c):
  c = np.clip(c, 0, 255).astype(np.uint8)
  return np.repeat(c[:, None], 3, axis=1) if c.ndim == 1 else c


def log_rerun(rr, rc, k, rho, gt=None, scale=None):
  T_wc = np.linalg.inv(rc.T[k])
  img = rc.image(k)
  h, w = img.shape
  rr.log(f'world/keyframes/{k}', rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]), static=True)
  rr.log(f'world/keyframes/{k}', rr.Pinhole(image_from_camera=rc.K, width=w, height=h, image_plane_distance=0.05),
         static=True)
  X, g = backproject(rc.K, rc.T[k], rho, 2, img)
  rr.log(f'world/depth/{k}', rr.Points3D(X, colors=np.repeat(g[:, None], 3, axis=1), radii=0.0015), static=True)
  if gt is not None and scale is not None:
    Xg, _ = backproject(rc.K, rc.T[k], np.where(gt > 0, scale / np.maximum(gt, 1e-6), np.nan), 4)
    rr.log(f'world/ground_truth/{k}', rr.Points3D(Xg, colors=[60, 200, 60], radii=0.0015), static=True)


def run(run_dir, engine, every=5, save=None, rerun=None, filters=(), fused=None):
  rc = Recon(run_dir)
  maps, ms = compute_maps(rc, engine, Path(run_dir) / f'recon_{engine}.npz')
  if 'discontinuity' in filters:
    maps = {k: discontinuities(*v) for k, v in maps.items()}
  if 'consistency' in filters:
    maps = {k: consistency(rc, maps, k) for k in maps}
  rr = None
  if rerun is not None:
    import rerun as rr
    rr.init('nanoslam-recon', spawn=rerun == 'spawn')
    if rerun != 'spawn':
      rr.save(rerun)
    rr.log('world', rr.ViewCoordinates.RDF, static=True)
    rr.log('world/sparse', rr.Points3D(rc.X, colors=[255, 140, 0], radii=0.004), static=True)
  if fused is not None:
    tsdf, weight, origin, voxel, color = fuse(rc, maps)
    pts = surface_points(tsdf, weight, origin, voxel, fused if fused > 0 else 1 / (0.5 * voxel) ** 2)
    gtp = gt_cloud(rc)
    scales = [keyframe_scale(rc, k, rc.data.depth(rc.frame[k])) for k in range(0, len(rc), 5) if rc.data.depth(rc.frame[k]) is not None]
    mpu = float(np.nanmedian(scales))
    raw = np.concatenate([backproject(rc.K, rc.T[k], maps[k][0], 4)[0] for k in maps])
    print('  raw keyframe points:', {k: round(v, 3) for k, v in evaluate_cloud(raw, gtp, mpu).items()})
    print(f'  fused (voxel {voxel * mpu * 100:.1f} cm, min weight {fused:g}):', {k: round(v, 3) for k, v in evaluate_cloud(pts, gtp, mpu).items()})
    if rr is not None:
      rr.log('world/fused', rr.Points3D(pts, colors=[90, 160, 255], radii=0.002), static=True)
  rows = []
  for k in range(0, len(rc), every):
    gt = rc.data.depth(rc.frame[k])
    if gt is None:
      continue
    scale = keyframe_scale(rc, k, gt)
    if not np.isfinite(scale):
      continue
    rho, sig = maps[k]
    rows.append(evaluate_keyframe(rho, gt, scale))
    if rr is not None:
      log_rerun(rr, rc, k, rho, gt, scale)
    if save is not None and len(rows) == 1:
      visualize(rc, k, rho, gt, scale, save)
  res = {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]}
  return res | {'keyframes': len(rows)} | ({'ms_per_keyframe': 1e3 * ms} if ms else {})


def visualize(rc, k, rho, gt, scale, path):
  '''image | ground truth | estimate (inverse depth, same colours)'''
  img = cv2.cvtColor(rc.image(k), cv2.COLOR_GRAY2BGR)
  inv_gt = np.where(gt > 0, 1 / np.maximum(gt, 1e-6), np.nan)
  lo, hi = np.nanpercentile(inv_gt, [2, 98])
  def col(x):
    c = cv2.applyColorMap(np.clip(255 * (x - lo) / (hi - lo), 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    c[~np.isfinite(x)] = 0
    return c
  cv2.imwrite(str(path), np.hstack([img, col(np.nan_to_num(inv_gt, nan=np.nan)), col(rho / scale)]))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('runs', nargs='+')
  ap.add_argument('--engine', default='sparse', help=f'one of {sorted(ENGINES)}; comma separated for --timelapse (side by side)')
  ap.add_argument('--cube', type=float, default=0.0, help='--timelapse: voxel map with cubes of this size [m] (0: none)')
  ap.add_argument('--every', type=int, default=5, help='evaluate every n-th keyframe')
  ap.add_argument('--save', help='visualization of the first evaluated keyframe (directory)')
  ap.add_argument('--rerun', nargs='?', const='spawn', help='3D view in rerun (spawn a viewer, or save to the given .rrd)')
  ap.add_argument('--filters', default='', help='comma separated: discontinuity, consistency')
  ap.add_argument('--fuse', type=float, help='TSDF fusion; the minimum accumulated weight of the extracted surface')
  ap.add_argument('--timelapse', help='rerun recording (.rrd) of the reconstruction growing along the frames')
  args = ap.parse_args()
  for r in args.runs:
    save = None if args.save is None else Path(args.save) / f'{Path(r).name}_{args.engine}.jpg'
    if save is not None:
      save.parent.mkdir(parents=True, exist_ok=True)
    if args.timelapse:
      rc = Recon(r)
      engines = {}
      for e in args.engine.split(','):
        maps, _ = compute_maps(rc, e, Path(r) / f'recon_{e}.npz')
        engines[e] = (maps, online_maps(rc, e, cache=Path(r) / f'recon_{e}_online.npz'))
      timelapse(rc, engines, args.timelapse, cube=args.cube)
      continue
    res = run(r, args.engine, args.every, save, args.rerun, [f for f in args.filters.split(',') if f], args.fuse)
    print(f'{Path(r).name:32s} {args.engine:10s} ' + '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()))


if __name__ == '__main__':
  main()
