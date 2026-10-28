'''
nanoslam turbo dense: live dense mapping on top of the SLAM (CPU, no learned components, no ground truth).

Per keyframe, once `delay` frames after it have arrived (a worker thread, on a snapshot of the SLAM state):
  1. DIS optical flow to the most covisible keyframes, triangulated with the known poses (closed form in inverse depth),
     forward-backward and epipolar checks; the per-pixel hypotheses fused by a chi2 test (Mur-Artal & Tardos 2015).
  2. Multi-baseline plane sweep (Okutomi & Kanade): ZNCC costs over frames at several offsets before and after the
     keyframe plus the covisible keyframes, summed as a function of inverse depth, within +-3 sigma of DIS (the whole
     range from the keyframe's sparse points where DIS has nothing); uniqueness test, parabola refinement.
  3. Sigma calibrated at the keyframe's sparse map points (median |z| of a Gaussian there; never shrunk).
  4. Fused into a TSDF (voxel hashing: 8^3 blocks allocated around the observed surfaces, free space in front carved),
     each observation weighted by 1 / sigma_depth^2 (Sigma-Fusion), so the weight is the fused uncertainty.

Output: the fused surface (zero crossings with a fused sigma below half a voxel) and dense points (every keyframe's
raw points next to a confident fused surface); optionally a mesh (marching cubes) and a voxel map (cubes, floor holes
closed). Loop closures / map merges re-fuse the stored depth maps with the corrected poses (and the scale change, from
the keyframe's own map points). Up is estimated from the keyframe orientations (cameras are held without roll).
'''
import queue
import threading
import time
import concurrent.futures

import cv2
import numpy as np
from numba import njit, prange

from geometry import inv_pose, transform, camera_center
from map import MAPPED


class DenseConfig:
  keyframes = 6                     # covisible keyframes (DIS flow and the sweep)
  frames_before = (1, 2, 4, 8, 16, 32)    # frame offsets before / after the keyframe in the sweep (small baselines)
  frames_after = (1, 2, 4, 8, 16, 32)
  delay = 32                        # frames after a keyframe before its depth map is computed
  # DIS
  dis_preset = 2                    # cv2.DISOPTICAL_FLOW_PRESET_ (0 ultrafast, 1 fast, 2 medium)
  fb_px = 1.0                       # forward-backward consistency
  epi_px = 1.5                      # distance of the flow match from the epipolar line
  min_views = 2                     # keyframes agreeing on a pixel (chi2 test)
  # sweep
  window = 3.0                      # search window: DIS inverse depth +- window sigma
  samples = 32                      # inverse-depth samples (at most) ...
  step_px = 0.5                     # ... about this far apart along the longest epipolar segment [px, sweep resolution]
  patch = 2                         # ZNCC patch half size (at the sweep's resolution)
  scale = 2                         # the sweep runs at 1/scale resolution
  sweep_views = 3                   # frames that must see a pixel
  uniqueness = 0.8                  # best cost < uniqueness x the best away from it
  # fusion
  voxel = 0.01                      # voxel size relative to the median depth of the first keyframe
  truncation = 4                    # TSDF truncation [voxels]
  surface_sigma = 0.3               # fused sigma of the extracted surface [voxels]
  points_sigma = 1.0                # dense points: next to voxels with this fused sigma [voxels] ...
  points_band = 0.5                 # ... and |tsdf| below this (fraction of the truncation)
  refuse = ('loop', 'merge')        # SLAM events after which the stored maps are fused again with the new poses
  refine = False                    # at the end: recompute every keyframe's map with all frames and keyframes
  max_queue = 0                     # real time: skip a keyframe when this many maps are waiting (0: never skip)
  # ablation switches (all on by default)
  use_dis = True                    # off: the sweep searches the whole range everywhere
  use_sweep = True                  # off: DIS only
  chi2_test = True                  # off: DIS hypotheses averaged (inverse variance) without the compatibility test
  calibrate = True                  # off: raw sigmas
  sigma_weights = True              # off: every observation weighs 1 (KinectFusion), thresholds in observations:
  min_count = (3, 2)                # ... surface, dense points
  carve = True                      # off: free space in front of a surface does not update voxels
  threads = 0                       # worker threads for numba (0: all)
  # display
  log_every = 10                    # keyframes between rerun refreshes of the fused model
  log_depth = True                  # every keyframe's depth points when computed
  points_step = 6                   # pixel step of the logged points
  mesh = False                      # marching cubes of the fused surface
  cube = 0                          # voxel map: cube size [voxels] (0: off)


# ------------------------------------------------------------------------------------------------- depth estimation

@njit(cache=True, nogil=True)
def flow_depth(f, b, K, R, t, fb_px, epi_px, rho, sig):
  '''
  inverse depth (reference view) and sigma of every pixel's flow match q = p + f(p) in the other view, NaN where
  rejected: forward-backward error |f(p) + b(q)| (bilinear) >= fb_px, or q >= epi_px from the epipolar line. The
  projection q = proj(K (R x / rho + t)) is linear in rho after multiplying out (closed form); sigma = max(0.3 px,
  forward-backward error) / |dq / drho|
  '''
  h, w = rho.shape
  B0 = K[0, 0] * t[0] + K[0, 2] * t[2]
  B1 = K[1, 1] * t[1] + K[1, 2] * t[2]
  B2 = t[2]
  for v in range(h):
    for u in range(w):
      rho[v, u] = np.nan
      sig[v, u] = np.nan
      qx, qy = u + f[v, u, 0], v + f[v, u, 1]
      if not (qx >= 0 and qy >= 0 and qx < w - 1 and qy < h - 1):
        continue
      iu, iv = int(qx), int(qy)
      fu, fv = qx - iu, qy - iv
      bx = ((1 - fu) * (1 - fv) * b[iv, iu, 0] + fu * (1 - fv) * b[iv, iu + 1, 0] + (1 - fu) * fv * b[iv + 1, iu, 0] +
            fu * fv * b[iv + 1, iu + 1, 0])
      by = ((1 - fu) * (1 - fv) * b[iv, iu, 1] + fu * (1 - fv) * b[iv, iu + 1, 1] + (1 - fu) * fv * b[iv + 1, iu, 1] +
            fu * fv * b[iv + 1, iu + 1, 1])
      fb = np.sqrt((f[v, u, 0] + bx) ** 2 + (f[v, u, 1] + by) ** 2)
      if fb >= fb_px:
        continue
      x0 = (u - K[0, 2]) / K[0, 0]
      x1 = (v - K[1, 2]) / K[1, 1]
      r0 = R[0, 0] * x0 + R[0, 1] * x1 + R[0, 2]
      r1 = R[1, 0] * x0 + R[1, 1] * x1 + R[1, 2]
      r2 = R[2, 0] * x0 + R[2, 1] * x1 + R[2, 2]
      A0, A1, A2 = K[0, 0] * r0 + K[0, 2] * r2, K[1, 1] * r1 + K[1, 2] * r2, r2
      # q (A2 + rho B2) = A + rho B  ->  rho (q B2 - B) = A - q A2
      c0, c1 = qx * B2 - B0, qy * B2 - B1
      r = ((A0 - qx * A2) * c0 + (A1 - qy * A2) * c1) / max(c0 * c0 + c1 * c1, 1e-12)
      den = A2 + r * B2
      if not r > 0 or abs(den) < 1e-12:
        continue
      px, py = (A0 + r * B0) / den, (A1 + r * B1) / den
      if (px - qx) ** 2 + (py - qy) ** 2 >= epi_px * epi_px:
        continue
      dx = (B0 * den - (A0 + r * B0) * B2) / (den * den)
      dy = (B1 * den - (A1 + r * B1) * B2) / (den * den)
      rho[v, u] = r
      sig[v, u] = max(0.3, fb) / max(np.sqrt(dx * dx + dy * dy), 1e-12)


@njit(cache=True, parallel=True, nogil=True)
def fuse_hypotheses(rho, sig, min_views, chi2):
  '''
  per pixel, hypotheses (N, H, W) with NaN for none: the largest set compatible with one of them (seed) by a chi2 test,
  fused by inverse variance (Mur-Artal & Tardos 2015); NaN where fewer than min_views agree
  '''
  N, h, w = rho.shape
  out, out_sig = np.full((h, w), np.nan), np.full((h, w), np.nan)
  for v in prange(h):
    for u in range(w):
      best, seed = 0, -1
      for i in range(N):
        if not np.isfinite(rho[i, v, u]):
          continue
        n = 0
        for j in range(N):
          if np.isfinite(rho[j, v, u]) and (rho[i, v, u] - rho[j, v, u]) ** 2 < chi2 * (sig[i, v, u] ** 2 + sig[j, v, u] ** 2):
            n += 1
        if n > best:
          best, seed = n, i
      if best < min_views:
        continue
      sw, swr = 0.0, 0.0
      for j in range(N):
        if np.isfinite(rho[j, v, u]) and (rho[seed, v, u] - rho[j, v, u]) ** 2 < chi2 * (sig[seed, v, u] ** 2 + sig[j, v, u] ** 2):
          wj = 1.0 / sig[j, v, u] ** 2
          sw += wj
          swr += wj * rho[j, v, u]
      out[v, u] = swr / sw
      out_sig[v, u] = 1.0 / np.sqrt(sw)
  return out, out_sig


class Flows:
  '''DIS flow in a thread pool: one DIS instance per worker thread (cv2 releases the GIL)'''

  def __init__(self, preset, threads):
    self.preset = preset
    self.local = threading.local()
    self.pool = concurrent.futures.ThreadPoolExecutor(threads)

  def calc(self, a, b):
    if not hasattr(self.local, 'dis'):
      self.local.dis = cv2.DISOpticalFlow_create(self.preset)
    return self.local.dis.calc(a, b, None)

  def map(self, fn, items):
    return list(self.pool.map(fn, items))


def dis(c, K, img, T, keyframes, flows):
  '''DIS flow to the keyframes [(image, T_cw)] (in parallel), triangulated with the known poses, fused across them'''
  h, w = img.shape
  if not keyframes:
    return np.full((h, w), np.nan), np.full((h, w), np.nan)
  rho, sig = np.empty((len(keyframes), h, w)), np.empty((len(keyframes), h, w))

  def one(i):
    other, T_j = keyframes[i]
    rel = T_j @ inv_pose(T)
    flow_depth(flows.calc(img, other), flows.calc(other, img), K, rel[:3, :3].copy(), rel[:3, 3].copy(), c.fb_px,
               c.epi_px, rho[i], sig[i])
  flows.map(one, range(len(keyframes)))
  return fuse_hypotheses(rho, sig, c.min_views, 5.99 if c.chi2_test else np.inf)


@njit(cache=True, parallel=True, nogil=True, fastmath={'nsz', 'arcp', 'contract', 'afn', 'reassoc'})   # (no nnan / ninf: infs mark samples)
def plane_sweep(ref, others, K, R, t, lo, hi, S, half, step_px, min_views, uniqueness):
  '''
  per reference pixel, inverse-depth samples in [lo, hi] (about step_px apart along the longest epipolar segment, at
  most S), each scored by the mean over the other images that see it (at least min_views) of the truncated 1 - ZNCC
  of a (2 half + 1)^2 patch (fronto-parallel: sampled around the projected centre); the minimum (uniqueness: below
  uniqueness x the best one more than 2 samples away, not at the ends) refined by a parabola; sigma from its
  curvature. NaN: none
  '''
  h, w = ref.shape
  F = len(others)
  rho, sig = np.full((h, w), np.nan), np.full((h, w), np.nan)
  n = (2 * half + 1) ** 2
  for v in prange(half, h - half):
    a = np.empty(n)
    b = np.empty(n)
    cost = np.empty(S)
    count = np.empty(S, np.int32)
    for u in range(half, w - half):
      if not hi[v, u] > lo[v, u]:
        continue
      m = 0
      for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
          a[m] = ref[v + dv, u + du]
          m += 1
      a -= a.mean()
      na = np.sqrt(np.sum(a * a))
      if na < 1e-3:
        continue
      x0 = (u - K[0, 2]) / K[0, 0]
      x1 = (v - K[1, 2]) / K[1, 1]
      span = 0.0
      for f in range(F):
        A0 = K[0, 0] * (R[f, 0, 0] * x0 + R[f, 0, 1] * x1 + R[f, 0, 2]) + K[0, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A1 = K[1, 1] * (R[f, 1, 0] * x0 + R[f, 1, 1] * x1 + R[f, 1, 2]) + K[1, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A2 = R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2]
        B0 = K[0, 0] * t[f, 0] + K[0, 2] * t[f, 2]
        B1 = K[1, 1] * t[f, 1] + K[1, 2] * t[f, 2]
        B2 = t[f, 2]
        z0, z1 = A2 + lo[v, u] * B2, A2 + hi[v, u] * B2
        if z0 > 1e-6 and z1 > 1e-6:
          du = (A0 + lo[v, u] * B0) / z0 - (A0 + hi[v, u] * B0) / z1
          dv = (A1 + lo[v, u] * B1) / z0 - (A1 + hi[v, u] * B1) / z1
          span = max(span, np.sqrt(du * du + dv * dv))
      Sp = min(S, max(5, int(span / step_px) + 2))
      step = (hi[v, u] - lo[v, u]) / (Sp - 1)
      cost[:Sp] = 0.0
      count[:Sp] = 0
      for f in range(F):
        # p(rho) = K (R x + rho t) / z = (A + rho B) / (A2 + rho B2)
        A0 = K[0, 0] * (R[f, 0, 0] * x0 + R[f, 0, 1] * x1 + R[f, 0, 2]) + K[0, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A1 = K[1, 1] * (R[f, 1, 0] * x0 + R[f, 1, 1] * x1 + R[f, 1, 2]) + K[1, 2] * (R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2])
        A2 = R[f, 2, 0] * x0 + R[f, 2, 1] * x1 + R[f, 2, 2]
        B0 = K[0, 0] * t[f, 0] + K[0, 2] * t[f, 2]
        B1 = K[1, 1] * t[f, 1] + K[1, 2] * t[f, 2]
        B2 = t[f, 2]
        for s in range(Sp):
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
          cost[s] += 1.0 if nb < 1e-3 else min(1.0 - np.sum(a * b) / (na * nb), 1.0)
          count[s] += 1
      best, c0 = -1, np.inf
      for s in range(Sp):
        cost[s] = cost[s] / count[s] if count[s] >= min_views else np.inf
        if cost[s] < c0:
          best, c0 = s, cost[s]
      if best <= 0 or best >= Sp - 1:
        continue
      c1 = np.inf
      for s in range(Sp):
        if abs(s - best) > 2:
          c1 = min(c1, cost[s])
      cl, cr = cost[best - 1], cost[best + 1]
      if not (c0 < uniqueness * c1) or not (cl < np.inf and cr < np.inf):
        continue
      curv = cl + cr - 2 * c0
      if not curv > 1e-6:
        continue
      rho[v, u] = lo[v, u] + (best + 0.5 * (cl - cr) / curv) * step
      sig[v, u] = step / np.sqrt(curv) * np.sqrt(max(c0, 1e-3))
  return rho, sig


def sweep(c, K, img, T, views, lo, hi):
  '''
  multi-baseline stereo over the views [(image, T_cw)] in per-pixel inverse-depth ranges [lo, hi] (full-resolution
  arrays): small baselines give one wide valley, large ones a sharp minimum among false ones, their sum one sharp
  valley; at 1/scale resolution, returned at full resolution
  '''
  H, W = img.shape
  h, w = H // c.scale, W // c.scale
  if not views:
    return np.full((H, W), np.nan), np.full((H, W), np.nan)
  small = lambda im: cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
  Ks = K.copy()
  Ks[:2] /= c.scale
  Ks[:2, 2] = (K[:2, 2] + 0.5) / c.scale - 0.5
  rel = np.array([T_j @ inv_pose(T) for _, T_j in views])
  lo_s = np.nan_to_num(cv2.resize(lo, (w, h), interpolation=cv2.INTER_NEAREST), nan=1.0)
  hi_s = np.nan_to_num(cv2.resize(hi, (w, h), interpolation=cv2.INTER_NEAREST), nan=0.0)
  rho, sig = plane_sweep(small(img), np.array([small(im) for im, _ in views]), Ks, rel[:, :3, :3].copy(),
                         rel[:, :3, 3].copy(), lo_s, hi_s, c.samples, c.patch, c.step_px, c.sweep_views, c.uniqueness)
  return cv2.resize(rho, (W, H), interpolation=cv2.INTER_NEAREST), cv2.resize(sig, (W, H), interpolation=cv2.INTER_NEAREST)


def calibrate(rho, sig, uv, z):
  '''
  inflation of sigma so that the errors at the keyframe's sparse map points (pixels uv, depths z from BA: much more
  precise) have the median |z| of a Gaussian (0.6745); >= 1 (sigma is never shrunk)
  '''
  ui, vi = np.round(uv).astype(int).T
  h, w = rho.shape
  ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
  r, s = rho[vi[ok], ui[ok]], sig[vi[ok], ui[ok]]
  have = np.isfinite(r) & (r > 0) & np.isfinite(s) & (s > 0)
  if have.sum() < 10:
    return 1.0
  return max(1.0, float(np.median(np.abs(r[have] - 1 / z[ok][have]) / s[have])) / 0.6745)


def depth_map(c, K, job, flows):
  '''
  a keyframe's inverse depth and sigma (full resolution, NaN: none): DIS gives the wide valley (a prior: rho +- window
  sigma), the multi-baseline sweep pinpoints the minimum in it; where DIS has nothing the sweep searches the range of
  the keyframe's sparse points; inside a DIS window where the sweep finds nothing, DIS stands
  '''
  if c.use_dis:
    rho, sig = dis(c, K, job.image, job.T, job.keyframes, flows)
  else:
    rho = sig = np.full(job.image.shape, np.nan)
  if not c.use_sweep:
    return rho, sig * (calibrate(rho, sig, job.uv, job.z) if c.calibrate else 1.0)
  inv = 1 / job.z[job.z > 0]
  a, b = np.percentile(inv, 1) * 0.5, np.percentile(inv, 99) * 1.5
  have = np.isfinite(rho)
  half_width = np.maximum(c.window * sig, 0.05 * rho)
  lo = np.where(have, np.maximum(rho - half_width, a * 0.5), a)
  hi = np.where(have, rho + half_width, b)
  out, s = sweep(c, K, job.image, job.T, job.views, lo, hi)
  out = np.where(np.isfinite(out), out, rho)
  s = np.where(np.isfinite(s), s, sig)
  return out, s * (calibrate(out, s, job.uv, job.z) if c.calibrate else 1.0)


# ------------------------------------------------------------------------------------------------- TSDF (voxel hashing)
# blocks of B^3 voxels, allocated around observed surfaces; an open-addressing hash from block coordinates to rows of
# the block arrays (tsdf in units of the truncation, weight = sum of 1 / sigma^2, colour)

B = 8
OFFSET = 1 << 20


@njit(cache=True, nogil=True)
def block_key(x, y, z):
  return ((x + OFFSET) << 42) | ((y + OFFSET) << 21) | (z + OFFSET)


@njit(cache=True, nogil=True)
def slot_of(keys, key):
  '''the hash slot of key, or of the empty slot where it would go'''
  mask = len(keys) - 1
  h = key * 0x9E3779B1
  i = (h ^ (h >> 29)) & mask
  while keys[i] != -1 and keys[i] != key:
    i = (i + 1) & mask
  return i


@njit(cache=True, nogil=True)
def find_block(keys, rows, x, y, z):
  i = slot_of(keys, block_key(x, y, z))
  return rows[i] if keys[i] != -1 else -1


@njit(cache=True, nogil=True)
def allocate(keys, rows, coords, n, K, T_wc, rho, trunc, voxel):
  '''
  allocates the blocks along every pixel's ray within the truncation of its depth; returns the new block count, or -1
  when the hash or the block arrays need to grow first (allocation is idempotent: just call again)
  '''
  h, w = rho.shape
  size = B * voxel
  for v in range(h):
    for u in range(w):
      r = rho[v, u]
      if not r > 0:
        continue
      x = (u - K[0, 2]) / K[0, 0]
      y = (v - K[1, 2]) / K[1, 1]
      d = 1.0 / r
      steps = int(2 * trunc / (0.5 * size)) + 1
      for s in range(steps + 1):
        z = d - trunc + s * (2 * trunc / steps)
        if z <= 0:
          continue
        X = T_wc[0, 0] * x * z + T_wc[0, 1] * y * z + T_wc[0, 2] * z + T_wc[0, 3]
        Y = T_wc[1, 0] * x * z + T_wc[1, 1] * y * z + T_wc[1, 2] * z + T_wc[1, 3]
        Z = T_wc[2, 0] * x * z + T_wc[2, 1] * y * z + T_wc[2, 2] * z + T_wc[2, 3]
        bx, by, bz = int(np.floor(X / size)), int(np.floor(Y / size)), int(np.floor(Z / size))
        key = block_key(bx, by, bz)
        i = slot_of(keys, key)
        if keys[i] == -1:
          if n >= len(coords) or 2 * (n + 1) > len(keys):
            return -1
          keys[i] = key
          rows[i] = n
          coords[n, 0], coords[n, 1], coords[n, 2] = bx, by, bz
          n += 1
  return n


@njit(cache=True, parallel=True, nogil=True)
def integrate(coords, n, tsdf, weight, color, voxel, K, T_cw, rho, sig, img, trunc, sigma_weights, carve):
  '''
  voxel-centric update of the blocks in the view: signed distance along the optical axis (truncated; free space in
  front sets +1), weighted by 1 / sigma_depth^2, nearest pixel
  '''
  h, w = rho.shape
  size = B * voxel
  margin = 0.87 * size
  for b in prange(n):
    cx = (coords[b, 0] + 0.5) * size
    cy = (coords[b, 1] + 0.5) * size
    cz = (coords[b, 2] + 0.5) * size
    zc = T_cw[2, 0] * cx + T_cw[2, 1] * cy + T_cw[2, 2] * cz + T_cw[2, 3]
    if zc <= -margin:
      continue
    if zc > margin:
      xc = T_cw[0, 0] * cx + T_cw[0, 1] * cy + T_cw[0, 2] * cz + T_cw[0, 3]
      yc = T_cw[1, 0] * cx + T_cw[1, 1] * cy + T_cw[1, 2] * cz + T_cw[1, 3]
      pad = K[0, 0] * margin / (zc - margin) + 1
      pu = K[0, 0] * xc / zc + K[0, 2]
      pv = K[1, 1] * yc / zc + K[1, 2]
      if pu < -pad or pv < -pad or pu > w + pad or pv > h + pad:
        continue
    for i in range(B):
      for j in range(B):
        for k in range(B):
          X = (coords[b, 0] * B + i + 0.5) * voxel
          Y = (coords[b, 1] * B + j + 0.5) * voxel
          Z = (coords[b, 2] * B + k + 0.5) * voxel
          z = T_cw[2, 0] * X + T_cw[2, 1] * Y + T_cw[2, 2] * Z + T_cw[2, 3]
          if z <= 1e-6:
            continue
          u = int(K[0, 0] * (T_cw[0, 0] * X + T_cw[0, 1] * Y + T_cw[0, 2] * Z + T_cw[0, 3]) / z + K[0, 2] + 0.5)
          v = int(K[1, 1] * (T_cw[1, 0] * X + T_cw[1, 1] * Y + T_cw[1, 2] * Z + T_cw[1, 3]) / z + K[1, 2] + 0.5)
          if u < 0 or v < 0 or u >= w or v >= h:
            continue
          r = rho[v, u]
          if not r > 0:
            continue
          sdf = 1.0 / r - z
          if sdf < -trunc or (sdf > trunc and not carve):
            continue
          sd = sig[v, u] / (r * r)
          wt = 1.0 / max(sd * sd, 1e-12) if sigma_weights else 1.0
          w0 = weight[b, i, j, k]
          tsdf[b, i, j, k] = (w0 * tsdf[b, i, j, k] + wt * min(sdf, trunc) / trunc) / (w0 + wt)
          if sdf < trunc:
            for ch in range(3):
              color[b, i, j, k, ch] = np.uint8(min(255.0, (w0 * color[b, i, j, k, ch] + wt * img[v, u, ch]) / (w0 + wt) + 0.5))
          weight[b, i, j, k] = w0 + wt


@njit(cache=True, nogil=True)
def voxel_at(keys, rows, gx, gy, gz):
  '''(block row, i, j, k) of the global voxel, row -1 if not allocated'''
  bx, by, bz = gx // B, gy // B, gz // B
  return find_block(keys, rows, bx, by, bz), gx - bx * B, gy - by * B, gz - bz * B


@njit(cache=True, nogil=True)
def crossings(keys, rows, coords, n, tsdf, weight, color, voxel, min_weight, out, out_color):
  '''
  zero crossings of the TSDF between neighbouring voxels (both with enough weight), linearly interpolated, and the
  colour of the nearer voxel; out=None: count only
  '''
  m = 0
  for b in range(n):
    for i in range(B):
      for j in range(B):
        for k in range(B):
          w0 = weight[b, i, j, k]
          if w0 < min_weight:
            continue
          t0 = tsdf[b, i, j, k]
          g = (coords[b, 0] * B + i, coords[b, 1] * B + j, coords[b, 2] * B + k)
          for a in range(3):
            g1 = (g[0] + int(a == 0), g[1] + int(a == 1), g[2] + int(a == 2))
            r, i1, j1, k1 = voxel_at(keys, rows, g1[0], g1[1], g1[2])
            if r < 0 or weight[r, i1, j1, k1] < min_weight:
              continue
            t1 = tsdf[r, i1, j1, k1]
            if (t0 < 0) == (t1 < 0) or abs(t0 - t1) >= 1.5:
              continue
            if out is not None:
              f = t0 / (t0 - t1)
              out[m, 0] = (g[0] + 0.5 + f * int(a == 0)) * voxel
              out[m, 1] = (g[1] + 0.5 + f * int(a == 1)) * voxel
              out[m, 2] = (g[2] + 0.5 + f * int(a == 2)) * voxel
              for ch in range(3):
                out_color[m, ch] = color[b, i, j, k, ch] if f < 0.5 else color[r, i1, j1, k1, ch]
            m += 1
  return m


@njit(cache=True, parallel=True, nogil=True)
def near_surface(keys, rows, tsdf, weight, voxel, min_weight, band, points):
  '''points in voxels next to a confident fused surface: enough weight and |tsdf| < band'''
  keep = np.zeros(len(points), np.bool_)
  for p in prange(len(points)):
    r, i, j, k = voxel_at(keys, rows, int(np.floor(points[p, 0] / voxel)), int(np.floor(points[p, 1] / voxel)),
                          int(np.floor(points[p, 2] / voxel)))
    keep[p] = r >= 0 and weight[r, i, j, k] >= min_weight and abs(tsdf[r, i, j, k]) < band
  return keep


class TSDF:
  def __init__(self, voxel, truncation, sigma_weights=True, carve=True, min_count=(3, 2)):
    self.voxel, self.trunc = voxel, truncation * voxel
    self.sigma_weights, self.carve, self.min_count = sigma_weights, carve, min_count
    self.keys = np.full(1 << 16, -1, np.int64)
    self.rows = np.zeros(1 << 16, np.int64)
    self.coords = np.zeros((1 << 12, 3), np.int64)
    self.tsdf = np.ones((1 << 12, B, B, B), np.float32)
    self.weight = np.zeros((1 << 12, B, B, B), np.float32)
    self.color = np.zeros((1 << 12, B, B, B, 3), np.uint8)
    self.n = 0

  def grow(self):
    if 4 * self.n > len(self.keys):        # rehash into twice the slots
      keys, rows = np.full(2 * len(self.keys), -1, np.int64), np.zeros(2 * len(self.keys), np.int64)
      rehash(keys, rows, self.coords, self.n)
      self.keys, self.rows = keys, rows
    if 2 * self.n > len(self.coords):
      cap = 2 * len(self.coords)
      for name, fill in [('coords', 0), ('tsdf', 1), ('weight', 0), ('color', 0)]:
        a = getattr(self, name)
        new = np.full((cap,) + a.shape[1:], fill, a.dtype)
        new[:self.n] = a[:self.n]
        setattr(self, name, new)

  def integrate(self, K, T_cw, rho, sig, img):
    '''an inverse-depth map (NaN: none) with its sigma and RGB image (same size; K of that size)'''
    rho = np.nan_to_num(rho, nan=-1.0)
    T_wc = inv_pose(T_cw)
    while True:
      n = allocate(self.keys, self.rows, self.coords, self.n, K, T_wc, rho, self.trunc, self.voxel)
      if n >= 0:
        self.n = n
        break
      self.grow()
    self.grow()
    integrate(self.coords, self.n, self.tsdf, self.weight, self.color, self.voxel, K, T_cw, rho,
              np.nan_to_num(sig, nan=1.0), img, self.trunc, self.sigma_weights, self.carve)

  def min_weight(self, sigma, count):
    '''fused sigma below sigma [voxels] (weights 1 / sigma^2), or count observations (unit weights)'''
    return 1 / (sigma * self.voxel) ** 2 if self.sigma_weights else count

  def surface(self, sigma):
    '''the zero crossings where the fused sigma is below sigma [voxels], and their colours'''
    args = (self.keys, self.rows, self.coords, self.n, self.tsdf, self.weight, self.color, self.voxel,
            self.min_weight(sigma, self.min_count[0]))
    m = crossings(*args, None, None)
    pts, col = np.empty((m, 3)), np.empty((m, 3), np.uint8)
    crossings(*args, pts, col)
    return pts, col

  def near_surface(self, points, sigma, band):
    return near_surface(self.keys, self.rows, self.tsdf, self.weight, self.voxel, self.min_weight(sigma, self.min_count[1]), band,
                        np.ascontiguousarray(points, np.float64))

  def mesh(self, sigma, max_voxels=200_000_000):
    '''marching cubes over the bounding box of the blocks (vertices in the map frame, a colour per vertex), or None'''
    from skimage.measure import marching_cubes
    if self.n == 0:
      return None
    c = self.coords[:self.n]
    lo = c.min(0)
    shape = (c.max(0) - lo + 1) * B
    if np.prod(shape) > max_voxels:
      return None
    vol = np.ones(shape, np.float32)
    good = np.zeros(shape, bool)
    col = np.zeros(tuple(shape) + (3,), np.uint8)
    for b, (x, y, z) in enumerate((c - lo) * B):
      vol[x:x + B, y:y + B, z:z + B] = self.tsdf[b]
      good[x:x + B, y:y + B, z:z + B] = self.weight[b] >= self.min_weight(sigma, self.min_count[0])
      col[x:x + B, y:y + B, z:z + B] = self.color[b]
    if not good.any():
      return None
    verts, faces, _, _ = marching_cubes(np.where(good, vol, 1.0), 0.0, mask=good)
    idx = np.clip(np.round(verts).astype(int), 0, np.array(shape) - 1)
    return (lo * B + verts + 0.5) * self.voxel, faces, col[idx[:, 0], idx[:, 1], idx[:, 2]]


@njit(cache=True, nogil=True)
def rehash(keys, rows, coords, n):
  for b in range(n):
    i = slot_of(keys, block_key(coords[b, 0], coords[b, 1], coords[b, 2]))
    keys[i] = block_key(coords[b, 0], coords[b, 1], coords[b, 2])
    rows[i] = b


# ------------------------------------------------------------------------------------------------- display helpers

def estimate_up(R_wc):
  '''
  world up from camera orientations (no gravity): cameras are held without roll, so their x axes are horizontal and
  up is the normal of the plane they span; if they barely turned (one direction), the mean of their -y axes
  '''
  if len(R_wc) == 0:
    return np.array([0.0, 0.0, 1.0])
  x, y = R_wc[:, :, 0], R_wc[:, :, 1]
  down = y.mean(0)
  _, s, Vt = np.linalg.svd(x, full_matrices=False)
  up = Vt[-1] if len(s) == 3 and s[1] > 0.2 * s[0] else -down
  return up / np.linalg.norm(up) * (1 if up @ down <= 0 else -1)


def up_rotation(up):
  '''the rotation taking up to +Z (rotating around the horizontal axis between them)'''
  z = np.array([0.0, 0.0, 1.0])
  v, c = np.cross(up, z), float(up @ z)
  if np.linalg.norm(v) < 1e-9:
    return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + V + V @ V / (1 + c)


def voxel_map(P, C, size, min_points=3, close=4):
  '''
  cubes of size (up = +Z frame) holding min_points points, coloured by their mean colour; the floor layer (the most
  populated height in the lower third, +-1 cube) closed in 2D by a disk of radius close cubes, holes taking the colour
  of the nearest floor cube. Returns cube centres and colours
  '''
  import scipy.ndimage as ndi
  key = np.floor(P / size).astype(np.int64)
  cells, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  col = np.zeros((len(cells), 3))
  np.add.at(col, inv.ravel(), C.astype(np.float64))
  col /= cnt[:, None]
  cells, col = cells[cnt >= min_points], col[cnt >= min_points]
  if close and len(cells):
    z = cells[:, 2]
    hist = np.bincount(z - z.min())
    floor = z.min() + int(np.argmax(hist[:max(1, len(hist) // 3)]))
    on = np.abs(z - floor) <= 1
    xy = cells[on, :2]
    o = xy.min(0) - close
    grid = np.zeros(tuple(xy.max(0) - o + close + 1), bool)
    grid[xy[:, 0] - o[0], xy[:, 1] - o[1]] = True
    cg = np.zeros(grid.shape + (3,))
    cg[xy[:, 0] - o[0], xy[:, 1] - o[1]] = col[on]
    yy, xx = np.mgrid[-close:close + 1, -close:close + 1]
    new = ndi.binary_closing(grid, structure=yy ** 2 + xx ** 2 <= close ** 2) & ~grid
    _, (iy, ix) = ndi.distance_transform_edt(~grid, return_indices=True)
    ny, nx = np.nonzero(new)
    cells = np.r_[cells, np.c_[ny + o[0], nx + o[1], np.full(len(ny), floor)]]
    col = np.r_[col, cg[iy[ny, nx], ix[ny, nx]]]
  return (cells + 0.5) * size, col.astype(np.uint8)


def backproject(K, T_cw, rho, step=1, img=None):
  '''world points (and image values) of an inverse-depth map, every step-th pixel'''
  h, w = rho.shape
  vv, uu = np.mgrid[0:h:step, 0:w:step]
  r = rho[vv, uu]
  ok = np.isfinite(r) & (r > 0)
  uu, vv, z = uu[ok], vv[ok], 1 / r[ok]
  X = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
  return transform(inv_pose(T_cw), X), (img[vv, uu] if img is not None else None)


# ------------------------------------------------------------------------------------------------- the live mapper

class Job:
  '''a keyframe's depth-map inputs, snapshotted from the SLAM (images are never modified, poses are copies)'''

  def __init__(self, **kw):
    self.__dict__.update(kw)


class Record:
  '''
  a fused keyframe depth map (half resolution) and how to re-place it: pose = T_rel @ T_cw of its reference keyframe
  (re-expressed when that one is culled), scale change measured at its map points (distances from the camera)
  '''

  def __init__(self, kf, frame, rho, sig, color, pts, dist, uv, z):
    self.ref, self.T_rel, self.frame = kf, np.eye(4), frame
    self.rho, self.sig, self.color = rho, sig, color
    self.pts, self.dist, self.uv, self.z = pts, dist, uv, z     # its map points (as computed: for evaluation)
    self.T, self.scale = None, 1.0         # (as fused)


class DenseMapper:
  '''
  live dense mapping: on_frame() after every SLAM frame (cheap: buffers images, snapshots jobs for the keyframes that
  are due); the depth maps, the fusion and the logging run in a worker thread, in order (results do not depend on
  timing); the records and the TSDF belong to the worker until finish()
  '''

  def __init__(self, config, K, load_color, load=None, rr=None, frame_offset=0, threaded=True):
    self.c = config
    self.K = K
    self.Kh = K.copy()          # (the stored maps: every 2nd pixel)
    self.Kh[:2] /= 2
    self.load_color, self.load = load_color, load       # frame -> RGB / grey (load: for refine)
    self.rr, self.frame_offset = rr, frame_offset        # rerun time = frame + frame_offset
    self.frames = {}              # recent frames: index -> grey image
    self.kf_images = {}           # keyframe -> grey image
    self.kf_seen = 0              # keyframes looked at so far
    self.pending = []             # keyframes waiting for their delay
    self.n_events = 0
    self.traj_rows = {}           # frame -> row of slam.traj
    self.records = {}             # keyframe -> Record (worker thread)
    self.tsdf = None
    self.timing = {'depth': 0.0, 'fuse': 0.0, 'refuse': 0.0, 'log': 0.0}
    self.counts = {'maps': 0, 'refuse': 0, 'skipped': 0}
    self.flows = Flows(config.dis_preset, config.keyframes)
    self.queue = queue.Queue()
    self.error = None
    self.worker = threading.Thread(target=self.work, daemon=True) if threaded else None
    if self.worker is not None:
      self.worker.start()

  # --- main thread

  def frame_pose(self, slam, f):
    row = self.traj_rows.get(f)
    if row is None:
      return None
    T = slam.traj.T[row] @ slam.map.kfs.T[slam.traj.ref[row]]
    return T if np.isfinite(T).all() else None

  def on_frame(self, slam, f, img):
    if self.error is not None:
      raise RuntimeError('dense worker failed') from self.error
    c = self.c
    self.frames[f] = img
    keep = max(c.frames_before, default=0) + max(c.delay, max(c.frames_after, default=0)) + 1
    self.frames.pop(f - keep, None)
    if slam.traj.n and slam.traj.frame[slam.traj.n - 1] == f:
      self.traj_rows[f] = slam.traj.n - 1
    m = slam.map
    for k in range(self.kf_seen, len(m.kfs)):         # new keyframes
      self.kf_images[k] = self.frames.get(int(m.kfs.frame[k]), img)
      self.pending.append(k)
    self.kf_seen = len(m.kfs)
    for e in slam.events[self.n_events:]:
      if e['type'] == 'cull_kf':
        self.kf_images.pop(e['kf'], None)
        parent = int(m.kfs.parent[e['kf']])
        while not m.kfs.alive[parent]:
          parent = int(m.kfs.parent[parent])
        self.submit(('cull', e['kf'], parent, m.kfs.T[e['kf']].copy(), m.kfs.T[parent].copy()))
      if e['type'] in c.refuse:
        self.submit(('refuse', f, *self.placements(slam)))
    self.n_events = len(slam.events)
    if not slam.initialized:
      return
    due = [k for k in self.pending if m.kfs.frame[k] + c.delay <= f]
    for k in due:
      self.pending.remove(k)
      if not m.kfs.alive[k]:
        self.kf_images.pop(k, None)
      elif c.max_queue and self.worker is not None and self.queue.qsize() >= c.max_queue:
        self.counts['skipped'] += 1           # (behind: its neighbours cover most of it)
      else:
        self.submit(('map', self.job(slam, k, f)))

  def job(self, slam, k, now, refine=False):
    '''the snapshot for keyframe k: its image and pose, covisible keyframes, frames around it, its sparse points'''
    c, m = self.c, slam.map
    fk = int(m.kfs.frame[k])
    T = m.kfs.T[k].copy()
    covis = m.covisibility(k)
    covis[~m.kfs.alive] = 0
    order = np.argsort(-covis, kind='stable')[:c.keyframes]
    keyframes = [j for j in order[covis[order] > 0] if j in self.kf_images or refine]
    frames, used = [], set()
    for o in [-o for o in c.frames_before] + list(c.frames_after):
      f = fk + o
      if f > now or f < 0:
        continue
      img = self.frames.get(f) if not refine else self.load(f)
      Tf = self.frame_pose(slam, f)
      if img is not None and Tf is not None:
        frames.append((img, Tf))
        used.add(f)
    o = m.obs_of(kfs=[k])
    o = o[m.pts.state[m.obs.pt[o]] == MAPPED]
    pts = m.obs.pt[o]
    Xc = transform(T, m.pts.X[pts])
    image = self.kf_images.get(k)
    if image is None:
      image = self.load(fk)
    kf_views = [(self.kf_images[j] if j in self.kf_images else self.load(int(m.kfs.frame[j])), m.kfs.T[j].copy())
                for j in keyframes]
    # (a keyframe that is also one of the offset frames counts once in the sweep)
    sweep_views = frames + [v for j, v in zip(keyframes, kf_views) if int(m.kfs.frame[j]) not in used]
    return Job(kf=k, frame=fk, now=now, image=image, T=T, keyframes=kf_views, views=sweep_views,
               uv=m.obs.uv[o].copy(), z=Xc[:, 2], pts=pts, dist=np.linalg.norm(Xc, axis=1))

  def placements(self, slam):
    '''every keyframe's pose and every point's position now (for re-fusing after a pose-graph correction)'''
    return slam.map.kfs.T.copy(), slam.map.pts.X.copy(), slam.map.pts.state.copy()

  def submit(self, item):
    if self.worker is None:
      self.process(item)
    else:
      self.queue.put(item)

  def finish(self, slam):
    '''the keyframes still waiting (with the frames there are), optionally refine, wait for the worker'''
    m = slam.map
    if slam.initialized:
      for k in self.pending:
        if m.kfs.alive[k]:
          self.submit(('map', self.job(slam, k, slam.frame)))
      self.pending = []
      if self.c.refine and self.load is not None:
        jobs = [self.job(slam, k, slam.frame, refine=True) for k in np.flatnonzero(m.kfs.alive)]
        self.submit(('refine', jobs, *self.placements(slam)))
      else:
        self.submit(('refuse', slam.frame, *self.placements(slam)))      # (with the final poses)
    self.submit(('log', slam.frame))
    if self.worker is not None:
      self.queue.put(None)
      self.worker.join()
    if self.error is not None:
      raise RuntimeError('dense worker failed') from self.error

  # --- worker thread

  def work(self):
    import numba
    if self.c.threads:
      numba.set_num_threads(self.c.threads)
    while (item := self.queue.get()) is not None:
      try:
        self.process(item)
      except Exception as e:      # (re-raised in the main thread)
        self.error = e
        return

  def process(self, item):
    kind = item[0]
    if kind == 'map':
      self.map_keyframe(item[1])
    elif kind == 'cull':
      _, k, parent, T_k, T_p = item
      for r in self.records.values():
        if r.ref == k:
          r.T_rel, r.ref = r.T_rel @ T_k @ inv_pose(T_p), parent
    elif kind == 'refuse':
      self.refuse(*item[1:])
    elif kind == 'refine':
      _, jobs, kfs_T, X, state = item
      self.records = {}
      t = time.perf_counter()
      for job in jobs:
        if (job.z > 0).sum() >= 10:
          self.records[job.kf] = self.record(job)
      self.timing['depth'] += time.perf_counter() - t
      self.refuse(jobs[-1].now if jobs else 0, kfs_T, X, state)
    elif kind == 'log':
      self.log(item[1], final=True)

  def record(self, job):
    rho, sig = depth_map(self.c, self.K, job, self.flows)
    h, w = rho.shape
    small = lambda a: a[::2, ::2][:h // 2, :w // 2].copy()
    color = self.load_color(job.frame)
    color = cv2.resize(color, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    self.counts['maps'] += 1
    return Record(job.kf, job.frame, small(rho).astype(np.float32), small(sig).astype(np.float32), color, job.pts,
                  job.dist, job.uv, job.z)

  def map_keyframe(self, job):
    if (job.z > 0).sum() < 10:      # (no depth range, no calibration: e.g. a keyframe of a part that lost its map)
      return
    t = time.perf_counter()
    r = self.record(job)
    self.timing['depth'] += time.perf_counter() - t
    t = time.perf_counter()
    if self.tsdf is None:
      self.tsdf = self.new_tsdf(self.c.voxel * float(np.median(job.z[job.z > 0])))
    self.records[job.kf] = r
    r.T, r.scale = job.T, 1.0
    self.tsdf.integrate(self.Kh, r.T, r.rho, r.sig, r.color)
    self.timing['fuse'] += time.perf_counter() - t
    if self.rr is not None:
      self.rr.set_time('frame', sequence=job.now + self.frame_offset)
      if self.c.log_depth:      # (in the camera frame, under the keyframe's transform: re-fusing moves only that)
        X, col = backproject(self.Kh, np.eye(4), r.rho, max(1, self.c.points_step // 2), r.color)
        self.rr.log(f'world/map/dense/keyframes/{job.kf}', self.rr.Points3D(X, colors=col, radii=0.002))
        self.log_placement(job.kf, r)
      if self.counts['maps'] % self.c.log_every == 0:
        self.log(job.now)

  def refuse(self, now, kfs_T, X, state):
    '''all stored maps fused again at their keyframes' current poses, rescaled by their map points' distances'''
    t = time.perf_counter()
    if self.tsdf is None:
      return
    self.tsdf = self.new_tsdf(self.tsdf.voxel)
    for r in self.records.values():
      r.T = r.T_rel @ kfs_T[r.ref]
      alive = state[r.pts] == MAPPED
      r.scale = 1.0
      if alive.sum() >= 10:
        d_now = np.linalg.norm(transform(r.T, X[r.pts[alive]]), axis=1)
        r.scale = float(np.median(d_now / r.dist[alive]))
      self.tsdf.integrate(self.Kh, r.T, r.rho / r.scale, r.sig / r.scale, r.color)
    self.counts['refuse'] += 1
    self.timing['refuse'] += time.perf_counter() - t
    if self.rr is not None:
      self.rr.set_time('frame', sequence=now + self.frame_offset)
      if self.c.log_depth:
        for k, r in self.records.items():
          self.log_placement(k, r)
      self.log(now)

  def new_tsdf(self, voxel):
    return TSDF(voxel, self.c.truncation, self.c.sigma_weights, self.c.carve, self.c.min_count)

  def log_placement(self, k, r):
    T_wc = inv_pose(r.T)
    self.rr.log(f'world/map/dense/keyframes/{k}', self.rr.Transform3D(translation=T_wc[:3, 3], mat3x3=r.scale * T_wc[:3, :3]))

  # --- outputs

  def surface(self):
    '''the fused surface points and colours (map frame)'''
    if self.tsdf is None:
      return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return self.tsdf.surface(self.c.surface_sigma)

  def dense_points(self, step=None):
    '''every keyframe's raw points next to a confident fused surface, and their colours (map frame)'''
    step = step or self.c.points_step // 2
    P, C = [np.zeros((0, 3))], [np.zeros((0, 3), np.uint8)]
    if self.tsdf is None:
      return P[0], C[0]
    for r in self.records.values():
      X, col = backproject(self.Kh, r.T, r.rho / r.scale, step, r.color)
      keep = self.tsdf.near_surface(X, self.c.points_sigma, self.c.points_band)
      P.append(X[keep])
      C.append(col[keep])
    return np.concatenate(P), np.concatenate(C)

  def up(self):
    '''the estimated up direction (map frame), from the fused keyframes' orientations'''
    T = np.array([r.T for r in self.records.values()])
    return estimate_up(inv_pose(T)[:, :3, :3]) if len(T) else np.array([0.0, 0.0, 1.0])

  def log(self, now, final=False):
    rr = self.rr
    if rr is None or self.tsdf is None:
      return
    t = time.perf_counter()
    rr.set_time('frame', sequence=now + self.frame_offset + (2 if final else 0))
    S, col = self.surface()
    rr.log('world/map/dense/surface', rr.Points3D(S, colors=col, radii=0.004))
    P, C = self.dense_points()
    rr.log('world/map/dense/points', rr.Points3D(P, colors=C, radii=0.002))
    if self.c.mesh:
      m = self.tsdf.mesh(self.c.surface_sigma)
      if m is not None:
        rr.log('world/map/dense/mesh', rr.Mesh3D(vertex_positions=m[0], triangle_indices=m[1], vertex_colors=m[2]))
    if self.c.cube:     # (cubes aligned with up: built in the up frame, rotated back into the map frame)
      from scipy.spatial.transform import Rotation
      R = up_rotation(self.up())
      size = self.c.cube * self.tsdf.voxel
      X, c = voxel_map(np.r_[S, P] @ R.T, np.r_[col, C], size)
      q = Rotation.from_matrix(R.T).as_quat()
      rr.log('world/map/dense/voxels', rr.Boxes3D(centers=X @ R, half_sizes=np.full((len(X), 3), size / 2), colors=c,
                                                  quaternions=np.tile(q, (len(X), 1)), fill_mode='solid'))
    self.timing['log'] += time.perf_counter() - t
