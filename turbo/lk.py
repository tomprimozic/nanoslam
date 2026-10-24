'''
Patch alignment from a keyframe to the current image ("search by projection" / re-measuring map points).

The template is sampled from the source keyframe around the point's observation, *warped* into the target view with
the local affine approximation `A` of the homography of the fronto-parallel plane through the point (SVO-style), from
the source pyramid level whose resolution matches. It is then aligned in the target pyramid, coarse to fine, with
inverse-compositional Lucas-Kanade on zero-mean / unit-variance patches (invariant to gain and bias changes, e.g.
auto exposure). Optionally, a brute-force ZNCC search on the top level finds the start for wide searches.
'''
import cv2
import numpy as np
from numba import njit


LEVELS = 4


def pyramid(img):
  levels = [img]
  for _ in range(LEVELS - 1):
    levels.append(cv2.pyrDown(levels[-1]))
  return tuple(levels)


@njit(cache=True, inline='always')
def bilinear(img, x, y):
  x0, y0 = int(x), int(y)
  ax, ay = x - x0, y - y0
  return ((1 - ay) * ((1 - ax) * img[y0, x0] + ax * img[y0, x0 + 1])
          + ay * ((1 - ax) * img[y0 + 1, x0] + ax * img[y0 + 1, x0 + 1]))


@njit(cache=True, inline='always')
def inside(img, x, y, border):
  return border <= x < img.shape[1] - 1 - border and border <= y < img.shape[0] - 1 - border


@njit(cache=True, nogil=True)
def sample_normalized(img, x, y, half, out):
  '''(2 half + 1)^2 patch at sub-pixel (x, y), normalized to zero mean / unit variance; False if outside / flat'''
  if not inside(img, x - half, y - half, 0) or not inside(img, x + half, y + half, 0):
    return False
  s, ss, n = 0.0, 0.0, (2 * half + 1) ** 2
  for i in range(2 * half + 1):
    for j in range(2 * half + 1):
      v = bilinear(img, x - half + j, y - half + i)
      out[i, j] = v
      s += v
      ss += v * v
  mean = s / n
  std = np.sqrt(max(ss / n - mean * mean, 0.0))
  if std < 1.0:
    return False
  for i in range(2 * half + 1):
    for j in range(2 * half + 1):
      out[i, j] = (out[i, j] - mean) / std
  return True


@njit(cache=True, nogil=True)
def sample_template(src, us, vs, A, scale_src, scale_tgt, half, T):
  '''
  template of size (2 half + 3)^2 (with a 1 px border for gradients) in target-level pixels: target offset d maps to
  source level-0 offset A @ d * scale_tgt, read from the source level with scale `scale_src`
  '''
  h = half + 1
  for i in range(2 * h + 1):
    for j in range(2 * h + 1):
      dx, dy = (j - h) * scale_tgt, (i - h) * scale_tgt
      x = (us + A[0, 0] * dx + A[0, 1] * dy) / scale_src
      y = (vs + A[1, 0] * dx + A[1, 1] * dy) / scale_src
      if not inside(src, x, y, 0):
        return False
      T[i, j] = bilinear(src, x, y)
  return True


@njit(cache=True, nogil=True)
def align_patches(src_pyr, src_index, tgt_pyr, uv_src, A, uv_guess, top_level, search_px, half, max_iter):
  '''
  src_pyr: pyramids of all source keyframes (a tuple of (n_keyframes, h, w) uint8 arrays, one per level), src_index:
  (n,) which one each point's patch comes from, tgt_pyr: pyramid of the target image (tuple of (h, w) uint8 arrays),
  uv_src: (n, 2) observations in the source keyframes, A: (n, 2, 2) affine warps target -> source (level 0 pixels),
  uv_guess: (n, 2) initial positions in the target, top_level: coarsest target level, search_px: half-width of the
  brute-force ZNCC search on the top level (0 = none)
  returns uv (n, 2), ncc (n,) (-1 = failed)
  '''
  n = len(uv_src)
  P = 2 * half + 1
  uv = uv_guess.copy()
  ncc = np.full(n, -1.0)
  T = np.empty((P + 2, P + 2))
  Tn = np.empty((P, P))
  gx = np.empty((P, P))
  gy = np.empty((P, P))
  I = np.empty((P, P))
  n_src = len(src_pyr)
  for k in range(n):
    footprint = np.sqrt(abs(A[k, 0, 0] * A[k, 1, 1] - A[k, 0, 1] * A[k, 1, 0]))
    x, y = uv_guess[k, 0] / 2 ** top_level, uv_guess[k, 1] / 2 ** top_level
    ok = True
    for level in range(top_level, -1, -1):
      scale = 2.0 ** level
      tgt = tgt_pyr[level]
      # source level whose pixels have about the same size as the target level's (after warping)
      ls = int(np.round(np.log2(max(footprint * scale, 1e-3))))
      ls = min(max(ls, 0), n_src - 1)
      if not sample_template(src_pyr[ls][src_index[k]], uv_src[k, 0], uv_src[k, 1], A[k], 2.0 ** ls, scale, half, T):
        ok = False
        break
      # normalize the template, gradients of the normalized template, Gauss-Newton Hessian
      s, ss = 0.0, 0.0
      for i in range(P):
        for j in range(P):
          s += T[i + 1, j + 1]
          ss += T[i + 1, j + 1] ** 2
      mean = s / P ** 2
      std = np.sqrt(max(ss / P ** 2 - mean * mean, 0.0))
      if std < 1.0:
        ok = False
        break
      H00, H01, H11 = 0.0, 0.0, 0.0
      for i in range(P):
        for j in range(P):
          Tn[i, j] = (T[i + 1, j + 1] - mean) / std
          gx[i, j] = (T[i + 1, j + 2] - T[i + 1, j]) / (2 * std)
          gy[i, j] = (T[i + 2, j + 1] - T[i, j + 1]) / (2 * std)
          H00 += gx[i, j] ** 2
          H01 += gx[i, j] * gy[i, j]
          H11 += gy[i, j] ** 2
      det = H00 * H11 - H01 * H01
      if det < 1e-6 * (H00 + H11) ** 2 or H00 + H11 < 1e-6:
        ok = False
        break

      if level == top_level and search_px > 0:
        # brute-force ZNCC search (integer offsets) around the guess
        best, bx, by = -2.0, x, y
        for oy in range(-search_px, search_px + 1):
          for ox in range(-search_px, search_px + 1):
            if sample_normalized(tgt, x + ox, y + oy, half, I):
              c = 0.0
              for i in range(P):
                for j in range(P):
                  c += I[i, j] * Tn[i, j]
              if c > best:
                best, bx, by = c, x + ox, y + oy
        x, y = bx, by

      for it in range(max_iter):
        if not sample_normalized(tgt, x, y, half, I):
          ok = False
          break
        b0, b1 = 0.0, 0.0
        for i in range(P):
          for j in range(P):
            r = I[i, j] - Tn[i, j]
            b0 += gx[i, j] * r
            b1 += gy[i, j] * r
        dx = (H11 * b0 - H01 * b1) / det
        dy = (H00 * b1 - H01 * b0) / det
        x -= dx
        y -= dy
        if dx * dx + dy * dy < 1e-4:
          break
      if not ok:
        break
      if level > 0:
        x, y = 2 * x, 2 * y
    if ok and sample_normalized(tgt_pyr[0], x, y, half, I):
      c = 0.0
      for i in range(P):
        for j in range(P):
          c += I[i, j] * Tn[i, j]
      uv[k, 0], uv[k, 1] = x, y
      ncc[k] = c / P ** 2
  return uv, ncc


def affine_warps(K, T_src, T_tgt, X, uv_src, d=4.0):
  '''
  (n, 2, 2) local affine maps from target to source pixels, induced by the plane through each point X (world) that
  is fronto-parallel in its source keyframe (T_src: (n, 4, 4), one per point); also the projection of X into the target
  '''
  Kinv = np.linalg.inv(K)
  z = np.einsum('nj,nj->n', T_src[:, 2, :3], X) + T_src[:, 2, 3]
  T_ts = T_tgt @ np.linalg.inv(T_src)

  def to_target(uv):
    rays = np.c_[uv, np.ones(len(uv))] @ Kinv.T * z[:, None]
    p = (np.einsum('nij,nj->ni', T_ts[:, :3, :3], rays) + T_ts[:, :3, 3]) @ K.T
    return p[:, :2] / p[:, 2:]

  u0 = to_target(uv_src)
  A_st = np.stack([(to_target(uv_src + [d, 0]) - u0) / d, (to_target(uv_src + [0, d]) - u0) / d], axis=2)
  det = A_st[:, 0, 0] * A_st[:, 1, 1] - A_st[:, 0, 1] * A_st[:, 1, 0]
  good = np.isfinite(det) & (np.abs(det) > 1e-3)
  A = np.zeros_like(A_st)
  A[good] = np.linalg.inv(A_st[good])
  return A, u0, good


class ImageStore:
  '''keyframe pyramids in one (capacity, h, w) array per level, so that patch alignment can read from any of them'''

  def __init__(self, pyr):
    self.levels = tuple(np.zeros((16,) + level.shape, np.uint8) for level in pyr)
    self.free = []
    self.n = 0

  def add(self, pyr):
    '''store a pyramid, returns its slot'''
    if self.free:
      slot = self.free.pop()
    else:
      if self.n == len(self.levels[0]):
        self.levels = tuple(np.concatenate([a, np.zeros_like(a)]) for a in self.levels)
      slot, self.n = self.n, self.n + 1
    for a, level in zip(self.levels, pyr):
      a[slot] = level
    return slot

  def release(self, slot):
    self.free.append(slot)
