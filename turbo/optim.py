'''
Non-linear least squares, hand-rolled with numba:
  - bundle adjustment: Levenberg-Marquardt, Huber loss, points eliminated with the Schur complement
  - motion-only pose refinement (Gauss-Newton, Huber)
  - multi-view triangulation (closest point to the rays, then Gauss-Newton on the reprojection error)
  - Sim3 pose graph (scipy least_squares with a sparse Jacobian)
Poses are T_cw = [R | t], updated as R <- exp(w) R, t <- t + v (in bundle adjustment: translation decoupled from the
rotation, so that a single translation component can be held fixed for the monocular scale gauge).
'''
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
from numba import njit

import geometry


CHI2_2D = 5.991           # 95% quantile, 2 DoF
HUBER = np.sqrt(CHI2_2D)  # px
BEHIND = 1e4              # cost of an observation behind the camera (must never be free, or BA "hides" outliers there)


@njit(cache=True, nogil=True)
def so3_exp(w):
  th = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
  K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
  if th < 1e-12:
    return np.eye(3) + K
  return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K)


@njit(cache=True, nogil=True)
def robust_weight(e2, delta):
  '''IRLS weight of a squared residual: Huber with scale delta'''
  return 1.0 if e2 <= delta * delta else delta / np.sqrt(e2)


@njit(cache=True, nogil=True)
def robust_cost(e2, delta):
  return 0.5 * e2 if e2 <= delta * delta else delta * (np.sqrt(e2) - 0.5 * delta)


@njit(cache=True, nogil=True)
def reprojection_cost(R, t, X, cam, pt, uv, f, c, delta):
  cost = 0.0
  for o in range(len(cam)):
    Xc = R[cam[o]] @ X[pt[o]] + t[cam[o]]
    if Xc[2] <= 1e-9:
      cost += BEHIND
      continue
    rx = f[0] * Xc[0] / Xc[2] + c[0] - uv[o, 0]
    ry = f[1] * Xc[1] / Xc[2] + c[1] - uv[o, 1]
    cost += robust_cost(rx * rx + ry * ry, delta)
  return cost


@njit(cache=True, nogil=True)
def linearize(R, t, X, cam, pt, uv, free, mask, f, c, delta, n_free):
  '''
  normal equations: U (pose blocks), V (point blocks), W (per observation pose-point block), gradients;
  mask (n_free, 6): 0 = this pose parameter is held fixed
  '''
  n_pts = len(X)
  U = np.zeros((n_free, 6, 6))
  gc = np.zeros((n_free, 6))
  V = np.zeros((n_pts, 3, 3))
  gp = np.zeros((n_pts, 3))
  W = np.zeros((len(cam), 6, 3))
  Jp = np.zeros((2, 3))
  Jc = np.zeros((2, 6))
  for o in range(len(cam)):
    k, p = cam[o], pt[o]
    Y = R[k] @ X[p]
    x, y, z = Y[0] + t[k, 0], Y[1] + t[k, 1], Y[2] + t[k, 2]
    if z <= 1e-9:
      continue
    r0 = f[0] * x / z + c[0] - uv[o, 0]
    r1 = f[1] * y / z + c[1] - uv[o, 1]
    w = robust_weight(r0 * r0 + r1 * r1, delta)
    # d proj / d Xc
    a00, a02 = f[0] / z, -f[0] * x / z ** 2
    a11, a12 = f[1] / z, -f[1] * y / z ** 2
    # pose: d Xc / d w = -[R X]x, d Xc / d v = I
    Jc[0, 0], Jc[0, 1], Jc[0, 2] = a02 * Y[1], a00 * Y[2] - a02 * Y[0], -a00 * Y[1]
    Jc[1, 0], Jc[1, 1], Jc[1, 2] = -a11 * Y[2] + a12 * Y[1], -a12 * Y[0], a11 * Y[0]
    Jc[0, 3], Jc[0, 4], Jc[0, 5] = a00, 0.0, a02
    Jc[1, 3], Jc[1, 4], Jc[1, 5] = 0.0, a11, a12
    # point: d Xc / d X = R
    for j in range(3):
      Jp[0, j] = a00 * R[k, 0, j] + a02 * R[k, 2, j]
      Jp[1, j] = a11 * R[k, 1, j] + a12 * R[k, 2, j]
    for i in range(3):
      gp[p, i] += w * (Jp[0, i] * r0 + Jp[1, i] * r1)
      for j in range(3):
        V[p, i, j] += w * (Jp[0, i] * Jp[0, j] + Jp[1, i] * Jp[1, j])
    q = free[k]
    if q >= 0:
      for i in range(6):
        Jc[0, i] *= mask[q, i]
        Jc[1, i] *= mask[q, i]
      for i in range(6):
        gc[q, i] += w * (Jc[0, i] * r0 + Jc[1, i] * r1)
        for j in range(6):
          U[q, i, j] += w * (Jc[0, i] * Jc[0, j] + Jc[1, i] * Jc[1, j])
        for j in range(3):
          W[o, i, j] = w * (Jc[0, i] * Jp[0, j] + Jc[1, i] * Jp[1, j])
  for q in range(n_free):
    for i in range(6):
      if mask[q, i] == 0:
        U[q, i, i] = 1.0
  return U, gc, V, gp, W


@njit(cache=True, nogil=True)
def inv3(A, out):
  '''closed-form inverse of a 3x3 matrix (no LAPACK)'''
  c00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
  c01 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
  c02 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
  det = A[0, 0] * c00 + A[0, 1] * c01 + A[0, 2] * c02
  if det == 0.0:
    out[:] = 0.0
    return 0.0
  d = 1.0 / det
  out[0, 0], out[1, 0], out[2, 0] = c00 * d, c01 * d, c02 * d
  out[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * d
  out[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * d
  out[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) * d
  out[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * d
  out[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) * d
  out[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) * d
  return det


@njit(cache=True, nogil=True)
def schur(U, gc, V, gp, W, cam, free, ptr, order, lam):
  '''reduced camera system S dc = b, with Marquardt damping lam (explicit loops: no temporaries)'''
  n_free = len(U)
  S = np.zeros((6 * n_free, 6 * n_free))
  b = np.zeros(6 * n_free)
  for q in range(n_free):
    for i in range(6):
      b[6 * q + i] = -gc[q, i]
      for j in range(6):
        S[6 * q + i, 6 * q + j] = U[q, i, j]
      S[6 * q + i, 6 * q + i] += lam * U[q, i, i] + 1e-9
  Vinv = np.zeros_like(V)
  WV = np.zeros((len(cam), 6, 3))
  Vd = np.zeros((3, 3))
  for p in range(len(V)):
    for i in range(3):
      for j in range(3):
        Vd[i, j] = V[p, i, j]
      Vd[i, i] += lam * V[p, i, i] + 1e-9
    inv3(Vd, Vinv[p])
    for a in range(ptr[p], ptr[p + 1]):
      oa = order[a]
      qa = free[cam[oa]]
      if qa < 0:
        continue
      for i in range(6):
        for j in range(3):
          WV[oa, i, j] = W[oa, i, 0] * Vinv[p, 0, j] + W[oa, i, 1] * Vinv[p, 1, j] + W[oa, i, 2] * Vinv[p, 2, j]
        b[6 * qa + i] += WV[oa, i, 0] * gp[p, 0] + WV[oa, i, 1] * gp[p, 1] + WV[oa, i, 2] * gp[p, 2]
      for bb in range(ptr[p], ptr[p + 1]):
        ob = order[bb]
        qb = free[cam[ob]]
        if qb < 0:
          continue
        for i in range(6):
          for j in range(6):
            S[6 * qa + i, 6 * qb + j] -= (WV[oa, i, 0] * W[ob, j, 0] + WV[oa, i, 1] * W[ob, j, 1]
                                          + WV[oa, i, 2] * W[ob, j, 2])
  return S, b, Vinv


@njit(cache=True, nogil=True)
def back_substitute(dc, Vinv, gp, W, cam, free, ptr, order):
  dp = np.zeros_like(gp)
  rhs = np.zeros(3)
  for p in range(len(gp)):
    for i in range(3):
      rhs[i] = -gp[p, i]
    for a in range(ptr[p], ptr[p + 1]):
      o = order[a]
      q = free[cam[o]]
      if q >= 0:
        for i in range(3):
          for j in range(6):
            rhs[i] -= W[o, j, i] * dc[6 * q + j]
    for i in range(3):
      dp[p, i] = Vinv[p, i, 0] * rhs[0] + Vinv[p, i, 1] * rhs[1] + Vinv[p, i, 2] * rhs[2]
  return dp


@njit(cache=True, nogil=True)
def apply_update(R, t, X, dc, dp, free):
  R, t = R.copy(), t.copy()
  for k in range(len(R)):
    q = free[k]
    if q >= 0:
      R[k] = so3_exp(dc[6 * q:6 * q + 3]) @ R[k]
      t[k] = t[k] + dc[6 * q + 3:6 * q + 6]
  return R, t, X + dp


def bundle_adjust(K, T_cw, fixed, X, cam, pt, uv, iterations=10, delta=HUBER, scale_fix=None):
  '''
  T_cw: (n_cams, 4, 4), fixed: (n_cams,) bool, X: (n_pts, 3), observations: cam / pt indices and pixels uv
  scale_fix: (camera, axis): hold this translation component fixed (monocular gauge with a single fixed camera)
  returns optimized T_cw, X and the squared reprojection error of every observation (inf if behind the camera)
  '''
  f, c = np.array([K[0, 0], K[1, 1]]), np.array([K[0, 2], K[1, 2]])
  R, t = T_cw[:, :3, :3].copy(), T_cw[:, :3, 3].copy()
  X = np.asarray(X, dtype=np.float64).copy()
  free = np.where(fixed, -1, np.cumsum(~fixed) - 1).astype(np.int64)
  n_free = int((~fixed).sum())
  mask = np.ones((n_free, 6))
  if scale_fix is not None:
    mask[free[scale_fix[0]], 3 + scale_fix[1]] = 0
  cam, pt, uv = cam.astype(np.int64), pt.astype(np.int64), uv.astype(np.float64)
  order = np.argsort(pt, kind='stable')
  ptr = np.searchsorted(pt[order], np.arange(len(X) + 1))
  cost = reprojection_cost(R, t, X, cam, pt, uv, f, c, delta)
  lam = 1e-4
  for _ in range(iterations):
    U, gc, V, gp, W = linearize(R, t, X, cam, pt, uv, free, mask, f, c, delta, n_free)
    improved = False
    while lam < 1e8:
      S, b, Vinv = schur(U, gc, V, gp, W, cam, free, ptr, order, lam)
      try:   # (not scipy.linalg.solve(assume_a='pos'): in scipy 1.18.1 / Accelerate it corrupts the heap)
        dc = scipy.linalg.cho_solve(scipy.linalg.cho_factor(S), b) if n_free else np.zeros(0)
      except (np.linalg.LinAlgError, ValueError):
        lam *= 10
        continue
      dp = back_substitute(dc, Vinv, gp, W, cam, free, ptr, order)
      R_new, t_new, X_new = apply_update(R, t, X, dc, dp, free)
      new_cost = reprojection_cost(R_new, t_new, X_new, cam, pt, uv, f, c, delta)
      if np.isfinite(new_cost) and new_cost < cost:
        improved = cost - new_cost > 1e-6 * cost
        R, t, X, cost = R_new, t_new, X_new, new_cost
        lam = max(lam / 5, 1e-8)
        break
      lam *= 5
    if not improved:
      break
  T_out = T_cw.copy()
  T_out[:, :3, :3], T_out[:, :3, 3] = R, t
  return T_out, X, squared_errors(R, t, X, cam, pt, uv, f, c)


@njit(cache=True, nogil=True)
def squared_errors(R, t, X, cam, pt, uv, f, c):
  e2 = np.empty(len(cam))
  for o in range(len(cam)):
    Xc = R[cam[o]] @ X[pt[o]] + t[cam[o]]
    if Xc[2] <= 1e-9:
      e2[o] = np.inf
    else:
      e2[o] = (f[0] * Xc[0] / Xc[2] + c[0] - uv[o, 0]) ** 2 + (f[1] * Xc[1] / Xc[2] + c[1] - uv[o, 1]) ** 2
  return e2


@njit(cache=True, nogil=True)
def _refine_pose(R, t, X, uv, f, c, delta, iterations):
  for _ in range(iterations):
    H = np.zeros((6, 6))
    g = np.zeros(6)
    J = np.zeros((2, 6))
    for o in range(len(X)):
      Xc = R @ X[o] + t
      x, y, z = Xc[0], Xc[1], Xc[2]
      if z <= 1e-9:
        continue
      r0 = f[0] * x / z + c[0] - uv[o, 0]
      r1 = f[1] * y / z + c[1] - uv[o, 1]
      w = robust_weight(r0 * r0 + r1 * r1, delta)
      a00, a02 = f[0] / z, -f[0] * x / z ** 2
      a11, a12 = f[1] / z, -f[1] * y / z ** 2
      J[0, 0], J[0, 1], J[0, 2] = a02 * y, a00 * z - a02 * x, -a00 * y
      J[1, 0], J[1, 1], J[1, 2] = -a11 * z + a12 * y, -a12 * x, a11 * x
      J[0, 3], J[0, 4], J[0, 5] = a00, 0.0, a02
      J[1, 3], J[1, 4], J[1, 5] = 0.0, a11, a12
      for i in range(6):
        g[i] += w * (J[0, i] * r0 + J[1, i] * r1)
        for j in range(6):
          H[i, j] += w * (J[0, i] * J[0, j] + J[1, i] * J[1, j])
    for i in range(6):
      H[i, i] += 1e-9
    d = -np.linalg.solve(H, g)
    dR = so3_exp(d[:3])
    R = dR @ R
    t = dR @ t + d[3:]
    if np.sum(d * d) < 1e-14:
      break
  return R, t


@njit(cache=True, nogil=True)
def landmark_costs(R, t, b, B, cam, uv, start, f, c, sigma, delta, rho):
  '''
  depth-integrated landmark likelihoods (slamfold): landmark i with host bearing b_i (tangent basis B_i), inverse depth
  rho_ig at grid node g, seen in frame j at x_ij = R_j (b_i + B_i db) + rho_ig t_j. Returns E[i, g] = -log of the
  likelihood with the bearing correction db integrated out (Laplace: the robust cost at the best db + 1/2 log det of
  its Hessian), residuals in units of sigma, Huber at delta. Observations of landmark i: start[i]:start[i + 1]
  (the host observation among them: it is the prior of db)
  '''
  n, G = rho.shape
  E = np.zeros((n, G))
  m = 0
  for i in range(n):
    m = max(m, start[i + 1] - start[i])
  Rb = np.zeros((m, 3))
  RB = np.zeros((m, 3, 2))
  for i in range(n):
    k0, k1 = start[i], start[i + 1]
    for o in range(k0, k1):     # R_j b_i and R_j B_i do not depend on the depth or the bearing correction
      j = cam[o]
      for r in range(3):
        Rb[o - k0, r] = R[j, r, 0] * b[i, 0] + R[j, r, 1] * b[i, 1] + R[j, r, 2] * b[i, 2]
        for q in range(2):
          RB[o - k0, r, q] = R[j, r, 0] * B[i, 0, q] + R[j, r, 1] * B[i, 1, q] + R[j, r, 2] * B[i, 2, q]
    db0, db1 = 0.0, 0.0
    for g in range(G):
      for it in range(4):
        h00, h01, h11, g0, g1, e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for o in range(k0, k1):
          q = o - k0
          j = cam[o]
          x0 = Rb[q, 0] + RB[q, 0, 0] * db0 + RB[q, 0, 1] * db1 + rho[i, g] * t[j, 0]
          x1 = Rb[q, 1] + RB[q, 1, 0] * db0 + RB[q, 1, 1] * db1 + rho[i, g] * t[j, 1]
          x2 = Rb[q, 2] + RB[q, 2, 0] * db0 + RB[q, 2, 1] * db1 + rho[i, g] * t[j, 2]
          if x2 <= 1e-9:
            e += BEHIND
            continue
          r0 = (f[0] * x0 / x2 + c[0] - uv[o, 0]) / sigma
          r1 = (f[1] * x1 / x2 + c[1] - uv[o, 1]) / sigma
          e2 = r0 * r0 + r1 * r1
          w = robust_weight(e2, delta)
          e += robust_cost(e2, delta)
          a00, a02 = f[0] / (x2 * sigma), -f[0] * x0 / (x2 * x2 * sigma)
          a11, a12 = f[1] / (x2 * sigma), -f[1] * x1 / (x2 * x2 * sigma)
          j00, j01 = a00 * RB[q, 0, 0] + a02 * RB[q, 2, 0], a00 * RB[q, 0, 1] + a02 * RB[q, 2, 1]
          j10, j11 = a11 * RB[q, 1, 0] + a12 * RB[q, 2, 0], a11 * RB[q, 1, 1] + a12 * RB[q, 2, 1]
          h00 += w * (j00 * j00 + j10 * j10)
          h01 += w * (j00 * j01 + j10 * j11)
          h11 += w * (j01 * j01 + j11 * j11)
          g0 += w * (j00 * r0 + j10 * r1)
          g1 += w * (j01 * r0 + j11 * r1)
        det = h00 * h11 - h01 * h01 + 1e-12
        if it == 3:
          E[i, g] = e + 0.5 * np.log(det)
          break
        db0 -= (h11 * g0 - h01 * g1) / det
        db1 -= (h00 * g1 - h01 * g0) / det
  return E


def depth_integrated(R, t, b, B, cam, uv, start, K, sigma, rho_ref):
  '''
  per landmark: the likelihood integrated over inverse depth with a flat prior on [0, U], for the support U that
  maximizes the total (scaling the translations by k is scaling U by k: this is the optimum of the translation
  magnitude, which the BA's gauge fixed). Returns the per-landmark log integrals and the posterior mean / std of rho
  at that U. rho_ref: a typical inverse depth (the grid spans 1e-5 .. 300 x it, refined around each posterior)
  '''
  f, c = np.array([K[0, 0], K[1, 1]]), np.array([K[0, 2], K[1, 2]])
  args = (np.ascontiguousarray(R), np.ascontiguousarray(t), b, B, cam, uv, start, f, c, sigma, HUBER)
  n = len(b)
  base = np.r_[0.0, rho_ref * np.logspace(-5, 2.5, 64)]
  grid = np.tile(base, (n, 1))
  E = landmark_costs(*args, grid)
  # refine around each landmark's best node: +-6 widths (from the curvature there, at least the local grid step)
  best = np.argmin(E, axis=1)
  ii = np.arange(n)
  k = np.clip(best, 1, grid.shape[1] - 2)
  h = (grid[ii, k + 1] - grid[ii, k - 1]) / 2
  curv = (E[ii, k + 1] - 2 * E[ii, k] + E[ii, k - 1]) / h ** 2
  width = np.clip(np.where(curv > 0, 1 / np.sqrt(np.maximum(curv, 1e-300)), h), 1e-3 * h, h)
  fine = np.maximum(grid[ii, best][:, None] + width[:, None] * np.linspace(-6, 6, 25), 0)
  grid, E = np.c_[grid, fine], np.c_[E, landmark_costs(*args, fine)]
  order = np.argsort(grid, axis=1)
  grid, E = np.take_along_axis(grid, order, 1), np.take_along_axis(E, order, 1)
  # cumulative integral of exp(-E) (trapezoid, in units of exp(-E0) for stability), then the average over [0, U]
  E0 = E.min(axis=1, keepdims=True)
  L = np.exp(-(E - E0))
  cum = np.c_[np.zeros(n), np.cumsum(np.diff(grid, axis=1) * (L[:, 1:] + L[:, :-1]) / 2, axis=1)]
  supports = rho_ref * np.logspace(-5, 2.5, 76)
  U = supports[:, None]
  j = np.minimum((grid[None] < U[..., None]).sum(axis=2), grid.shape[1] - 1)     # (supports, landmarks)
  jl = np.maximum(j - 1, 0)
  g_l, g_r = np.take_along_axis(grid, jl.T, 1).T, np.take_along_axis(grid, j.T, 1).T
  L_l, L_r = np.take_along_axis(L, jl.T, 1).T, np.take_along_axis(L, j.T, 1).T
  a_ = np.clip((U - g_l) / np.maximum(g_r - g_l, 1e-300), 0, 1)
  C = np.take_along_axis(cum, jl.T, 1).T + (U - g_l).clip(0, None) * (2 * L_l + a_ * (L_r - L_l)) / 2
  C = np.where(U >= grid[:, -1][None], cum[:, -1][None], C)
  log_avg = np.log(np.maximum(C, 1e-300)) - np.log(U) - E0[:, 0][None]
  q = int(np.argmax(log_avg.sum(axis=1)))
  U = supports[q]
  inside = grid <= U
  w = np.zeros_like(grid)
  dg = np.diff(grid, axis=1)
  w[:, :-1] += dg / 2
  w[:, 1:] += dg / 2
  p = np.where(inside, L * w, 0)
  p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
  mean = np.sum(p * grid, axis=1)
  std = np.sqrt(np.maximum(np.sum(p * grid ** 2, axis=1) - mean ** 2, 0))
  return log_avg[q], mean, std


def refine_pose(K, T_cw, X, uv, iterations=10, delta=HUBER):
  '''motion-only Gauss-Newton; returns T_cw and the squared reprojection errors'''
  f, c = np.array([K[0, 0], K[1, 1]]), np.array([K[0, 2], K[1, 2]])
  X, uv = np.asarray(X, np.float64), np.asarray(uv, np.float64)
  R, t = _refine_pose(T_cw[:3, :3].copy(), T_cw[:3, 3].copy(), X, uv, f, c, delta, iterations)
  T = geometry.make_pose(R, t)
  idx = np.arange(len(X))
  return T, squared_errors(R[None], t[None], X, idx * 0, idx, uv, f, c)


@njit(cache=True, nogil=True)
def _triangulate(R, t, cam, xn, ptr, order, iterations):
  n = len(ptr) - 1
  X = np.zeros((n, 3))
  max_angle = np.zeros(n)
  max_err = np.zeros(n)     # in normalized image coordinates
  min_depth = np.zeros(n)
  for p in range(n):
    m = ptr[p + 1] - ptr[p]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    centers = np.zeros((m, 3))
    for a in range(m):
      o = order[ptr[p] + a]
      k = cam[o]
      C = -R[k].T @ t[k]
      d = R[k].T @ np.array([xn[o, 0], xn[o, 1], 1.0])
      d /= np.sqrt(np.sum(d * d))
      P = np.eye(3) - np.outer(d, d)
      A += P
      b += P @ C
      centers[a] = C
    if abs(np.linalg.det(A)) < 1e-12:
      X[p] = np.nan
      continue
    x = np.linalg.solve(A, b)
    for _ in range(iterations):
      H = np.zeros((3, 3))
      g = np.zeros(3)
      for a in range(m):
        o = order[ptr[p] + a]
        k = cam[o]
        Xc = R[k] @ x + t[k]
        if Xc[2] <= 1e-9:
          continue
        r0 = Xc[0] / Xc[2] - xn[o, 0]
        r1 = Xc[1] / Xc[2] - xn[o, 1]
        J = np.zeros((2, 3))
        J[0, 0], J[0, 2] = 1 / Xc[2], -Xc[0] / Xc[2] ** 2
        J[1, 1], J[1, 2] = 1 / Xc[2], -Xc[1] / Xc[2] ** 2
        J = J @ R[k]
        H += J.T @ J
        g += J[0] * r0 + J[1] * r1
      if abs(np.linalg.det(H)) < 1e-18:
        break
      x = x - np.linalg.solve(H, g)
    X[p] = x
    min_depth[p] = np.inf
    for a in range(m):
      o = order[ptr[p] + a]
      k = cam[o]
      Xc = R[k] @ x + t[k]
      min_depth[p] = min(min_depth[p], Xc[2])
      if Xc[2] > 1e-9:
        e = np.sqrt((Xc[0] / Xc[2] - xn[o, 0]) ** 2 + (Xc[1] / Xc[2] - xn[o, 1]) ** 2)
        max_err[p] = max(max_err[p], e)
      ra = x - centers[a]
      for bb in range(a):
        rb = x - centers[bb]
        cos = np.sum(ra * rb) / np.sqrt(np.sum(ra * ra) * np.sum(rb * rb) + 1e-300)
        max_angle[p] = max(max_angle[p], np.arccos(min(max(cos, -1.0), 1.0)))
  return X, max_angle, max_err, min_depth


def triangulate(K, T_cw, cam, pt, uv, n_points, iterations=3):
  '''
  multi-view triangulation of n_points points from their observations (cam / pt indices, pixels uv)
  returns X, the largest ray angle (rad), the largest reprojection error (px) and the smallest depth per point
  '''
  pt = pt.astype(np.int64)
  order = np.argsort(pt, kind='stable')
  ptr = np.searchsorted(pt[order], np.arange(n_points + 1))
  xn = geometry.unproject(K, uv)[:, :2]
  X, angle, err, depth = _triangulate(T_cw[:, :3, :3].copy(), T_cw[:, :3, 3].copy(), cam.astype(np.int64), xn, ptr,
                                      order, iterations)
  return X, angle, err * np.sqrt(K[0, 0] * K[1, 1]), depth


@njit(cache=True, nogil=True)
def so3_log(R):
  cos = min(max((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2, -1.0), 1.0)
  th = np.arccos(cos)
  w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if th < 1e-6:
    return w / 2
  if th > np.pi - 1e-4:     # near pi: axis from the diagonal
    a = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0.0))
    i = np.argmax(a)
    axis = R[:, i] + np.eye(3)[i]
    axis /= np.sqrt(np.sum(axis * axis))
    return th * axis
  return th / (2 * np.sin(th)) * w


@njit(cache=True, nogil=True)
def sim3_edge_error(si, Ri, ti, sj, Rj, tj, sm, Rm, tm):
  '''log(inv(M) S_i inv(S_j)) as (rotation, translation, log scale), with S = (s, R, t): x -> s R x + t'''
  s_ij = si / sj
  R_ij = Ri @ Rj.T
  t_ij = ti - s_ij * (R_ij @ tj)
  r = np.empty(7)
  r[:3] = so3_log(Rm.T @ R_ij)
  r[3:6] = Rm.T @ (t_ij - tm) / sm
  r[6] = np.log(s_ij / sm)
  return r


@njit(cache=True, nogil=True)
def perturb(s, R, t, d):
  return s * np.exp(d[6]), so3_exp(d[:3]) @ R, t + d[3:6]


@njit(cache=True, nogil=True)
def _edge_error(si, Ri, ti, sj, Rj, tj, sm, Rm, tm, r, E):
  '''sim3_edge_error without allocations: r (7,) gets the residual, E (3, 3) is scratch'''
  s_ij = si / sj
  # R_ij = Ri Rj^T, t_ij = ti - s_ij R_ij tj, E = Rm^T R_ij
  Rij = E     # (scratch: R_ij)
  for p in range(3):
    for q in range(3):
      Rij[p, q] = Ri[p, 0] * Rj[q, 0] + Ri[p, 1] * Rj[q, 1] + Ri[p, 2] * Rj[q, 2]
  d0 = ti[0] - s_ij * (Rij[0, 0] * tj[0] + Rij[0, 1] * tj[1] + Rij[0, 2] * tj[2]) - tm[0]
  d1 = ti[1] - s_ij * (Rij[1, 0] * tj[0] + Rij[1, 1] * tj[1] + Rij[1, 2] * tj[2]) - tm[1]
  d2 = ti[2] - s_ij * (Rij[2, 0] * tj[0] + Rij[2, 1] * tj[1] + Rij[2, 2] * tj[2]) - tm[2]
  for p in range(3):
    r[3 + p] = (Rm[0, p] * d0 + Rm[1, p] * d1 + Rm[2, p] * d2) / sm
  r[6] = np.log(s_ij / sm)
  # E = Rm^T R_ij, then its log
  e00 = Rm[0, 0] * Rij[0, 0] + Rm[1, 0] * Rij[1, 0] + Rm[2, 0] * Rij[2, 0]
  e01 = Rm[0, 0] * Rij[0, 1] + Rm[1, 0] * Rij[1, 1] + Rm[2, 0] * Rij[2, 1]
  e02 = Rm[0, 0] * Rij[0, 2] + Rm[1, 0] * Rij[1, 2] + Rm[2, 0] * Rij[2, 2]
  e10 = Rm[0, 1] * Rij[0, 0] + Rm[1, 1] * Rij[1, 0] + Rm[2, 1] * Rij[2, 0]
  e11 = Rm[0, 1] * Rij[0, 1] + Rm[1, 1] * Rij[1, 1] + Rm[2, 1] * Rij[2, 1]
  e12 = Rm[0, 1] * Rij[0, 2] + Rm[1, 1] * Rij[1, 2] + Rm[2, 1] * Rij[2, 2]
  e20 = Rm[0, 2] * Rij[0, 0] + Rm[1, 2] * Rij[1, 0] + Rm[2, 2] * Rij[2, 0]
  e21 = Rm[0, 2] * Rij[0, 1] + Rm[1, 2] * Rij[1, 1] + Rm[2, 2] * Rij[2, 1]
  e22 = Rm[0, 2] * Rij[0, 2] + Rm[1, 2] * Rij[1, 2] + Rm[2, 2] * Rij[2, 2]
  cos = min(max((e00 + e11 + e22 - 1) / 2, -1.0), 1.0)
  th = np.arccos(cos)
  w0, w1, w2 = e21 - e12, e02 - e20, e10 - e01
  if th < 1e-6:
    r[0], r[1], r[2] = w0 / 2, w1 / 2, w2 / 2
  elif th > np.pi - 1e-4:     # near pi: the general path (rare)
    for p in range(3):
      E[p, 0], E[p, 1], E[p, 2] = (e00, e01, e02) if p == 0 else ((e10, e11, e12) if p == 1 else (e20, e21, e22))
    v = so3_log(E)
    r[0], r[1], r[2] = v[0], v[1], v[2]
  else:
    k = th / (2 * np.sin(th))
    r[0], r[1], r[2] = k * w0, k * w1, k * w2


@njit(cache=True, nogil=True)
def pose_graph_linearize(s, R, t, I, J, sm, Rm, tm, w, var):
  '''per edge residual and numerical Jacobians w.r.t. local increments of both nodes; also the cost'''
  n = len(I)
  r = np.zeros((n, 7))
  Ji = np.zeros((n, 7, 7))
  Jj = np.zeros((n, 7, 7))
  eps = 1e-6
  cost = 0.0
  Q = np.zeros((3, 3, 3))     # exp(eps e_a): the rotation perturbations (left-multiplied, as in perturb)
  for a in range(3):
    d = np.zeros(3)
    d[a] = eps
    Q[a] = so3_exp(d)
  E = np.zeros((3, 3))
  Rp = np.zeros((3, 3))
  tp = np.zeros(3)
  rp = np.zeros(7)
  for e in range(n):
    i, j = I[e], J[e]
    _edge_error(s[i], R[i], t[i], s[j], R[j], t[j], sm[e], Rm[e], tm[e], r[e], E)
    for q in range(7):
      cost += w[e] * r[e, q] * r[e, q]
    for side in range(2):
      k = i if side == 0 else j
      if var[k] < 0:
        continue
      for a in range(7):
        sk = s[k]
        for p in range(3):
          tp[p] = t[k, p]
          for q in range(3):
            Rp[p, q] = R[k, p, q]
        if a < 3:
          for p in range(3):
            for q in range(3):
              Rp[p, q] = Q[a, p, 0] * R[k, 0, q] + Q[a, p, 1] * R[k, 1, q] + Q[a, p, 2] * R[k, 2, q]
        elif a < 6:
          tp[a - 3] += eps
        else:
          sk = s[k] * np.exp(eps)
        if side == 0:
          _edge_error(sk, Rp, tp, s[j], R[j], t[j], sm[e], Rm[e], tm[e], rp, E)
          for q in range(7):
            Ji[e, q, a] = (rp[q] - r[e, q]) / eps
        else:
          _edge_error(s[i], R[i], t[i], sk, Rp, tp, sm[e], Rm[e], tm[e], rp, E)
          for q in range(7):
            Jj[e, q, a] = (rp[q] - r[e, q]) / eps
  return r, Ji, Jj, cost


@njit(cache=True, nogil=True)
def pose_graph_cost(s, R, t, I, J, sm, Rm, tm, w):
  cost = 0.0
  E = np.zeros((3, 3))
  r = np.zeros(7)
  for e in range(len(I)):
    _edge_error(s[I[e]], R[I[e]], t[I[e]], s[J[e]], R[J[e]], t[J[e]], sm[e], Rm[e], tm[e], r, E)
    for q in range(7):
      cost += w[e] * r[q] * r[q]
  return cost


@njit(cache=True, nogil=True)
def pose_graph_update(s, R, t, dx, var):
  s, R, t = s.copy(), R.copy(), t.copy()
  for k in range(len(s)):
    if var[k] >= 0:
      s[k], R[k], t[k] = perturb(s[k], R[k], t[k], dx[7 * var[k]:7 * var[k] + 7])
  return s, R, t


@njit(cache=True, nogil=True)
def pose_graph_normal(Ji, Jj, r, w, blk, n_blocks, gi, gj, nv):
  '''normal equations of the pose graph: the 7x7 blocks (in the fixed pattern, blk[e] = slots of ii, jj, ij, ji or -1)
  and the gradient (gi / gj: the node's variable index or -1)'''
  H = np.zeros((n_blocks, 7, 7))
  g = np.zeros(7 * nv)
  for e in range(len(w)):
    for k in range(4):
      q = blk[e, k]
      if q < 0:
        continue
      A = Ji if k == 0 or k == 2 else Jj
      B = Ji if k == 0 or k == 3 else Jj
      for p in range(7):
        for c in range(7):
          acc = 0.0
          for m in range(7):
            acc += A[e, m, p] * B[e, m, c]
          H[q, p, c] += w[e] * acc
    for k, (J, v) in enumerate(((Ji, gi[e]), (Jj, gj[e]))):
      if v < 0:
        continue
      for p in range(7):
        acc = 0.0
        for m in range(7):
          acc += J[e, m, p] * r[e, m]
        g[7 * v + p] += w[e] * acc
  return H, g


def pose_graph(S_cw, edges, fixed, iterations=20):
  '''
  Sim3 pose graph (Levenberg-Marquardt, sparse normal equations). S_cw: (n, 4, 4) world-to-camera Sim3 per node,
  fixed: (n,) bool, edges: (i, j, S_ij, weight) with S_ij ~ S_i inv(S_j) the measured relative transform
  returns the optimized S_cw
  '''
  import scipy.sparse.linalg
  s = geometry.sim3_scale(S_cw)
  R = S_cw[:, :3, :3] / s[:, None, None]
  t = S_cw[:, :3, 3].copy()
  I = np.array([e[0] for e in edges], np.int64)
  J = np.array([e[1] for e in edges], np.int64)
  M = np.array([e[2] for e in edges])
  w = np.array([e[3] for e in edges], np.float64)
  sm = geometry.sim3_scale(M)
  Rm, tm = M[:, :3, :3] / sm[:, None, None], M[:, :3, 3].copy()
  var = np.where(fixed, -1, np.cumsum(~fixed) - 1).astype(np.int64)
  nv = int((~fixed).sum())
  # fixed sparsity pattern: one 7x7 block per (variable, variable) pair that an edge (or the diagonal) touches
  vi, vj = var[I], var[J]
  pairs = np.r_[np.c_[np.arange(nv), np.arange(nv)], np.c_[vi, vj], np.c_[vj, vi]]
  pairs = pairs[(pairs >= 0).all(axis=1)]
  keys, slot = np.unique(pairs[:, 0] * nv + pairs[:, 1], return_inverse=True)
  bi, bj = keys // nv, keys % nv
  lookup = dict(zip(keys.tolist(), range(len(keys))))
  key = lambda a, b: np.array([lookup.get(int(x) * nv + int(y), -1) if x >= 0 and y >= 0 else -1 for x, y in zip(a, b)])
  blk = np.c_[key(vi, vi), key(vj, vj), key(vi, vj), key(vj, vi)].astype(np.int64)
  rr, cc = np.meshgrid(np.arange(7), np.arange(7), indexing='ij')
  rows = (7 * bi)[:, None, None] + rr
  cols = (7 * bj)[:, None, None] + cc
  pattern = scipy.sparse.csc_matrix((np.arange(rows.size, dtype=np.float64) + 1, (rows.ravel(), cols.ravel())),
                                    shape=(7 * nv, 7 * nv))
  perm = pattern.data.astype(np.int64) - 1      # csc data position -> flat block entry
  diag = np.flatnonzero(pattern.indices == np.repeat(np.arange(7 * nv), np.diff(pattern.indptr)))
  lam = 1e-4
  r, Ji, Jj, cost = pose_graph_linearize(s, R, t, I, J, sm, Rm, tm, w, var)
  for _ in range(iterations):
    Hb, g = pose_graph_normal(Ji, Jj, r, w, blk, len(keys), vi, vj, nv)
    H = pattern.copy()
    H.data = Hb.reshape(-1)[perm]
    d = H.data[diag].copy()
    improved = False
    while lam < 1e8:
      H.data[diag] = d * (1 + lam) + 1e-12
      # (SPD: symmetric mode, no pivoting)
      dx = scipy.sparse.linalg.splu(H, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                                    options=dict(SymmetricMode=True)).solve(-g)
      s_new, R_new, t_new = pose_graph_update(s, R, t, dx, var)
      new_cost = pose_graph_cost(s_new, R_new, t_new, I, J, sm, Rm, tm, w)
      if np.isfinite(new_cost) and new_cost < cost:
        improved = cost - new_cost > 1e-8 * cost
        s, R, t = s_new, R_new, t_new
        lam = max(lam / 5, 1e-8)
        r, Ji, Jj, cost = pose_graph_linearize(s, R, t, I, J, sm, Rm, tm, w, var)
        break
      lam *= 5
    if not improved:
      break
  out = np.tile(np.eye(4), (len(s), 1, 1))
  out[:, :3, :3] = s[:, None, None] * R
  out[:, :3, 3] = t
  return out