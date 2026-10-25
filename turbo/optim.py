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
def pose_graph_linearize(s, R, t, I, J, sm, Rm, tm, w, var):
  '''per edge residual and numerical Jacobians w.r.t. local increments of both nodes; also the cost'''
  n = len(I)
  r = np.zeros((n, 7))
  Ji = np.zeros((n, 7, 7))
  Jj = np.zeros((n, 7, 7))
  eps = 1e-6
  cost = 0.0
  for e in range(n):
    i, j = I[e], J[e]
    r[e] = sim3_edge_error(s[i], R[i], t[i], s[j], R[j], t[j], sm[e], Rm[e], tm[e])
    cost += w[e] * np.sum(r[e] ** 2)
    for a in range(7):
      d = np.zeros(7)
      d[a] = eps
      if var[i] >= 0:
        si, Ri, ti = perturb(s[i], R[i], t[i], d)
        Ji[e, :, a] = (sim3_edge_error(si, Ri, ti, s[j], R[j], t[j], sm[e], Rm[e], tm[e]) - r[e]) / eps
      if var[j] >= 0:
        sj, Rj, tj = perturb(s[j], R[j], t[j], d)
        Jj[e, :, a] = (sim3_edge_error(s[i], R[i], t[i], sj, Rj, tj, sm[e], Rm[e], tm[e]) - r[e]) / eps
  return r, Ji, Jj, cost


@njit(cache=True, nogil=True)
def pose_graph_cost(s, R, t, I, J, sm, Rm, tm, w):
  cost = 0.0
  for e in range(len(I)):
    r = sim3_edge_error(s[I[e]], R[I[e]], t[I[e]], s[J[e]], R[J[e]], t[J[e]], sm[e], Rm[e], tm[e])
    cost += w[e] * np.sum(r ** 2)
  return cost


@njit(cache=True, nogil=True)
def pose_graph_update(s, R, t, dx, var):
  s, R, t = s.copy(), R.copy(), t.copy()
  for k in range(len(s)):
    if var[k] >= 0:
      s[k], R[k], t[k] = perturb(s[k], R[k], t[k], dx[7 * var[k]:7 * var[k] + 7])
  return s, R, t


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
  # sparsity pattern of the normal equations (blocks ii, jj, ij, ji)
  blocks = [(a, b) for a, b in [(I, I), (J, J), (I, J), (J, I)]]
  lam = 1e-4
  r, Ji, Jj, cost = pose_graph_linearize(s, R, t, I, J, sm, Rm, tm, w, var)
  for _ in range(iterations):
    rows, cols, vals = [], [], []
    for (A, B), (JA, JB) in zip(blocks, [(Ji, Ji), (Jj, Jj), (Ji, Jj), (Jj, Ji)]):
      ok = (var[A] >= 0) & (var[B] >= 0)
      H = np.einsum('nki,nkj->nij', JA[ok] * w[ok, None, None], JB[ok])
      rr, cc = np.meshgrid(np.arange(7), np.arange(7), indexing='ij')
      rows.append((7 * var[A[ok]])[:, None, None] + rr)
      cols.append((7 * var[B[ok]])[:, None, None] + cc)
      vals.append(H)
    H = scipy.sparse.coo_matrix((np.concatenate(vals, axis=None), (np.concatenate(rows, axis=None),
                                np.concatenate(cols, axis=None))), shape=(7 * nv, 7 * nv)).tocsc()
    g = np.zeros(7 * nv)
    for A, JA in [(I, Ji), (J, Jj)]:
      ok = var[A] >= 0
      np.add.at(g, (7 * var[A[ok]])[:, None] + np.arange(7), np.einsum('nki,nk->ni', JA[ok] * w[ok, None, None], r[ok]))
    diag = H.diagonal()
    improved = False
    while lam < 1e8:
      Hd = H + scipy.sparse.diags(lam * diag + 1e-12)
      dx = scipy.sparse.linalg.spsolve(Hd.tocsc(), -g)
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
