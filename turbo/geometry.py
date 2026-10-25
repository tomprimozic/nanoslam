'''
Conventions: T_cw is a 4x4 world-to-camera transform (x_c = R x_w + t); T_wc = inv(T_cw) is the camera pose.
Sim3 transforms are 4x4 matrices with the scale folded into the rotation block (x -> s R x + t).
'''
import numpy as np


def make_pose(R, t):
  T = np.eye(4)
  T[:3, :3], T[:3, 3] = R, t
  return T


def inv_pose(T):
  R, t = T[..., :3, :3], T[..., :3, 3]
  out = np.zeros_like(T)
  out[..., :3, :3] = np.swapaxes(R, -1, -2)
  out[..., :3, 3] = -np.einsum('...ji,...j->...i', R, t)
  out[..., 3, 3] = 1
  return out


def orthonormalize(T):
  '''project the rotation block back onto SO(3) (poses updated by repeated multiplication drift off it)'''
  U, _, VT = np.linalg.svd(T[:3, :3])
  T = T.copy()
  T[:3, :3] = U @ VT
  return T


def transform(T, X):
  return X @ T[:3, :3].T + T[:3, 3]


def camera_center(T_cw):
  return -np.einsum('...ji,...j->...i', T_cw[..., :3, :3], T_cw[..., :3, 3])


def project(K, T_cw, X):
  '''pixels and depths of world points X (n, 3)'''
  Xc = transform(T_cw, X)
  z = Xc[:, 2]
  with np.errstate(divide='ignore', invalid='ignore'):
    uv = Xc[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
  return uv, z


def unproject(K, uv):
  '''pixels -> normalized image coordinates (z = 1)'''
  return np.c_[(uv - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]], np.ones(len(uv))]


def rotation_angle(R):
  return np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def sim3(s, R, t):
  S = np.eye(4)
  S[:3, :3], S[:3, 3] = s * R, t
  return S


def sim3_scale(S):
  return np.cbrt(np.linalg.det(S[..., :3, :3]))


def sim3_inv(S):
  s = sim3_scale(S)
  R = S[:3, :3] / s
  return sim3(1 / s, R.T, -(R.T @ S[:3, 3]) / s)


def sim3_to_se3(S):
  '''the SE3 pose of a camera whose world-to-camera Sim3 is S (x_c = s R x_w + t ~ R x_w + t / s up to scale)'''
  s = sim3_scale(S)
  return make_pose(S[:3, :3] / s, S[:3, 3] / s)


def umeyama(A, B):
  '''least-squares s, R, t with A ~ s R B + t (points in rows)'''
  n, m = A.shape
  EA, EB = A.mean(axis=0), B.mean(axis=0)
  H = (A - EA).T @ (B - EB) / n
  U, D, VT = np.linalg.svd(H)
  S = np.eye(m)
  S[-1, -1] = np.sign(np.linalg.det(U) * np.linalg.det(VT))
  R = U @ S @ VT
  s = np.trace(np.diag(D) @ S) / np.mean(np.sum((B - EB) ** 2, axis=1))
  return s, R, EA - s * R @ EB
