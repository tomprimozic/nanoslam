'''
The map: flat numpy tables of keyframes, points and observations.

  keyframes     pose T_cw, frame index, parent (spanning tree), segment, alive flag; pyramids in `Map.images`
  points        position, state, reference keyframe + pixel (where its patch comes from), observation counts
  observations  (keyframe, point, pixel) triples, with an alive flag; compacted when half of them are dead

Every detected feature is a point from the start: a *candidate* collects keyframe observations while it is tracked
and becomes *mapped* once multi-view triangulation succeeds (enough parallax), so no observations are thrown away.
'''
import numpy as np
import scipy.sparse


CANDIDATE, MAPPED, DEAD = 0, 1, -1


def group_starts(sorted_keys):
  '''mask of the first element of every run of equal keys'''
  return np.r_[True, sorted_keys[1:] != sorted_keys[:-1]][:len(sorted_keys)]


class Table:
  '''named columns of equal length; rows are appended (amortized doubling) and never removed (use a flag)'''

  def __init__(self, **columns):     # name=(dtype, shape, fill)
    self._spec = columns
    self._data = {k: np.full((64,) + shape, fill, dtype) for k, (dtype, shape, fill) in columns.items()}
    self.n = 0

  def __getattr__(self, name):
    try:
      return self.__dict__['_data'][name][:self.n]
    except KeyError:
      raise AttributeError(name) from None

  def __len__(self):
    return self.n

  def add(self, n, **values):
    if self.n + n > len(next(iter(self._data.values()))):
      cap = max(2 * (self.n + n), 64)
      for k, (dtype, shape, fill) in self._spec.items():
        new = np.full((cap,) + shape, fill, dtype)
        new[:self.n] = self._data[k][:self.n]
        self._data[k] = new
    ids = np.arange(self.n, self.n + n)
    self.n += n
    for k, v in values.items():
      self._data[k][ids] = v
    return ids

  def compact(self, keep):
    '''drop rows (invalidates row ids)'''
    m = int(keep.sum())
    for k, (dtype, shape, fill) in self._spec.items():
      self._data[k][:m] = self._data[k][:self.n][keep]
      self._data[k][m:self.n] = fill
    self.n = m


class Map:
  def __init__(self):
    self.kfs = Table(T=(np.float64, (4, 4), 0), frame=(np.int64, (), -1), alive=(bool, (), True),
                     parent=(np.int64, (), -1), segment=(np.int64, (), 0), noise=(np.float64, (), 1.0),
                     slot=(np.int64, (), -1))
    self.images = None     # lk.ImageStore of the keyframe pyramids (kfs.slot)
    self.pts = Table(X=(np.float64, (3,), np.nan), state=(np.int8, (), CANDIDATE), ref=(np.int64, (), -1),
                     ref_uv=(np.float64, (2,), np.nan), n_obs=(np.int32, (), 0), first_kf=(np.int64, (), -1),
                     last_kf=(np.int64, (), -1), visible=(np.int32, (), 0), found=(np.int32, (), 0))
    self.obs = Table(kf=(np.int64, (), -1), pt=(np.int64, (), -1), uv=(np.float64, (2,), np.nan),
                     alive=(bool, (), True))

  # --- points & observations
  def add_points(self, kf, uv):
    ids = self.pts.add(len(uv), state=CANDIDATE, ref=kf, ref_uv=uv, first_kf=kf)
    self.add_obs(kf, ids, uv)
    return ids

  def add_obs(self, kf, pt, uv):
    self.obs.add(len(pt), kf=kf, pt=pt, uv=uv)
    np.add.at(self.pts._data['n_obs'], pt, 1)
    self.pts.last_kf[pt] = np.maximum(self.pts.last_kf[pt], kf)

  def remove_obs(self, o):
    o = o[self.obs.alive[o]]
    self.obs.alive[o] = False
    np.subtract.at(self.pts._data['n_obs'], self.obs.pt[o], 1)
    pts = np.unique(self.obs.pt[o])
    self.kill(pts[(self.pts.state[pts] == MAPPED) & (self.pts.n_obs[pts] < 2)])
    lost_ref = self.obs.pt[o][self.obs.kf[o] == self.pts.ref[self.obs.pt[o]]]
    self.rebase(lost_ref[self.pts.state[lost_ref] != DEAD])

  def rebase(self, pts):
    '''move the reference of points to their earliest remaining observation'''
    if len(pts) == 0:
      return
    o = self.obs_of(pts=pts)
    o = o[np.lexsort((self.obs.kf[o], self.obs.pt[o]))]
    first = o[group_starts(self.obs.pt[o])]
    self.pts.ref[self.obs.pt[first]] = self.obs.kf[first]
    self.pts.ref_uv[self.obs.pt[first]] = self.obs.uv[first]
    orphans = np.setdiff1d(pts, self.obs.pt[first])
    self.kill(orphans)

  def kill(self, pts):
    if len(pts) == 0:
      return
    self.pts.state[pts] = DEAD
    mask = np.zeros(len(self.pts), bool)
    mask[pts] = True
    o = np.flatnonzero(self.obs.alive & mask[self.obs.pt])
    self.obs.alive[o] = False
    self.pts.n_obs[pts] = 0

  def obs_of(self, kfs=None, pts=None):
    sel = self.obs.alive.copy()
    if kfs is not None:
      mask = np.zeros(len(self.kfs), bool)
      mask[kfs] = True
      sel &= mask[self.obs.kf]
    if pts is not None:
      mask = np.zeros(len(self.pts), bool)
      mask[pts] = True
      sel &= mask[self.obs.pt]
    return np.flatnonzero(sel)

  def points_of(self, kf, state=MAPPED):
    p = self.obs.pt[self.obs_of(kfs=[kf])]
    return p[self.pts.state[p] == state]

  def covisibility(self, kf):
    '''number of shared mapped points with every keyframe'''
    o = self.obs_of(pts=self.points_of(kf))
    counts = np.bincount(self.obs.kf[o], minlength=len(self.kfs))
    counts[kf] = 0
    return counts

  def covisibility_matrix(self):
    '''(n_kfs, n_kfs) sparse matrix of shared mapped points'''
    o = np.flatnonzero(self.obs.alive & (self.pts.state[self.obs.pt] == MAPPED))
    A = scipy.sparse.csr_matrix((np.ones(len(o)), (self.obs.kf[o], self.obs.pt[o])), shape=(len(self.kfs), len(self.pts)))
    return (A @ A.T).tocsr()

  def merge(self, keep, drop):
    '''fuse points `drop` into `keep` (pairwise); observations of `keep` win where both were seen'''
    o = self.obs_of(pts=drop)
    target = np.full(len(self.pts), -1)
    target[drop] = keep
    new_pt = target[self.obs.pt[o]]
    # (point, keyframe) keys: an observation is a duplicate if `keep` (or another moved observation) has it already
    o_keep = self.obs_of(pts=keep)
    key = new_pt * len(self.kfs) + self.obs.kf[o]
    dup = np.isin(key, self.obs.pt[o_keep] * len(self.kfs) + self.obs.kf[o_keep])
    _, first = np.unique(key, return_index=True)
    dup[np.setdiff1d(np.arange(len(o)), first)] = True
    self.obs.alive[o[dup]] = False
    self.obs.pt[o[~dup]] = new_pt[~dup]
    np.add.at(self.pts._data['n_obs'], new_pt[~dup], 1)
    self.pts.last_kf[keep] = np.maximum(self.pts.last_kf[keep], self.pts.last_kf[drop])
    self.pts.state[drop] = DEAD
    self.pts.n_obs[drop] = 0

  def compact_obs(self):
    if self.obs.alive.mean() < 0.5:
      self.obs.compact(self.obs.alive)
