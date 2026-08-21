#!/usr/bin/env python3
"""
Trap-weight gradient inversion with iterative sample subtraction
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import warnings
warnings.filterwarnings("ignore")

import contextlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.preprocessing import MinMaxScaler
# ------------------------------------------------------------------ constants
DATABASES = {"cifar10": (32*32*3, 10), "emnist": (28*28, 62), "fashion_mnist": (28*28, 10), "mnist": (28*28, 10), "cifar100": (32*32*3, 100), 
  "svhn": (32*32*3, 10), 'harus': (561,6)} #harus for tabular data
L2_DIST = 0.01                  # exact-recovery scoring threshold (post-attack)
SIGMA = 0.5                     # weight magnitude scale
SEED = 23
DTYPE = tf.float64              # residue-scale separation needs float64

ATTACK_DEVICE = '/CPU:0'

#to use GPU, the following lines are needed to ensure deterministic behavior.
#float32 error can affect recall, so it needs to be disabled.
"""
ATTACK_DEVICE = '/GPU:0'  # for speed, but the cascade may differ from the report
tf.config.experimental.enable_tensor_float_32_execution(False) 
tf.config.experimental.enable_op_determinism()
TF_DETERMINISTIC_OPS=1
"""

def _device():
  return tf.device(ATTACK_DEVICE) if ATTACK_DEVICE else contextlib.nullcontext()

def load_data(dataset, B, train = True):
  """Load data for the specified dataset."""
  split = 'train' if train else 'test'
  if dataset in ("cifar10", "cifar100", "fashion_mnist", "mnist"):
    x,y = _load_keras_subset(dataset, train, B)
  elif dataset == "emnist":
    x, y = _load_tfds_subset('emnist/byclass', split, B)
    # tfds returns (28,28,1); squeeze to match Keras MNIST shape (28,28)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
  elif dataset == "svhn":
    split = 'train' if train else 'test'
    x, y = _load_tfds_subset('svhn_cropped', split, B)
  elif dataset == 'harus':
    x,y = _load_harus_subset(split,B)
  else:
      raise ValueError(f"Unsupported dataset: {dataset}")

  return x,y

def _load_keras_subset(dataset_name, train , B):
    """Load from tf.keras.datasets"""
    if dataset_name == "cifar10":
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    elif dataset_name == "fashion_mnist":
        (x, y), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == "mnist":
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        raise ValueError(f"Unknown keras dataset: {dataset_name}")
    x, y = x[:B], y[:B]
    x_test, y_test = x_test[:B], y_test[:B]
    if train:
      return x.astype(np.float64) / 255.0, y.flatten().astype(int)
    else:
      return x_test.astype(np.float64) / 255.0, y_test.flatten().astype(int)

def _load_tfds_subset(dataset_name, split, B):
    """split: train or test."""
    split = f"{split}[:{B}]"
    ds = tfds.load(dataset_name, split=split, as_supervised=True)
    x, y = [], []
    for img, label in tfds.as_numpy(ds):
        x.append(img)
        y.append(label)
    
    x = np.array(x, dtype=np.float64) / 255.0
    y = np.array(y).flatten().astype(int)
    return x, y

def _load_harus_subset(split,B):
  scaler = MinMaxScaler()
  x_train = np.loadtxt("UCI HAR Dataset/train/X_train.txt",dtype=np.float64)
  y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int) - 1
  x_train = scaler.fit_transform(x_train)
  if(split == 'train'):
    return x_train[:B],y_train[:B]
  else:
    x_test  = np.loadtxt("UCI HAR Dataset/test/X_test.txt",dtype=np.float64)
    y_test  = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int) - 1
    x_test = scaler.transform(x_test)
    return x_test[:B],y_test[:B]

# --------------------- weight construction --------------------------------------

def trap_column(n, rng, s, sigma=SIGMA):
  """One column of W1 (independent construction). s = 1 is plain N(0, sigma)."""
  w = -np.abs(rng.normal(0.0, sigma, n))
  flip = rng.random(n) < 0.5
  w[flip] = -s * w[flip]
  return w


def mirrored_column(n, rng, s, sigma=SIGMA):
  """Published Algorithm 1: z+ = -s * z-, shuffled into place."""
  half = n // 2
  z_minus = -np.abs(rng.normal(0.0, sigma, half))
  col = np.concatenate([z_minus, -s * z_minus])
  rng.shuffle(col)
  if len(col) < n:                      # odd n: one extra negative entry
    col = np.append(col, -abs(rng.normal(0.0, sigma)))
  return col


def make_W1(rng, s, n_neurons, n_features, sigma=SIGMA, mirrored=False):
  W = np.empty((n_features, n_neurons))
  for j in range(n_neurons):
    col = mirrored_column if mirrored else trap_column
    W[:, j] = col(n_features, rng, s, sigma)
  return W


# --------------- LT-code -----------------------------
def robust_soliton_degrees(rng, n_neurons, B, c=0.05, delta=0.1):
  """
  Sample n_neurons check-node degrees from Luby's Robust Soliton.
  """
  d = np.arange(1, B + 1, dtype=np.float64)
  rho = np.where(d == 1.0, 1.0 / B, 1.0 / (d * (d - 1.0)))
  if(c==0): #ideal soliton
    return rng.choice(np.arange(1, B + 1), size=n_neurons, p=rho)
  R = c * np.log(B / delta) * np.sqrt(B)
  k = int(round(B / R))                      
  k = min(max(k, 2), B)
  tau = np.where(d < k, R / (d * B), 0.0)
  tau[k - 1] = R * np.log(R / delta) / B      
  beta = rho + tau
  beta /= beta.sum()
  return rng.choice(np.arange(1, B + 1), size=n_neurons, p=beta)


def calibrate_biases(W1, calib_x, degrees):
  """for when the server has access to some data. calculate b_j such that neuron j fires 
  on exactly degrees[j] samples of calib_x.
  Neuron j fires iff z_j + b_j > 0.  Sorting row j's pre-activations and
  placing the threshold midway between the d_j-th and (d_j+1)-th largest
  makes its activation set the d_j closest samples.
  """

  calib_x = np.asarray(calib_x, dtype=np.float64)
  input_dim = calib_x.shape[1]
  B, N = calib_x.shape[0], W1.shape[1]
  z = np.asarray(calib_x, dtype=np.float64) @ np.asarray(W1, dtype=np.float64)

  degrees = np.asarray(degrees)
  b = np.empty(N, dtype=np.float64)
  zs = np.sort(z, axis=0)                     # ascending per column
  for j in range(N):
    d = int(min(max(degrees[j], 1), B))
    hi = zs[B - d, j]                         # smallest z of the active set
    lo = zs[B - d - 1, j] if B - d - 1 >= 0 else hi - 1.0
    b[j] = -0.5 * (hi + lo)
  return b


# ------------------------------------------------------------------ model

def analytic_biases(W1, degrees, B):
  """Data-free counterpart of calibrate_biases. in Trap-Biases section of the report.
  """
  from statistics import NormalDist
  W = np.asarray(W1, dtype=np.float64)
  mu = 0.5 * W.sum(axis=0)
  sd = np.sqrt((W ** 2).sum(axis=0) / 12.0)
  nd = NormalDist()
  degrees = np.asarray(degrees, dtype=np.float64)
  q = np.array([
      nd.inv_cdf(1.0 - d/B) if (d != B) 
      else 3 # 3 standard deviations above the mean for d==B, to avoid error ( cdf(3) arppoximately 0.9986 )
      for d in degrees
  ])
  return -(mu + sd * q)


def build_model(input_dim, num_classes, n_neurons=1000,
                mode='None',mirrored = False, s=1.0, sigma=SIGMA, seed=SEED,
                downstream=None, soliton=(0.05, 0.1), B=None, calib_x=None):
  """The server's model: Dense(n_neurons) -> ReLU -> `downstream` -> logits.

  mode selects the first-layer construction: soliton_free, soliton_data, or None
  mirrored selects the first layer weight construction: mirrored or independent.
  `s` is meaningful only for 'independent' / 'mirrored';
  the soliton modes set activation fractions per row through the bias, so s is ignored there.

  `downstream` is a callable mapping the ReLU activation tensor to the logit
  tensor; it may contain anything differentiable.  Default: one Dense layer
  to `num_classes` logits (Glorot uniform from the same generator).

  Returns a TrapModel.  Only the first layer's structure is fixed; swap
  `downstream` to change the rest of the architecture.
  """
  rng = np.random.default_rng(seed)
  if mode == 'trap-weights':
    W1 = make_W1(rng, s, n_neurons, input_dim, sigma, mirrored)
    b1 = np.zeros(n_neurons)
  elif mode in ('soliton_free', 'soliton_data'):
    if calib_x is not None:
      B = len(calib_x)
    if B is None:
      raise ValueError("soliton modes need B (anticipated batch size), "
                       "either directly or via calib_x")
    W1 = make_W1(rng, 1.0, n_neurons, input_dim, sigma, mirrored)
    deg = robust_soliton_degrees(rng, n_neurons, B, *soliton)
    if mode == 'soliton_data':
      if calib_x is None:
        raise ValueError("mode='soliton_data' requires calib_x")
      # Ensure calib_x is shaped (B, input_dim). Accept image tensors
      # like (B, H, W, C) and flatten them here for calibration.
      calib_arr = np.asarray(calib_x)
      if calib_arr.ndim != 2 or calib_arr.shape[1] != input_dim:
        try:
          calib_arr = calib_arr.reshape(B, input_dim)
        except Exception:
          raise ValueError(
              f"calib_x has unexpected shape {calib_arr.shape}; "
              f"cannot reshape to ({{B}}, {{input_dim}})")
      b1 = calibrate_biases(W1, calib_arr, deg)
    else:
      b1 = analytic_biases(W1, deg, B)
  else:
    raise ValueError(f"unknown mode {mode!r}")

  if downstream is None:
    limit = np.sqrt(6.0 / (n_neurons + num_classes))
    W2 = rng.uniform(-limit, limit, size=(n_neurons, num_classes))
    b2 = np.zeros(num_classes)
    W2c, b2c = tf.constant(W2, DTYPE), tf.constant(b2, DTYPE)
    downstream = lambda a: a @ W2c + b2c

  return TrapModel(W1, b1, downstream, num_classes)


class TrapModel:
  """
  weights + forward/autodiff primitives the attack needs.
  """

  def __init__(self, W1, b1, downstream, num_classes):
    with _device():
      self.W1 = tf.constant(W1, DTYPE)
      self.b1 = tf.constant(b1, DTYPE)
    self.downstream = downstream
    self.num_classes = num_classes

  # ---- primitives ----------------------------------------------------

  def logits_from_z(self, z):
    """u(z): the part of the network downstream of the first ReLU."""
    return self.downstream(tf.nn.relu(z))

  def forward(self, x):
    """z (first-layer pre-activation) and logits u for samples x."""
    x = tf.convert_to_tensor(x, DTYPE)
    z = x @ self.W1 + self.b1
    return z, self.logits_from_z(z)

  # ---- autodiff: the three passes of Section 2 ------------------------
  def batch_gradient(self, x, y, batch_size):
    """
    Gradient of (1/B) * sum_i CE(x_i, y_i) w.r.t. (W1, b1), one batched backward
    pass for the whole group.
    """
    with _device():
      x = tf.convert_to_tensor(x, DTYPE)
      y = tf.constant(np.asarray(y), dtype=tf.int64)
      W1, b1 = self.W1, self.b1
      with tf.GradientTape() as tape:
        tape.watch([W1, b1])
        u = self.logits_from_z(tf.nn.relu(x @ W1 + b1))
        loss = (tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y, logits=u))
                / tf.cast(batch_size, DTYPE))
      return tape.gradient(loss, [W1, b1])

  def JT_p(self, x):
    """J_i^T p_i, ONE backward-mode pass (Section 2.2).

    Backpropagates the cotangent p_i from the logits: a derivative of the
    scalar p^T u, NOT of any loss, so no label is involved.
    Returns (J^T p, p, z) so the caller reuses the forward pass.
    """
    with _device():
      x = tf.convert_to_tensor(x, DTYPE)
      with tf.GradientTape() as tape:
        tape.watch(x)
        z = x @ self.W1 + self.b1
        u = self.logits_from_z(z)
        p = tf.nn.softmax(u)
        scalar = tf.reduce_sum(tf.stop_gradient(p) * u, axis=1)
      return tape.gradient(scalar, z), p, z       # (n, N), (n, C), (n, N)

  def jacobian_rows(self, z, rows):
    """
    J_i[j, :] for the given rows, one forward-mode pass per row
    """
    z = tf.convert_to_tensor(z, DTYPE)
    if z.shape.rank == 2:
      assert z.shape[0] == 1, "jacobian_rows is per-sample"
      z = z[0]
    if not len(rows):
      return tf.zeros((0, self.num_classes), DTYPE)
    return self.jvp_rows(
        tf.repeat(z[None, :], len(rows), axis=0), rows)

  def jvp_rows(self, primals, rows):
    """
    Batched forward-mode JVPs
    """
    rows = np.asarray(rows)
    with _device():
      primals = tf.convert_to_tensor(primals, DTYPE)
      n = int(primals.shape[1])
      tangents = tf.one_hot(rows, n, dtype=DTYPE)
      with tf.autodiff.ForwardAccumulator(primals=primals,
                                          tangents=tangents) as acc:
        u = self.logits_from_z(primals)
      return acc.jvp(u)                           # (M, C)


# ------------------------------------------------------- victim (one round)

def victim_gradient(model, x, y):
  """
  The client computes one batch-averaged gradient and returns it.
  """
  with _device():
    x = tf.convert_to_tensor(x, DTYPE)
    y = tf.constant(np.asarray(y), dtype=tf.int64)
    B = tf.cast(x.shape[0], DTYPE)
    W1, b1 = model.W1, model.b1
    with tf.GradientTape() as tape:
      tape.watch([W1, b1])
      u = model.logits_from_z(tf.nn.relu(x @ W1 + b1))
      loss = tf.reduce_sum(
          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=u)) / B
    gw, gb = tape.gradient(loss, [W1, b1])
  return gw.numpy(), gb.numpy()


def build_problem(model, x_all, y_all, B):
  """One FedSGD round: the victim batches B samples and returns a gradient."""
  input_dim = int(model.W1.shape[0])
  x = x_all[:B].reshape(B, input_dim)
  y = np.asarray(y_all[:B]).flatten().astype(int)
  gw, gb = victim_gradient(model, x, y)
  return {'x': x, 'y': y, 'B': B, 'model': model, 'gw': gw, 'gb': gb}


# ------------------------------------------------------------------ attack

def ratio_columns(res_w, res_b, tol=1e-12):
  """dW[:, j] / db[j] for every valid row"""
  with np.errstate(divide='ignore', invalid='ignore'):
    r = res_w * (1.0 / res_b)
  live = (~np.isnan(r).any(axis=0)
          & ~np.isinf(r).any(axis=0)
          & (np.abs(res_b) > tol))
  return r[:, live].T, np.where(live)[0]


class IterativeSubtractionAttack:
  """
  model       : TrapModel (the server's own weights)
  batch_size  : the victim's B
  ratio_tol   : |h[j]| threshold for a row to be considered live (tau)
  cert_tol    : residual-bias fit tolerance.
  dedup_tol   : two ratio columns closer than this are the same sample
  max_iters   : safety cap on the cascade
  """

  def __init__(self, model, batch_size, ratio_tol=1e-10, cert_tol=1e-8,
               dedup_tol=1e-6, max_iters=30):
    self.model = model
    self.B = batch_size
    self.ratio_tol = ratio_tol
    self.cert_tol = cert_tol
    self.dedup_tol = dedup_tol
    self.max_iters = max_iters

  # ---- the certificate --------------------------

  def _row_fit(self, Jp, Jrows, rows, res_b):
    """ 
    a linear combination has no single sample behind it and fits no row.
    """
    rows = np.asarray(rows)
    pred = (Jp[rows][:, None] - Jrows) / self.B   # (|S|, C) predicted h[j]
    obs = res_b[rows][:, None]                    # (|S|, 1)
    r = np.abs(pred - obs)                        # (|S|, C)

    fit_rows, votes = [], []
    for k, j in enumerate(rows):
      l_j = int(np.argmin(r[k]))
      if r[k, l_j] < self.cert_tol:
        fit_rows.append(int(j))
        votes.append((l_j, r[k, l_j]))
    if not fit_rows:
      return False, 0, float(np.min(r)), []

    counts = np.bincount([v[0] for v in votes],
                         minlength=self.model.num_classes)
    top = np.where(counts == counts.max())[0]
    label = int(min(top, key=lambda l: sum(v[1] for v in votes if v[0] == l)))
    eps = float(np.linalg.norm(
        [r[k, label] for k, j in enumerate(rows) if j in fit_rows]))
    return eps < self.cert_tol, label, eps, fit_rows
  # ---- the loop (Section 2.4) ------------------------------------------

  def run(self, gw, gb):
    """Algorithm 1.  Autodiff is batched across candidates: per iteration
    this costs one backward pass for all J^T p, one forward-mode pass for
    all source-row JVPs (stage 1), one forward-mode pass for all surviving
    (candidate, row) JVPs (stage 2), and one backward pass for all
    subtractions, instead of several passes per candidate."""
    m, B = self.model, self.B
    res_w, res_b = gw.copy(), gb.copy()

    recovered = []        # list of dicts: x, label, eps, rows, iteration
    productive = set()    # distinct rows whose ratio column produced an image
    trace = []

    for it in range(self.max_iters):
      cols, src_rows = ratio_columns(res_w, res_b, self.ratio_tol)
      if len(cols) == 0:
        break

      # deduplicate candidates
      keep, alias = [], {}                    # keep position -> [src rows]
      for k in range(len(cols)):
        hit = None
        for q in keep:
          if np.linalg.norm(cols[k] - cols[q]) < self.dedup_tol:
            hit = q
            break
        if hit is not None:
          alias[keep.index(hit)].append(int(src_rows[k]))
          continue
        if any(np.linalg.norm(cols[k] - f['x']) < self.dedup_tol
               for f in recovered):
          productive.add(int(src_rows[k]))    # re-reconstruction of a
          continue                            # certified sample
        alias[len(keep)] = [int(src_rows[k])]
        keep.append(k)
      if not keep:
        trace.append({'iter': it, 'live': len(cols), 'new': 0})
        break
      cands = cols[keep]
      srcs = src_rows[keep]

      # one backward-mode pass for all candidates
      Jp_all, _, z_all = m.JT_p(cands)
      Jp_all, z_all = Jp_all.numpy(), z_all.numpy()

      #one forward-mode pass for all source-row JVPs
      Jsrc = m.jvp_rows(z_all, srcs).numpy()      # (M, C)
      pred0 = (Jp_all[np.arange(len(keep)), srcs][:, None] - Jsrc) / B
      r0 = np.abs(pred0 - res_b[srcs][:, None]).min(axis=1)
      surv = np.where(r0 < self.cert_tol)[0]

      #full certificate for the survivors only
      live = np.where(np.abs(res_b) > self.ratio_tol)[0]
      live_set = set(live.tolist())
      row_lists, flat_z, flat_rows, splits = [], [], [], [0]
      for k in surv:
        firing = set(np.where(z_all[k] > 0)[0].tolist())
        rows_k = sorted((firing & live_set) | {int(srcs[k])})
        row_lists.append(rows_k)
        flat_z.extend([z_all[k]] * len(rows_k))
        flat_rows.extend(rows_k)
        splits.append(splits[-1] + len(rows_k))
      Jrows_all = (m.jvp_rows(np.asarray(flat_z), flat_rows).numpy()
                   if flat_rows else np.zeros((0, m.num_classes)))

      fresh = []
      for i, k in enumerate(surv):
        Jrows_k = Jrows_all[splits[i]:splits[i + 1]]
        ok, label, eps, fit_rows = self._row_fit(
            Jp_all[k], Jrows_k, row_lists[i], res_b)
        if ok:
          fresh.append({'x': cands[k], 'label': label, 'eps': eps,
                        'rows': fit_rows, 'src': int(srcs[k]),
                        'iteration': it})
          productive.update(alias[k])         # all rows that handed over
                                              # this image, duplicates too

      trace.append({'iter': it, 'live': len(cols), 'new': len(fresh)})
      if not fresh:
        break

      # subtract every newly certified sample
      imgs = np.stack([f['x'] for f in fresh])
      labs = [f['label'] for f in fresh]
      dW, db = m.batch_gradient(imgs, labs, B)
      res_w -= dW.numpy()
      res_b -= db.numpy()
      recovered.extend(fresh)

    return {'samples': recovered,
            'B0': len(recovered),
            # G1: productive rows -- every distinct row whose ratio column
            # was a certified image, including duplicate hand-overs of the
            # same sample
            'G1': len(productive),
            'trace': trace,
            'iters': len([t for t in trace if t['new'] > 0])}


# ------------------------------------------------------------------ scoring
# Ground truth enters only here, after the attack, to measure what it
# achieved.

def pairwise_dists(A, B_):
  if len(A) == 0 or len(B_) == 0:
    return np.zeros((len(A), len(B_)))
  a2 = np.einsum('ij,ij->i', A, A)[:, None]
  b2 = np.einsum('ij,ij->i', B_, B_)[None, :]
  d2 = a2 - 2.0 * (A @ B_.T) + b2
  np.maximum(d2, 0.0, out=d2)
  return np.sqrt(d2, out=d2)


def score_attack(result, prob, l2_dist=L2_DIST):
  """
  Recall / label accuracy / certification margins against ground truth.
  Matches each recovered sample to its nearest true batch element.
  """
  x, y, B = prob['x'], prob['y'], prob['B']
  rec = result['samples']
  if not rec:
    return {'recall': 0.0, 'lab_acc': 0.0, 'B0': 0,
            'genuine_eps': [], 'matched': []}
  imgs = np.stack([f['x'] for f in rec])
  d = pairwise_dists(imgs, x)
  matched, lab_ok, eps = [], 0, []
  used = set()
  for k, f in enumerate(rec):
    i = int(np.argmin(d[k]))
    exact = d[k, i] < l2_dist
    matched.append({'rec': k, 'true': i, 'dist': float(d[k, i]),
                    'exact': bool(exact), 'eps': f['eps'],
                    'label': f['label'], 'true_label': int(y[i]),
                    'iteration': f['iteration']})
    if exact and i not in used:
      used.add(i)
      eps.append(f['eps'])
      lab_ok += int(f['label'] == int(y[i]))
  return {'recall': len(used) / B,
          'lab_acc': lab_ok / len(used) if used else 0.0,
          'B0': len(used),
          'genuine_eps': eps,
          'matched': matched}


def attack_baseline(prob):
  """used in When The Curious Abondon Honesty paper, uses ground truth to find singletons"""
  x = prob['x']
  cols, _ = ratio_columns(prob['gw'], prob['gb'])
  d = pairwise_dists(cols, x)
  hit = d < L2_DIST
  return {'G1': int(hit.any(axis=1).sum()) if len(cols) else 0,
          'B0': int(hit.any(axis=0).sum()),
          'recall': int(hit.any(axis=0).sum()) / prob['B']}


def activation_stats(prob):
  m = prob['model']
  z = (prob['x'] @ m.W1.numpy() + m.b1.numpy())
  per = (z > 0).sum(axis=0)
  fired = int((per > 0).sum())
  depth = float(per[per > 0].mean()) if fired else 0.0
  return fired, int((per == 1).sum()), depth


def metric_row(A, G1, B0, B):
  return {'P': G1 / A if A else 0.0, 'R': B0 / B,
          'Y': B0 / A if A else 0.0,
          'rho': G1 / B0 if B0 else float('nan')}
