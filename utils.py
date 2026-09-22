"""Shared latent-state CIC model used by the empirical and simulation notebooks."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp


MODEL_SPEC_VERSION = "K4_POISSON6_MEAN_FIELD_LOG_ONLY_NO_ACTIVITY_CONTROL_V1"
COVARIATE_TRANSFORMS = (
    "raw", "log_only", "normalize_only", "log_normalize",
)

def transition_log_probabilities(
        mean_field, trans_alpha, beta, previous_state=None):
    """Return log transition probabilities under alpha + beta * mean_field."""
    mean_field = np.asarray(mean_field, dtype=np.float64)
    if previous_state is None:
        logits = trans_alpha + np.einsum(
            "...k,vuk->...vu", mean_field, beta, optimize=True)
    else:
        previous_state = np.asarray(previous_state, dtype=np.int64)
        logits = (
            trans_alpha[previous_state]
            + np.einsum(
                "...k,...uk->...u", mean_field, beta[previous_state],
                optimize=True,
            )
        )
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def g_tr(x):
    return np.log1p(np.maximum(x, 0.0))

class LatentStateCICModel:

    def __init__(self, K=4, n_em=25, seed=0, rho_cic=0.6,
                 y_weight=10.0, sigma_b=1.0, reg_eta=0.02, reg_theta=0.05,
                 w_dir_prior=1.1, gam_beta_prior=2.0,
                 use_indiv=False, em_tol=2e-5, cic_channels=("deg", "eig"),
                 emission="poisson", alpha_mode="dim", alpha_cap=10.0,
                 obs_dims=6, mean_field=True, mf_l2=0.01,
                 mf_maxiter=35, covariate_transform="log_only",
                 verbose=False):
        self.K, self.n_em, self.seed = K, n_em, seed
        self.rho_cic = rho_cic
        self.y_weight, self.sigma_b = y_weight, sigma_b
        self.use_indiv, self.em_tol = use_indiv, em_tol
        self.emission = emission         # "poisson" | "nb"
        self.obs_dims = obs_dims         # 6: sent per-layer + recv total
        self.alpha = None                # NB dispersion [K, D_obs]
        self.alpha_mode = alpha_mode     # "dim": alpha shared across states
                                         #   (dispersion modeled, but state
                                         #   discrimination stays mu-driven);
                                         # "state_dim": free per state-dim
        self.alpha_cap = alpha_cap
        self.mean_field = bool(mean_field)
        self.mf_l2 = float(mf_l2)
        self.mf_maxiter = int(mf_maxiter)
        if covariate_transform not in COVARIATE_TRANSFORMS:
            raise ValueError(
                f"covariate_transform must be one of {COVARIATE_TRANSFORMS}"
            )
        self.covariate_transform = covariate_transform
        self.A_init_ = None          # fixed EM initialization only [v,u]
        self.trans_alpha_ = None       # fitted baseline transition logits [v,u]
        self.beta_ = None              # peer-state effects [v,u,k]
        # CIC channels, canonical order deg -> eig -> auth. "deg" carries the
        # gamma/delta mix AND is the only channel through which w_s receives
        # gradient in the M-step (eig/auth are held fixed there), so dropping
        # "deg" leaves w_s and gamma unidentified (they stay at their priors).
        order = [c for c in ("deg", "eig", "auth") if c in cic_channels]
        self.cic_channels = tuple(order)
        self.n_cic = len(self.cic_channels)
        self.cov_mu = self.cov_sd = None   # training transformation statistics
        self.reg_eta, self.reg_theta = reg_eta, reg_theta
        self.w_dir_prior = w_dir_prior   # Dirichlet(alpha) MAP on w_s rows
        self.gam_beta_prior = gam_beta_prior  # Beta(a,a) MAP on gamma: keeps the
                                              # in/out mix interior (the likelihood
                                              # is nearly flat in gamma once
                                              # eig/HITS are in the CIC, so an
                                              # unregularized gamma drifts to a
                                              # 0/1 corner)
        self.verbose = verbose

    # ---------------- data ----------------
    def _prep(self, edges_, LSTR_, presence_mask=None):
        self.T_, self.N_, self.L_ = LSTR_.shape[0], LSTR_.shape[1], LSTR_.shape[2]
        self.present = (np.ones((self.T_, self.N_), dtype=bool)
                        if presence_mask is None
                        else np.asarray(presence_mask, dtype=bool))
        if self.present.shape != (self.T_, self.N_):
            raise ValueError("presence_mask must have shape [T, N]")
        self.OUT = LSTR_[..., 0].astype(np.float64)
        if self.obs_dims == 10:      # per-layer sent AND received (appendix)
            self.X_obs = np.concatenate([LSTR_[..., 0], LSTR_[..., 1]],
                                        -1).astype(np.float64)
        else:                        # 6-dim: per-layer sent + total received
            recv = LSTR_[..., 1].sum(-1, keepdims=True)
            self.X_obs = np.concatenate([LSTR_[..., 0], recv],
                                        -1).astype(np.float64)
        self.D_obs = self.X_obs.shape[-1]
        self.matsT = [[sp.csr_matrix((e["c"][:, l], (e["j"], e["i"])),
                                     shape=(self.N_, self.N_))
                       for l in range(self.L_)] for e in edges_]
        # Directed neighbor pool used by Eq. (1): i considers j a
        # neighbor after i has initiated at least one interaction with j.
        # Weekly binary matrices are accumulated causally as needed; we do
        # not store T cumulative matrices.
        self.neigh_week = []
        for e in edges_:
            g = sp.csr_matrix((np.ones(len(e["i"]), dtype=np.float64),
                               (e["i"], e["j"])),
                              shape=(self.N_, self.N_))
            g.sum_duplicates()
            if g.nnz:
                g.data[:] = 1.0
            self.neigh_week.append(g)

    def _B(self, q):
        B = np.zeros((self.T_, self.N_, self.L_, self.K))
        for t in range(self.T_):
            for l in range(self.L_):
                B[t, :, l, :] = self.matsT[t][l] @ q[t]
        return B

    def _neighbor_props(self, q):
        """Causal mean-field proportions m[t,i,k].

        For the transition into week t, the directed neighbor pool contains
        ties initiated before t (weeks 0,...,t-1), and neighbor states are the
        posterior probabilities at t-1. Users without prior neighbors receive
        a zero vector, so their transition falls back to alpha.
        """
        Tn, Nn, K = q.shape
        out = np.zeros((Tn, Nn, K), dtype=np.float64)
        hist = sp.csr_matrix((Nn, Nn), dtype=np.float64)
        for t in range(1, Tn):
            hist = (hist + self.neigh_week[t - 1]).tocsr()
            hist.sum_duplicates()
            if hist.nnz:
                hist.data[:] = 1.0
            # Only neighbors present at t-1 contribute to the mean field.
            active_deg = np.asarray(
                hist @ self.present[t - 1].astype(float)).ravel()
            raw = hist @ q[t - 1]
            np.divide(raw, active_deg[:, None], out=out[t],
                      where=active_deg[:, None] > 0)
        return out

    @staticmethod
    def _transition_logp(m, trans_alpha, beta):
        """log P(S_t=u | S_t-1=v, m_t-1) for arbitrary leading dims."""
        return transition_log_probabilities(m, trans_alpha, beta)

    def _m_step_transition(self, xi, mfield, trans_alpha, beta):
        """MAP update for alpha and beta using fractional transition counts.

        Destination K-1 and neighbor-state K-1 are reference categories. This
        removes the softmax and compositional non-identifiability. The ridge is
        applied only to beta, on the average transition negative log-likelihood.
        """
        K = self.K
        X = mfield[1:, :, :K - 1].reshape(-1, K - 1)
        new_alpha = np.zeros((K, K), dtype=np.float64)
        new_beta = np.zeros((K, K, K), dtype=np.float64)

        for v in range(K):
            Yv = xi[:, :, v, :].reshape(-1, K)
            mass = Yv.sum(axis=1)
            keep = mass > 1e-12
            Xv, Yv, mass = X[keep], Yv[keep], mass[keep]
            denom = max(float(mass.sum()), 1.0)

            a0 = trans_alpha[v, :K - 1] - trans_alpha[v, K - 1]
            b0 = (beta[v, :K - 1, :K - 1]
                  - beta[v, K - 1, :K - 1][None, :])
            z0 = np.concatenate([a0, b0.ravel()])

            def objective(z):
                a = z[:K - 1]
                b = z[K - 1:].reshape(K - 1, K - 1)
                free = a[None, :] + Xv @ b.T
                logits = np.concatenate(
                    [free, np.zeros((len(free), 1), dtype=free.dtype)], axis=1)
                logp = logits - logsumexp(logits, axis=1, keepdims=True)
                p = np.exp(logp)
                f = -(Yv * logp).sum() / denom                     + 0.5 * self.mf_l2 * np.sum(b * b)
                err = (p * mass[:, None] - Yv) / denom
                ga = err[:, :K - 1].sum(axis=0)
                gb = err[:, :K - 1].T @ Xv + self.mf_l2 * b
                return float(f), np.concatenate([ga, gb.ravel()])

            opt = minimize(objective, z0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=self.mf_maxiter,
                                        ftol=1e-9, gtol=1e-6))
            a = opt.x[:K - 1]
            b = opt.x[K - 1:].reshape(K - 1, K - 1)
            new_alpha[v, :K - 1] = a
            new_beta[v, :K - 1, :K - 1] = b

        return new_alpha, new_beta

    @staticmethod
    def _ema(x, r):
        s = np.empty_like(x); s[0] = x[0]
        for t in range(1, x.shape[0]):
            s[t] = (1 - r) * s[t - 1] + r * x[t]
        return s

    # ---- eigenvector / HITS authority on the effective adjacency matrix ----
    def _collapseT(self, t, wbar):
        """A~_t^T with sender-state weights: column i scaled by wbar[i, l]."""
        S = None
        for l in range(self.L_):
            M = self.matsT[t][l].multiply(wbar[:, l][None, :])
            S = M if S is None else S + M
        return S.tocsr()

    @staticmethod
    def _eig_auth_t(S, iters=20):
        n = S.shape[0]
        x = np.ones(n)
        for _ in range(iters):                       # eigenvector (receiving)
            xn = S @ x; nrm = np.linalg.norm(xn)
            if nrm < 1e-12:
                x = np.zeros(n); break
            x = xn / nrm
        h = np.ones(n); a = np.ones(n)
        for _ in range(iters):                       # HITS authority
            an = S @ h; na = np.linalg.norm(an)
            if na < 1e-12:
                a = np.zeros(n); break
            a = an / na
            hn = S.T @ a; nh = np.linalg.norm(hn)
            if nh < 1e-12:
                break
            h = hn / nh
        mx, ma = x.max(), a.max()
        return (x / mx if mx > 0 else x), (a / ma if ma > 0 else a)

    def _eig_auth(self, q, w):
        """[T,N,2] max-normalized weekly eigenvector/authority, EMA-smoothed."""
        Tn, Nn = q.shape[0], self.N_
        EA = np.zeros((Tn, Nn, 2))
        for t in range(Tn):
            wbar = q[t] @ w                          # [N, L] expected weights
            S = self._collapseT(t, wbar)
            EA[t, :, 0], EA[t, :, 1] = self._eig_auth_t(S)
        return self._ema(EA, self.rho_cic)

    def _covs(self, OUT, B, w, gamma, EA):
        """Return the selected state-specific CIC channels [T,N,K,n_cic]."""
        Oe = self._ema(OUT, self.rho_cic)
        Be = self._ema(B, self.rho_cic)
        ones = np.ones(self.K)
        out_k = np.einsum("tnl,kl->tnk", Oe, w)
        in_n = np.einsum("tnlk,kl->tn", Be, w)[..., None] * ones
        cdeg = gamma * out_k + (1 - gamma) * in_n
        chan = dict(deg=cdeg,
                    eig=EA[..., 0][..., None] * ones,
                    auth=EA[..., 1][..., None] * ones)
        return np.stack([chan[c] for c in self.cic_channels], -1)

    # ---------------- likelihood pieces ----------------
    MU_FLOOR = 0.02   # below this mean, alpha is unidentifiable -> Poisson

    def _log_emis(self, mu, alpha=None):
        """activity emission log-lik: NB(mu, alpha), Poisson when alpha
        is None. Called on the DATA-side instance (self.X_obs) with the
        fitted parameters passed in."""
        mu = np.clip(mu, 1e-6, None)
        X = self.X_obs
        if alpha is None:
            return (X[..., None, :] * np.log(mu) - mu
                    - gammaln(X[..., None, :] + 1)).sum(-1)
        r = 1.0 / np.clip(alpha, 1e-4, None)               # [K, D]
        lp = (gammaln(X[..., None, :] + r) - gammaln(r)
              - gammaln(X[..., None, :] + 1.0)
              + r * np.log(r / (r + mu))
              + X[..., None, :] * np.log(mu / (r + mu)))
        return lp.sum(-1)

    def _covariate_base(self, covs):
        """Apply the selected monotone transformation before any scaling."""
        if self.covariate_transform in ("log_only", "log_normalize"):
            return g_tr(covs)
        return np.asarray(covs, dtype=np.float64)

    def _transform_covariates(self, covs):
        """Transform CIC covariates using statistics learned on training rows."""
        base = self._covariate_base(covs)
        if self.covariate_transform in ("normalize_only", "log_normalize"):
            return (base - self.cov_mu) / self.cov_sd
        return base

    def _transform_derivative(self, raw_value, channel):
        """Derivative of the selected CIC transform with respect to raw CIC."""
        derivative = np.ones_like(raw_value, dtype=np.float64)
        if self.covariate_transform in ("log_only", "log_normalize"):
            derivative /= 1.0 + np.maximum(raw_value, 0.0)
        if self.covariate_transform in ("normalize_only", "log_normalize"):
            derivative /= self.cov_sd[channel]
        return derivative

    def _log_y(self, covs, eta, theta, b):
        z = eta[None, None] + np.einsum("tnkc,c->tnk",
                                        self._transform_covariates(covs), theta) \
            + b[None, :, None]
        p = expit(z)
        Yf = self.Yfit[..., None]
        ll = Yf * np.log(p + 1e-12) + (1 - Yf) * np.log(1 - p + 1e-12)
        return ll * self.ymask[..., None]  # retained evaluation employee-weeks

    def _fb(self, lp, trans_alpha, beta, pi, mfield):
        """Forward-backward over each employee's retained contiguous span."""
        Tn, Nn, K = lp.shape
        present = self.present
        lt = self._transition_logp(mfield[1:], trans_alpha, beta)
        lal = np.zeros((Tn, Nn, K), dtype=np.float64)
        start = present.copy()
        start[1:] &= ~present[:-1]
        cont = present[1:] & present[:-1]

        if start[0].any():
            lal[0, start[0]] = np.log(pi + 1e-12) + lp[0, start[0]]
        for t in range(1, Tn):
            if start[t].any():
                lal[t, start[t]] = np.log(pi + 1e-12) + lp[t, start[t]]
            if cont[t - 1].any():
                idx = cont[t - 1]
                lal[t, idx] = lp[t, idx] + logsumexp(
                    lal[t - 1, idx, :, None] + lt[t - 1, idx], axis=1)

        lbe = np.zeros((Tn, Nn, K), dtype=np.float64)
        for t in range(Tn - 2, -1, -1):
            if cont[t].any():
                idx = cont[t]
                lbe[t, idx] = logsumexp(
                    lt[t, idx] + lp[t + 1, idx, None]
                    + lbe[t + 1, idx, None], axis=2)

        q = np.zeros((Tn, Nn, K), dtype=np.float64)
        lg = lal + lbe
        q[present] = np.exp(
            lg[present] - logsumexp(lg[present], axis=1, keepdims=True))

        xi = np.zeros((max(Tn - 1, 0), Nn, K, K), dtype=np.float64)
        for t in range(Tn - 1):
            if cont[t].any():
                idx = cont[t]
                lxi = (lal[t, idx, :, None] + lt[t, idx]
                       + lp[t + 1, idx, None] + lbe[t + 1, idx, None])
                lxi -= logsumexp(lxi, axis=(1, 2), keepdims=True)
                xi[t, idx] = np.exp(lxi)

        end = present.copy()
        end[:-1] &= ~present[1:]
        ll = float(logsumexp(lal[end], axis=1).sum())
        return q, xi, ll

    # ---------------- EM (train weeks only) ----------------
    def fit(self, edges_tr, LSTR_tr, Y_tr, y_user_mask=None,
            presence_mask=None):
        rs = np.random.RandomState(self.seed)
        self._prep(edges_tr, LSTR_tr, presence_mask)
        K, L, Nn, Tn = self.K, self.L_, self.N_, self.T_
        C = self.n_cic
        self.Cx = C
        self.cov_mu, self.cov_sd = np.zeros(C), np.ones(C)
        self.Yfit = Y_tr.astype(np.float64)
        user_mask = (np.ones(Nn, dtype=bool) if y_user_mask is None
                     else np.asarray(y_user_mask, dtype=bool))
        self.ymask = self.present & user_mask[None, :]

        from sklearn.cluster import KMeans
        rows = self.X_obs[self.present]
        samp = rows[rs.choice(len(rows), min(30000, len(rows)), replace=False)]
        km = KMeans(K, n_init=4, random_state=self.seed).fit(np.log1p(samp))
        lam = np.maximum(np.expm1(km.cluster_centers_), 1e-4)
        lam = lam[np.argsort(lam.sum(1))] * (1 + 0.05 * rs.rand(K, self.D_obs))

        # Initialization only; fitted transitions use trans_alpha + beta * m.
        A_init = np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (0.95 - 0.05 / (K - 1))
        A_init /= A_init.sum(1, keepdims=True)
        self.A_init_ = A_init.copy()
        trans_alpha = np.log(A_init + 1e-12)
        trans_alpha -= trans_alpha[:, -1:]
        beta = np.zeros((K, K, K), dtype=np.float64)
        pi = np.full(K, 1 / K)
        alpha = (np.full((K, self.D_obs), 0.5)
                 if self.emission == "nb" else None)
        w = np.full((K, L), 1.0 / L)
        gamma = 0.5
        # Base rate over retained outcome employee-weeks only.
        if not self.ymask.any():
            raise ValueError("No observed outcome rows in the training span.")
        p0 = float(np.clip(self.Yfit[self.ymask].mean(), 1e-6, 1 - 1e-6))
        self.base = float(np.log(p0 / (1 - p0)))
        eta = self.base + 0.1 * rs.randn(K)
        theta = np.full(C, 0.05)                 # global CIC coefficients
        b = np.zeros(Nn)
        q = np.zeros((Tn, Nn, K), dtype=np.float64)
        q[self.present] = 1.0 / K

        ll_prev = None
        self.em_iters_, self.loglik_ = 0, None
        for it in range(self.n_em):
            B = self._B(q)
            EA = self._eig_auth(q, w)            # refresh eig/HITS from q, w
            covs = self._covs(self.OUT, B, w, gamma, EA)
            lp = self._log_emis(lam, alpha) \
                + self.y_weight * self._log_y(covs, eta, theta, b)
            mfield = self._neighbor_props(q)
            q, xi, ll = self._fb(lp, trans_alpha, beta, pi, mfield)
            starts = self.present.copy()
            starts[1:] &= ~self.present[:-1]
            pi = q[starts].mean(0).clip(1e-8)
            pi /= pi.sum()
            trans_alpha, beta = self._m_step_transition(
                xi, mfield, trans_alpha, beta)
            # Baseline transition probabilities when the mean-field term is zero.
            baseline_transition = np.exp(trans_alpha - logsumexp(
                trans_alpha, axis=1, keepdims=True))
            wq = q.reshape(-1, K)
            rows_ = self.X_obs.reshape(-1, self.D_obs)
            sw_ = wq.sum(0)[:, None].clip(1e-9)
            m1_ = (wq.T @ rows_) / sw_
            lam = m1_.clip(1e-6)
            if self.emission == "nb":                      # moment alpha
                m2_ = (wq.T @ rows_ ** 2) / sw_
                var_ = np.maximum(m2_ - m1_ ** 2, 1e-9)
                if self.alpha_mode == "dim":
                    # shared alpha_d: ratio of pooled within-state excess
                    # variance to pooled mu^2 (stable; keeps states separated
                    # by their means rather than by their dispersions)
                    num = (sw_ * (var_ - m1_)).sum(0)
                    den = np.maximum((sw_ * m1_ ** 2).sum(0), 1e-9)
                    a_d = np.clip(num / den, 1e-4, self.alpha_cap)
                    alpha = np.tile(a_d, (K, 1))
                else:
                    alpha = np.clip(
                        (var_ - m1_) / np.maximum(m1_ ** 2, 1e-9),
                        1e-4, self.alpha_cap)
                alpha[m1_ < self.MU_FLOOR] = 1e-4          # Poisson limit
            B = self._B(q)
            EA = self._eig_auth(q, w)
            w, gamma, eta, theta, b = self._m_step_emission(
                q, B, EA, w, gamma, eta, theta, b)
            self.em_iters_, self.loglik_ = it + 1, ll
            if self.verbose:
                print(f"    EM {it+1}/{self.n_em}  ll={ll:.1f}", flush=True)
            if (ll_prev is not None
                    and abs(ll - ll_prev) <= self.em_tol * abs(ll_prev)):
                break                            # joint log-lik plateau
            ll_prev = ll

        self.lam, self.A_, self.pi_ = lam, baseline_transition, pi
        self.trans_alpha_, self.beta_ = trans_alpha, beta
        self.alpha = alpha
        self.w, self.gamma = w, gamma
        self.eta, self.theta, self.b = eta, theta, b
        return self

    def _m_step_emission(self, q, B, EA, w, gamma, eta, theta, b):
        """L-BFGS with analytic gradients on the expected outcome log-lik.
        C_deg is linear in (w, gamma) with EMA pre-applied; C_eig/C_auth are
        held fixed within the M-step (recomputed each EM iteration)."""
        K, L, Nn = self.K, self.L_, self.N_
        C = self.n_cic
        Oe = self._ema(self.OUT, self.rho_cic)
        Be = self._ema(B, self.rho_cic)
        Yf = self.Yfit[..., None]
        um = self.ymask[..., None]
        n_eff = max(float(self.ymask.sum()), 1.0)
        sc = 1000.0 / n_eff
        prior = sc / (self.sigma_b ** 2)
        lam_dir = self.w_dir_prior - 1.0
        lam_gam = self.gam_beta_prior - 1.0
        ones = np.ones(K)
        chan_fix = dict(eig=EA[..., 0][..., None] * ones,
                        auth=EA[..., 1][..., None] * ones)
        fixed_block = ([chan_fix[c][..., None]
                        for c in self.cic_channels if c != "deg"])
        use_deg = "deg" in self.cic_channels

        use_b = self.use_indiv
        zero_b = np.zeros(Nn)

        # Covariate transformation statistics from the current covariates.
        # Retained outcome employee-weeks only; held fixed within this
        # M-step and reused at prediction time.
        covs_now = self._covs(self.OUT, B, w, gamma, EA)
        base_now = self._covariate_base(covs_now[self.ymask]).reshape(-1, C)
        if self.covariate_transform in ("normalize_only", "log_normalize"):
            self.cov_mu = base_now.mean(0)
            self.cov_sd = base_now.std(0) + 1e-6
        else:
            self.cov_mu = np.zeros(C)
            self.cov_sd = np.ones(C)

        def unpack(v):
            wl = v[:K * L].reshape(K, L)
            w_ = np.exp(wl - wl.max(1, keepdims=True))
            w_ /= w_.sum(1, keepdims=True)
            g_ = expit(v[K * L])                     # gamma in (0, 1)
            e_ = v[K * L + 1:K * L + 1 + K]
            t_ = v[K * L + 1 + K:K * L + 1 + K + C]
            b_ = v[K * L + 1 + K + C:] if use_b else zero_b
            return w_, g_, e_, t_, b_

        def obj_grad(v):
            w_, g_, e_, t_, b_ = unpack(v)
            out_k = np.einsum("tnl,kl->tnk", Oe, w_)
            in_n = np.einsum("tnlk,kl->tn", Be, w_)[..., None] * ones
            cdeg = g_ * out_k + (1 - g_) * in_n
            blocks = ([cdeg[..., None]] if use_deg else []) + fixed_block
            covs = np.concatenate(blocks, -1)
            gcv = self._transform_covariates(covs)
            z = e_[None, None] + np.einsum("tnkc,c->tnk", gcv, t_) + b_[None, :, None]
            p = expit(z)
            ll = (Yf * np.log(p + 1e-12)
                  + (1 - Yf) * np.log(1 - p + 1e-12)) * um
            pen = self.reg_eta * np.sum((e_ - self.base) ** 2) \
                + self.reg_theta * np.sum(t_ ** 2) + 0.5 * prior * np.sum(b_ ** 2) \
                - lam_dir * np.sum(np.log(w_ + 1e-12)) \
                - lam_gam * (np.log(g_ + 1e-12) + np.log(1 - g_ + 1e-12))
            f = -(q * ll).sum() * sc + pen
            dz = -(q * (Yf - p) * um) * sc
            g_e = dz.sum((0, 1)) + 2 * self.reg_eta * (e_ - self.base)
            g_t = np.einsum("tnk,tnkc->c", dz, gcv) + 2 * self.reg_theta * t_
            if use_deg:      # deg is always channel 0 when present
                dcd = dz * t_[0] * self._transform_derivative(
                    cdeg, channel=0)                          # dL/dC_deg
                gw = g_ * np.einsum("tnk,tnl->kl", dcd, Oe) \
                    + (1 - g_) * np.einsum("tn,tnlk->kl", dcd.sum(-1), Be)
                gwl = w_ * (gw - (gw * w_).sum(1, keepdims=True))
                gwl = gwl + lam_dir * (L * w_ - 1.0)
                g_gam = (dcd * (out_k - in_n)).sum() * g_ * (1 - g_) \
                    + lam_gam * (2.0 * g_ - 1.0)
            else:            # no deg channel: w/gamma feel only their priors
                gwl = lam_dir * (L * w_ - 1.0)
                g_gam = lam_gam * (2.0 * g_ - 1.0)
            parts = [gwl.ravel(), [g_gam], g_e, g_t]
            if use_b:
                parts.append(dz.sum((0, 2)) + prior * b_)
            return f, np.concatenate(parts)

        gam0 = float(np.log(gamma / (1 - gamma + 1e-12) + 1e-12))
        v0 = np.concatenate([np.log(w.clip(1e-6)).ravel(), [gam0], eta, theta]
                            + ([b] if use_b else []))
        res = minimize(obj_grad, v0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500))
        return unpack(res.x)

    # ---------------- causal prediction on the full timeline ----------------
    def predict_full(self, edges_all, LSTR_all, Y_all=None, y_user_mask=None,
                     presence_mask=None):
        """Causal forward filtering over each employee's retained span.
        Outcome arguments are accepted only for compatibility and are ignored."""
        m = LatentStateCICModel(
            K=self.K, rho_cic=self.rho_cic, obs_dims=self.obs_dims,
            mean_field=self.mean_field, mf_l2=self.mf_l2,
            mf_maxiter=self.mf_maxiter,
            covariate_transform=self.covariate_transform)
        m._prep(edges_all, LSTR_all, presence_mask)
        Tn, Nn, K = m.T_, m.N_, self.K
        lp = m._log_emis(self.lam, self.alpha)
        qf = np.zeros((Tn, Nn, K))
        p = np.full((Tn, Nn), np.nan)
        hist = sp.csr_matrix((Nn, Nn), dtype=np.float64)
        b_stat = self.b if len(self.b) == Nn else np.zeros(Nn)
        st = {}
        lal = np.zeros((Nn, K), dtype=np.float64)
        for t in range(Tn):
            current = m.present[t]
            start = current if t == 0 else current & ~m.present[t - 1]
            cont = np.zeros(Nn, dtype=bool) if t == 0 else current & m.present[t - 1]
            lal_new = np.zeros((Nn, K), dtype=np.float64)
            if start.any():
                lal_new[start] = np.log(self.pi_ + 1e-12) + lp[t, start]
            if t > 0:
                hist = (hist + m.neigh_week[t - 1]).tocsr()
                hist.sum_duplicates()
                if hist.nnz:
                    hist.data[:] = 1.0
                active_deg = np.asarray(
                    hist @ m.present[t - 1].astype(float)).ravel()
                mf = np.zeros((Nn, K), dtype=np.float64)
                raw = hist @ qf[t - 1]
                np.divide(raw, active_deg[:, None], out=mf,
                          where=active_deg[:, None] > 0)
                la_t = self._transition_logp(
                    mf, self.trans_alpha_, self.beta_)
                if cont.any():
                    lal_new[cont] = logsumexp(
                        lal[cont, :, None] + la_t[cont], axis=1) + lp[t, cont]
            lal = lal_new
            if current.any():
                qf[t, current] = np.exp(
                    lal[current] - logsumexp(lal[current], 1, keepdims=True))
            Bt = np.zeros((Nn, self.L_, K))
            for l in range(self.L_):
                Bt[:, l, :] = m.matsT[t][l] @ qf[t]
            out_k = np.einsum("nl,kl->nk", m.OUT[t], self.w)
            in_n = np.einsum("nlk,kl->n", Bt, self.w)
            wbar = qf[t] @ self.w
            S = m._collapseT(t, wbar)
            eig_t, auth_t = self._eig_auth_t(S)
            r = self.rho_cic
            for key, val in (("o", out_k), ("i", in_n),
                             ("e", eig_t), ("a", auth_t)):
                st[key] = val if t == 0 else (1 - r) * st[key] + r * val
            cdeg = self.gamma * st["o"] + (1 - self.gamma) * st["i"][:, None]
            chan = dict(deg=cdeg,
                        eig=st["e"][:, None] * np.ones(K),
                        auth=st["a"][:, None] * np.ones(K))
            covs_t = np.stack([chan[c] for c in self.cic_channels], -1)
            z = self.eta[None] + np.einsum("nkc,c->nk",
                                           self._transform_covariates(covs_t),
                                           self.theta) \
                + b_stat[:, None]
            pY = expit(z)
            p[t, current] = (qf[t, current] * pY[current]).sum(-1)
        self.qf_ = qf
        return p, qf

# backward-compatible alias used by other cells
DegreeOnlyLatentStateModel = LatentStateCICModel

__all__ = [
    "LatentStateCICModel", "g_tr", "transition_log_probabilities",
    "MODEL_SPEC_VERSION", "COVARIATE_TRANSFORMS",
]
