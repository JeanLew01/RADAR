import sys

sys.path.append('..')

import numpy as np

"""
3D Evasion Problem (drop-in structure mirroring the Spacecraft code)

State:  x = [x1, x2, x3]^T  (x1, x2: planar position; x3: heading)
Input:  u in R (turn-rate control)
Param:  v > 0 (constant forward speed, allowed to be uncertain for MC)

Continuous dynamics:
    x1_dot = -v + v*cos(x3) + u*x2
    x2_dot =  v*sin(x3)    - u*x1
    x3_dot = -u

Discretization: forward Euler (or swap for RK4 easily)
    x_{k+1} = x_k + dt * f(x_k, u_k) + w_k

This module mirrors the spacecraft API:
  - Model:            predict_mean, predict_mean_linearized, Monte Carlo propagation
  - EvasionProblem:   Planning problem wrapper (similar to SpacecraftProblem)
  - EvasionSimulator: Lightweight simulator (sampling states/controls)
"""


class Model:
    # dimensions
    n_x, y_dim = 3, 3
    n_u, u_dim = 1, 1
    n_params   = 1  # uncertain speed v

    # obstacle params (used by external modules)
    robot_radius = 0.05
    nb_pos_dim   = 2  # only (x1, x2) are positional

    # nominal parameter and uncertainty bounds
    v_nom    = 1.0
    v_deltas = 0.2  # v in [v_nom - v_deltas, v_nom + v_deltas]

    # time step
    dt = 0.1

    # Uncertainty propagation method
    B_UP_method = 'randUP'  # 'randUP' supported; 'robUP' not implemented here

    # Additive disturbances (per-state bounds)
    w_nom    = np.zeros(n_x)
    w_deltas = 1e-3 * np.ones(n_x)

    # Quadratic costs (compatible with previous pipeline)
    quadratic_cost_matrix_controls = 1.0 * np.eye(n_u)
    quadratic_cost_matrix_state    = np.zeros((n_x, n_x))

    # Monte-Carlo parameter stores
    vs_MC = np.zeros(0)        # (N_MC,)
    ws_MC = np.zeros((0, 0, n_x))  # (N_MC, N-1, n_x)

    def __init__(self):
        print('[evasion::__init__] Initializing 3D evasion Model (uncertain v).')
        self.reset()

    def reset(self):
        print('[evasion::reset] resetting parameter ranges.')
        # you can modify v_nom, v_deltas, dt here if needed

    # ---------- costs ----------
    def get_quadratic_costs(self):
        R  = self.quadratic_cost_matrix_controls
        Q  = self.quadratic_cost_matrix_state
        QN = np.zeros([self.n_x, self.n_x])
        return Q, QN, R

    # ---------- continuous-time RHS ----------
    def f_ct(self, x, u, v):
        x = np.asarray(x, float).reshape(-1)
        u = float(np.asarray(u, float).reshape(-1)[0])
        x1, x2, x3 = x

        f1 = -v + v*np.cos(x3) + u*x2
        f2 =  v*np.sin(x3)     - u*x1
        f3 = -u
        return np.array([f1, f2, f3], dtype=float)

    # ---------- discrete-time single step ----------
    def f_dt(self, xk, uk, v, wk):
        # forward Euler + additive disturbance
        return np.asarray(xk, float) + self.dt * self.f_ct(xk, uk, v) + wk

    # ---------- API: mean prediction ----------
    def predict_mean(self, x_k, u_k):
        return self.fnom_dt(x_k, u_k)

    def predict_mean_linearized(self, x_k, u_k):
        return self.fnom_dt_dx(x_k, u_k), self.fnom_dt_du(x_k, u_k)

    # ---------- nominal discrete map and Jacobians ----------
    def fnom_dt(self, x_k, u_k):
        return self.f_dt(x_k, u_k, self.v_nom, np.zeros(self.y_dim))

    def fnom_dt_dx(self, x_k, u_k):
        x = np.asarray(x_k, float).reshape(-1)
        u = float(np.asarray(u_k, float).reshape(-1)[0])
        x1, x2, x3 = x
        v = self.v_nom
        # A = df/dx
        A = np.array([
            [ 0.0,  u,     -v*np.sin(x3)],
            [-u,   0.0,     v*np.cos(x3)],
            [ 0.0,  0.0,    0.0         ]
        ], dtype=float)
        return np.eye(self.y_dim) + self.dt * A

    def fnom_dt_du(self, x_k, u_k):
        x = np.asarray(x_k, float).reshape(-1)
        x1, x2, _ = x
        # B = df/du = [x2, -x1, -1]^T
        B = np.array([[x2], [-x1], [-1.0]], dtype=float)
        return self.dt * B

    # ---------- batched propagation ----------
    def f_dt_batched(self, xs_k, us_k, vs, ws_k):
        """
        Inputs:
          xs_k: (N_MC, n_x)
          us_k: (N_MC, n_u) or (N_MC,)
          vs  : (N_MC,)
          ws_k: (N_MC, n_x)
        Output:
          xs_{k+1}: (N_MC, n_x)
        """
        xs_k = np.asarray(xs_k, float)
        us_k = np.asarray(us_k, float).reshape(-1)
        vs   = np.asarray(vs, float).reshape(-1)
        ws_k = np.asarray(ws_k, float)
        out  = np.empty_like(xs_k)
        for i in range(xs_k.shape[0]):
            out[i] = self.f_dt(xs_k[i], us_k[i], vs[i], ws_k[i])
        return out

    def f_dt_dx_batched(self, xs_k, us_k, vs, ws_k):
        """Discrete-time Jacobians wrt x, batched.
        Returns A: (N_MC, n_x, n_x)
        """
        xs_k = np.asarray(xs_k, float)
        us_k = np.asarray(us_k, float).reshape(-1)
        vs   = np.asarray(vs, float).reshape(-1)
        N    = xs_k.shape[0]
        A = np.repeat(np.eye(self.n_x)[None, :, :], N, axis=0)
        for i in range(N):
            x1, x2, x3 = xs_k[i]
            u, v = us_k[i], vs[i]
            A_ct = np.array([[0.0,  u,     -v*np.sin(x3)],
                              [-u,  0.0,    v*np.cos(x3)],
                              [0.0,  0.0,   0.0        ]], dtype=float)
            A[i] = np.eye(self.n_x) + self.dt * A_ct
        return A

    def f_dt_dparams_batched(self, xs_k, us_k, vs, ws_k):
        """
        Derivatives wrt parameter v and disturbances w.
        Returns:
          fs_dv:   (N_MC, n_x)
          fs_dwks: (N_MC, n_x, n_x)   # identity (since additive)
        """
        xs_k = np.asarray(xs_k, float)
        us_k = np.asarray(us_k, float).reshape(-1)
        N    = xs_k.shape[0]
        fs_dv   = np.zeros((N, self.n_x))
        fs_dwks = np.repeat(np.eye(self.n_x)[None, :, :], N, axis=0)
        for i in range(N):
            _, _, x3 = xs_k[i]
            # df/dv (continuous): [-1+cos x3, sin x3, 0]
            df_dv = np.array([-1.0 + np.cos(x3), np.sin(x3), 0.0], dtype=float)
            fs_dv[i] = self.dt * df_dv
        return fs_dv, fs_dwks

    # ---------- Monte-Carlo utilities ----------
    def sample_dynamics_params(self, n_models):
        vmin, vmax = self.v_nom - self.v_deltas, self.v_nom + self.v_deltas
        self.vs_MC = np.random.uniform(low=vmin, high=vmax, size=n_models)
        return self.vs_MC

    def sample_disturbances(self, N, N_MC):
        min_w, max_w = self.w_nom - self.w_deltas, self.w_nom + self.w_deltas
        self.ws_MC = np.random.uniform(low=min_w, high=max_w, size=(N_MC, N, self.n_x))
        return self.ws_MC

    def simulate_batch(self, x_init, X_nom, U_nom,
                       N_MC=10, B_feedback=False, B_resample=True):
        if B_feedback:
            raise NotImplementedError('[evasion::simulate_batch] Feedback not implemented yet.')
        n_x, n_u, N = self.n_x, self.n_u, X_nom.shape[1]
        Xs, Us = np.zeros((N_MC, n_x, N)), np.zeros((N_MC, n_u, N-1))
        if B_resample:
            self.sample_dynamics_params(N_MC)
            self.sample_disturbances(N-1, N_MC)
        Xs[:, :, 0] = np.tile(x_init, (N_MC, 1))
        for k in range(N-1):
            Us[:, :, k] = np.tile(U_nom[:, k], (N_MC, 1))
            Xs[:, :, k+1] = self.f_dt_batched(Xs[:, :, k], Us[:, :, k],
                                               self.vs_MC, self.ws_MC[:, k, :])
        return Xs, Us

    def predict_confsets_monteCarlo(self, Xnom, Unom, A_all, B_all,
                                    shape="rectangular",
                                    N_MC=10, prob=0.9,
                                    B_feedback=False,
                                    B_reuse_presampled_dynamics=False):
        n_x, n_u, N = self.y_dim, self.u_dim, Xnom.shape[1]
        if B_feedback:
            raise NotImplementedError('[evasion::predict_confsets_monteCarlo] Feedback not supported yet.')
        # predict mean on the nominal model
        Xmean = np.zeros((N, n_x))
        Xmean[0, :] = Xnom[:, 0]
        for k in range(N-1):
            Xmean[k+1, :] = self.predict_mean(Xmean[k, :], Unom[:, k])
        # particles
        parts = np.zeros((N_MC, n_x + n_u, N))
        Xs, Us = self.simulate_batch(Xnom[:, 0], Xmean.T, Unom,
                                     N_MC=N_MC, B_feedback=B_feedback, B_resample=True)
        if self.B_UP_method == 'randUP':
            pass
        elif self.B_UP_method == 'robUP':
            raise NotImplementedError('[evasion::predict_confsets_monteCarlo] robUP not implemented.')
        else:
            raise NotImplementedError('[evasion::predict_confsets_monteCarlo] Unknown method.')
        parts[:, :n_x, :]       = Xs
        parts[:, n_x:, :(N-1)]  = np.repeat(Unom[np.newaxis, :], N_MC, axis=0)
        if shape == 'rectangular':
            Deltas       = np.zeros((n_x, N))
            Deltas[:, 0] = np.zeros(n_x)
            for k in range(1, N):
                mean        = Xmean[k, :]
                deltas      = np.repeat(mean[np.newaxis, :], N_MC, axis=0) - parts[:, :n_x, k]
                Deltas[:, k] = np.max(deltas, 0)
            QDs     = Deltas
            QDs_dxu = np.zeros((n_x, N, n_x + n_u, N))
        else:
            raise NotImplementedError('[evasion::predict_confsets_monteCarlo] Only rectangular supported.')
        return QDs, QDs_dxu, parts


# ---------------- Planning problem wrapper (mirrors SpacecraftProblem) ----------------
class EvasionProblem:
    """Goal-set planning for 3D evasion (x1,x2 position; x3 heading)."""
    def __init__(self, x0, xgoal, N, u_abs_max=1.0, pos_bound=10.0):
        n_x = x0.shape[0]
        assert n_x == 3
        # state and input bounds
        x_max = np.array([ pos_bound,  pos_bound,  np.pi])
        x_min = np.array([-pos_bound, -pos_bound, -np.pi])
        u_max = np.array([ u_abs_max ])
        u_min = np.array([-u_abs_max ])
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max

        # spherical (circular) obstacles in (x1,x2); each: [[x1,x2], radius]
        self.sphere_obstacles = [
            [[ 2.0,  2.0], 0.8],
            [[-1.5,  3.0], 0.6],
        ]
        self.poly_obstacles = []

        # uncertainty on initial state ellipsoid (optional, here tiny)
        Q0 = 1e-6 * np.eye(n_x)
        half_widths = 1e-3 * np.ones(n_x)
        X_safe = [x0,    half_widths]
        X_goal = [xgoal, half_widths]
        self.X_safe           = X_safe
        self.X_goal           = X_goal
        self.B_go_to_safe_set = False

        self.x_init = x0
        self.N      = N
        self.Q0     = Q0

# ---------------- Simple simulator for sampling states/controls ----------------
class EvasionSimulator:
    def __init__(self, dt=0.1, v_nom=1.0, v_deltas=0.2,
                 w_nom=np.zeros(3), w_deltas=1e-3*np.ones(3)):
        self.n_x, self.y_dim = 3, 3
        self.n_u, self.u_dim = 1, 1
        self.dt = float(dt)
        self.v_nom = float(v_nom)
        self.v_deltas = float(v_deltas)
        self.w_nom = np.asarray(w_nom, float)
        self.w_deltas = np.asarray(w_deltas, float)

        # defaults
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.state = self.x0.copy()

        # state sampling limits
        pos = 10.0
        self.states_min = np.array([-pos, -pos, -np.pi])
        self.states_max = np.array([ pos,  pos,  np.pi])
        # control sampling limits (turn rate)
        umax = 1.0
        self.control_min = np.array([-umax])
        self.control_max = np.array([ umax])
        self.control_diff_min = 0.1 * self.control_min
        self.control_diff_max = 0.1 * self.control_max

    def reset_state(self):
        self.state = self.x0.copy()

    def f_ct(self, x, u, v=None):
        if v is None:
            v = self.v_nom
        x = np.asarray(x, float).reshape(-1)
        u = float(np.asarray(u, float).reshape(-1)[0])
        x1, x2, x3 = x
        f1 = -v + v*np.cos(x3) + u*x2
        f2 =  v*np.sin(x3)     - u*x1
        f3 = -u
        return np.array([f1, f2, f3], dtype=float)

    def f_dt(self, x_k, u_k, v=None, w_k=None):
        if v is None:
            v = self.v_nom
        if w_k is None:
            w_k = np.zeros(3)
        return np.asarray(x_k, float) + self.dt * self.f_ct(x_k, u_k, v) + w_k

    # sampling utilities
    def sample_states(self, n_samples=()):
        n_samples = ((n_samples,) if isinstance(n_samples, int) else tuple(n_samples))
        lows  = self.states_min
        highs = self.states_max
        return np.random.uniform(low=lows, high=highs, size=n_samples + (self.n_x,))

    def sample_controls(self, n_samples=()):
        n_samples = ((n_samples,) if isinstance(n_samples, int) else tuple(n_samples))
        nom  = np.random.uniform(low=self.control_min, high=self.control_max, size=n_samples + (self.u_dim,))
        diff = np.random.uniform(low=self.control_diff_min, high=self.control_diff_max, size=n_samples + (self.u_dim,))
        u = nom + diff
        return np.clip(u, self.control_min, self.control_max)