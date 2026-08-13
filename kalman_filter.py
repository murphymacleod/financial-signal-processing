"""
kalman_filter.py — From-scratch Kalman filter implementation.

Two variants, both written with plain numpy — no scipy.signal, no Kalman libraries.
The prediction and update equations are written explicitly so the algorithm is
transparent and every step can be discussed.

  KalmanFilter1D : scalar state (price modelled as a random walk)
  KalmanFilter2D : 2-state model (price + local trend / velocity)
"""

import numpy as np


class KalmanFilter1D:
    """
    Scalar Kalman filter for tracking a 1-D state such as price.

    Model
    -----
    State transition:  x_{k+1} = x_k + w,   w ~ N(0, Q)   [random walk]
    Observation:       z_k     = x_k + v,   v ~ N(0, R)

    Parameters
    ----------
    Q : float
        Process noise variance.  Controls how fast the filter believes the true
        state can change.  Larger Q → more agile, responds quickly but noisier.
    R : float
        Measurement noise variance.  Controls how much each observation is
        trusted.  Larger R → smoother estimate but with more lag.
    x0 : float
        Initial state estimate (use the first measurement).
    P0 : float
        Initial error covariance (default 1.0; use a large value like 1e6 if
        you are completely uncertain about the starting state).
    """

    def __init__(self, Q: float, R: float, x0: float, P0: float = 1.0) -> None:
        self.Q = float(Q)
        self.R = float(R)
        self.x = float(x0)   # current state estimate
        self.P = float(P0)   # current error covariance

    # ── Core steps ──────────────────────────────────────────────────────────

    def predict(self) -> tuple[float, float]:
        """Prediction step.  Returns (x_pred, P_pred)."""
        x_pred = self.x            # random walk: predicted state = current state
        P_pred = self.P + self.Q   # uncertainty grows by Q each step
        return x_pred, P_pred

    def update(
        self, z: float, x_pred: float, P_pred: float
    ) -> tuple[float, float, float]:
        """
        Correction step given measurement z.
        Returns (x_new, P_new, K) where K is the Kalman gain.
        """
        K     = P_pred / (P_pred + self.R)       # Kalman gain ∈ (0, 1)
        x_new = x_pred + K * (z - x_pred)        # weighted blend of prediction and measurement
        P_new = (1.0 - K) * P_pred               # reduced uncertainty after incorporating z
        return x_new, P_new, K

    # ── Convenience interfaces ───────────────────────────────────────────────

    def step(self, z: float) -> float:
        """Process one measurement; update internal state. Returns current estimate."""
        x_pred, P_pred = self.predict()
        self.x, self.P, _ = self.update(float(z), x_pred, P_pred)
        return self.x

    def filter(
        self, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the filter over an entire measurement series.

        Returns
        -------
        estimates : (N,) float array of state estimates
        gains     : (N,) float array of Kalman gains
        variances : (N,) float array of posterior error covariances
        """
        N = len(measurements)
        estimates = np.empty(N)
        gains     = np.empty(N)
        variances = np.empty(N)
        for i, z in enumerate(measurements):
            x_pred, P_pred      = self.predict()
            self.x, self.P, K   = self.update(float(z), x_pred, P_pred)
            estimates[i] = self.x
            gains[i]     = K
            variances[i] = self.P
        return estimates, gains, variances


class KalmanFilter2D:
    """
    Two-state Kalman filter: price + trend (local slope / velocity).

    Model
    -----
    State vector:  x = [price, trend]ᵀ

    Transition matrix  F = [[1, 1],    price_{k+1} = price_k + trend_k + noise
                             [0, 1]]   trend_{k+1} = trend_k            + noise

    Observation matrix H = [[1, 0]]   we observe only price, not trend directly

    Q : 2×2 array — process noise covariance; typically np.diag([q_p, q_t])
    R : scalar    — measurement noise variance

    Parameters
    ----------
    Q  : np.ndarray, shape (2, 2)
    R  : float
    x0 : np.ndarray, shape (2,)  — [initial_price, initial_trend]
    P0 : np.ndarray, shape (2, 2), optional
         Defaults to 1000·I₂ (high initial uncertainty).
    """

    def __init__(
        self,
        Q: np.ndarray,
        R: float,
        x0: np.ndarray,
        P0: np.ndarray | None = None,
    ) -> None:
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)
        self.x = np.asarray(x0, dtype=float).copy()           # shape (2,)
        self.P = P0 if P0 is not None else np.eye(2) * 1000.0  # shape (2, 2)

    # ── Core steps ──────────────────────────────────────────────────────────

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prediction step.  Returns (x_pred shape(2,), P_pred shape(2,2))."""
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self, z: float, x_pred: np.ndarray, P_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Correction step.
        Returns (x_new shape(2,), P_new shape(2,2), K shape(2,)).
        """
        y = float(z) - float((self.H @ x_pred).item())         # innovation (scalar)
        S = float((self.H @ P_pred @ self.H.T).item()) + self.R # innovation covariance (scalar)
        K = (P_pred @ self.H.T) / S                             # Kalman gain, shape (2, 1)
        x_new = x_pred + K.flatten() * y
        P_new = (np.eye(2) - K @ self.H) @ P_pred
        return x_new, P_new, K.flatten()

    # ── Convenience interfaces ───────────────────────────────────────────────

    def step(self, z: float) -> tuple[float, float]:
        """Process one measurement. Returns (price_estimate, trend_estimate)."""
        x_pred, P_pred = self.predict()
        self.x, self.P, _ = self.update(z, x_pred, P_pred)
        return float(self.x[0]), float(self.x[1])

    def filter(
        self, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the filter over an entire measurement series.

        Returns
        -------
        prices : (N,) price estimates
        trends : (N,) trend (velocity) estimates
        gains  : (N, 2) Kalman gain vectors
        """
        N      = len(measurements)
        prices = np.empty(N)
        trends = np.empty(N)
        gains  = np.empty((N, 2))
        for i, z in enumerate(measurements):
            x_pred, P_pred      = self.predict()
            self.x, self.P, K   = self.update(float(z), x_pred, P_pred)
            prices[i] = self.x[0]
            trends[i] = self.x[1]
            gains[i]  = K
        return prices, trends, gains


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== KalmanFilter1D sanity check ===")
    true_price   = 100.0
    measurements = true_price + rng.standard_normal(80) * 5.0
    kf = KalmanFilter1D(Q=0.1, R=25.0, x0=measurements[0])
    estimates, gains, variances = kf.filter(measurements)
    print(f"  Final estimate : {estimates[-1]:.2f}  (true = {true_price})")
    print(f"  Final gain     : {gains[-1]:.4f}  (should be small after convergence)")
    print(f"  Final variance : {variances[-1]:.4f}")
    assert abs(estimates[-1] - true_price) < 8.0, "1D filter did not converge"
    print("  PASSED\n")

    print("=== KalmanFilter2D sanity check ===")
    trend_true  = 1.0
    true_prices = np.cumsum(np.ones(150) * trend_true) + 50.0
    meas2       = true_prices + rng.standard_normal(150) * 3.0
    Q2 = np.diag([0.5, 0.05])
    kf2 = KalmanFilter2D(Q=Q2, R=9.0, x0=np.array([meas2[0], 0.0]))
    prices2, trends2, _ = kf2.filter(meas2)
    print(f"  Final price estimate : {prices2[-1]:.2f}  (true ≈ {true_prices[-1]:.2f})")
    print(f"  Final trend estimate : {trends2[-1]:.4f}  (true = {trend_true})")
    assert abs(trends2[-1] - trend_true) < 0.5, "2D filter trend did not converge"
    print("  PASSED")
