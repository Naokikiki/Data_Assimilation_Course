import numpy as np
import math
import random

def lorenz96(x, F):
    """Lorenz 96 model with constant forcing and explicit boundary conditions"""
    N = len(x)
    d = np.zeros(N)
    
    # Main equations for the bulk of the system
    for i in range(2, N - 1):
        d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + F
    
    # Periodic boundary conditions
    d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0] + F
    d[1] = (x[2] - x[N - 1]) * x[0] - x[1] + F
    d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1] + F
    
    return d

def rk4_step(x, dt=0.05, F=8):
    """4th order Runge-Kutta integration step"""
    k1 = dt * lorenz96(x, F)
    k2 = dt * lorenz96(x + k1 / 2, F)
    k3 = dt * lorenz96(x + k2 / 2, F)
    k4 = dt * lorenz96(x + k3, F)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def box_muller(mu=0.0, sigma=1.0):
    """Generate a Gaussian random number using the Box-Muller transform."""
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return mu + sigma * r * math.cos(theta)


def rmse(a, b):
    """Calculate Root Mean Square Error between arrays a and b."""
    return np.sqrt(np.mean((a - b) ** 2))

def tangent_linear(model,x,delta=0.0, **kwargs):       
    """
    tangent linear model for Lorenz 96.
    
    Parameters:
    model : function
        Function to compute the state transition matrix.
    x : np.ndarray
        State vector.
    delta : float
        approximation parameter for the differentiation.

    Returns:
    M : np.ndarray
        Tangent Linear Mode(Jacobian of M).
    """ 
    N = len(x)
    M = np.zeros((N, N))
    
    # Fill the Jacobian matrix
    for i in range(N):
        M[:,i] = model(x + delta * np.eye(N)[:,i], dt=0.05/3, F=8) - model(x, dt=0.05/3, F=8)
    M /=  delta 
    return M

def kalman_filter(x, P, H, R, y, Q, model=rk4_step,alpha=0.0):
    """
    Kalman Filter update step with inflation factor.
    
    Parameters:
    x : np.ndarray
        State vector.
    P : np.ndarray
        State covariance matrix.
    H : np.ndarray
        Observation matrix.
    R : np.ndarray
        Observation noise covariance matrix.
    y : np.ndarray
        Observation vector.
    M : function
        
    Q : np.ndarray
        Process noise covariance matrix.
    model : 
        Function to update the state.
    alpha : float
        Inflation factor for the covariance matrix.
    
    Returns:
    x_new : np.ndarray
        Updated state vector.
    P_new : np.ndarray
        Updated state covariance matrix.
    """
    
    M_1 = tangent_linear(model=rk4_step, x=x, delta=0.00001)  
    x_1 = model(x, dt=0.05/3, F=8)
    M_2 = tangent_linear(model=rk4_step, x=x_1, delta=0.00001) 
    x_2 = model(x_1, dt=0.05/3, F=8)
    M_3 = tangent_linear(model=rk4_step, x=x_2, delta=0.00001)  
    x_pred = model(x_2, dt=0.05/3, F=8) # State prediction non linear 
    P_pred = M_3 @ M_2 @ M_1 @ P @ M_1.T @ M_2.T @ M_3.T+ Q  # Covariance prediction
    P_pred *= (1 + alpha)  # Apply inflation factor

    # Update step
    innovation = y - H @ x_pred  # Innovation (residual)
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

    x_new = x_pred + K @ innovation  # Updated state estimate
    P_new = (np.eye(len(P)) - K @ H) @ P_pred  # Updated covariance estimate

    return x_new, P_new

def ensemble_kalman_filter(ensemble, H, R, y, Q, model=rk4_step, alpha=0.0):
    """
    A simple stochastic EnKF updating each ensemble member (with perturbed obs).
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (Nx, m). Nx = state dimension, m = ensemble size.
    H : np.ndarray
        Observation operator of shape (Ny, Nx).
    R : np.ndarray
        Observation error covariance of shape (Ny, Ny).
    y : np.ndarray
        Observed vector of shape (Ny,).
    Q : np.ndarray
        Process noise covariance of shape (Nx, Nx).
    model : function
        Model propagation function that takes a state vector and returns the next state.
    alpha : float
        Inflation factor for the forecast covariance.

    Returns:
    --------
    updated_ensemble : np.ndarray
        The updated ensemble with shape (Nx,m).
    """
    Nx, m = ensemble.shape  # m ensemble members, Nx state dimension

    # 1) Forecast each member forward
    ens_forecast = np.zeros_like(ensemble)
    for i in range(m):
        ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)

    # 2) Compute forecast mean and anomalies
    x_bar = np.mean(ens_forecast, axis=1)  # shape (Nx,)
    X_f = ens_forecast - x_bar[:, None]     # shape (Nx, m)

    # 3) Forecast covariance and inflation
    X_f *= np.sqrt(1 + alpha)                        # Inflate the anomalies
    Pf = (X_f @ X_f.T) / (m - 1) + Q                  # background covariance
    
    # 4) Perturb observations for each ensemble member (PO)
    Ny = H.shape[0]
    # np.random.multivariate_normal gives shape (m, Ny), so transpose to (Ny, m)
    obs_perturb = np.random.multivariate_normal(mean=np.zeros(Ny), cov=R, size=m).T  
    # H @ ens_forecast is Ny×m, so we can add the perturbations directly
    Y_ens = (H @ ens_forecast) + obs_perturb  # shape (Ny, m)

    # 5) Compute Kalman gain
    S = H @ Pf @ H.T + R                              # innovation covariance
    K = Pf @ H.T @ np.linalg.inv(S)                   # Kalman gain

    # 6) Update each ensemble member
    updated_ensemble = np.zeros_like(ens_forecast)
    for i in range(m):
        innovation_i = y - Y_ens[:, i]               # shape (Ny,)
        updated_ensemble[:, i] = ens_forecast[:, i] + K @ innovation_i

    return updated_ensemble

def ensemble_transform_kalman_filter(ensemble, H, R, y, Q, model=rk4_step, alpha=0.0):
    """ 
    Ensemble Transform Kalman Filter (ETKF) update step with inflation factor.

    Parameters:     
    -----------
    ensemble : np.ndarray
        Shape (Nx, m). Nx = state dimension, m = ensemble size.
    H : np.ndarray
        Observation operator of shape (Ny, Nx).
    R : np.ndarray
        Observation error covariance of shape (Ny, Ny).
    y : np.ndarray
        Observed vector of shape (Ny,).
    Q : np.ndarray
        Process noise covariance of shape (Nx, Nx).
    model : function
        Model propagation function that takes a state vector and returns the next state.
    alpha : float
        Inflation factor for the forecast covariance.
    Returns:
    --------
    updated_ensemble : np.ndarray
        The updated ensemble with shape (Nx, m).            
    """
    Nx, m = ensemble.shape
    Ny = H.shape[0]

    # 1) Forecast step
    ens_forecast = np.zeros_like(ensemble)
    for i in range(m):
        ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)

    # 2) Mean, anomalies, inflation
    x_bar = np.mean(ens_forecast, axis=1)  # shape (Nx,)
    delta_Xb = ens_forecast - x_bar[:, None]  # (Nx, m)
    
    # 3) Project to observation space
    y_bar = H @ x_bar                      # shape (Ny,)
    delta_Yb = H @ delta_Xb                          # shape (Ny, m)
    d = y - y_bar                          # innovation, shape (Ny,)

    # 4) ETKF update
    Yb_rinv = delta_Yb.T @ np.linalg.inv(R)             # shape (m, Ny)
    P_tilde_inv = np.eye(m) * (m-1)/(1 + alpha) + (Yb_rinv @ delta_Yb)   # shape (m, m)

    U, s, Vt = np.linalg.svd(P_tilde_inv)
    # Weights for mean
    w = (U @ np.diag(1.0 / s) @ U.T @ Yb_rinv @ d)   # shape (m,)

    x_a = x_bar + delta_Xb @ w  # Updated mean, shape (Nx,)

    # Compute transform for perturbations
    T = U @ np.diag(1.0 / np.sqrt(s)) @ U.T * np.sqrt(m - 1)  # shape (m, m)
    delta_Xa = delta_Xb @ T  # (Nx, m)

    # 5) Form updated ensemble
    updated_ensemble = x_a[:, None] + delta_Xa  # shape (Nx, m)

    return updated_ensemble

def local_ensemble_transform_kalman_filter(ensemble, H, R, y, Q, loc_radius=5, model=rk4_step, alpha=0.0):
    """
    Local Ensemble Transform Kalman Filter (LETKF) with R-localization.
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (Nx, m). Nx = state dimension, m = ensemble size.
    H : np.ndarray
        Observation operator of shape (Ny, Nx).
    R : np.ndarray
        Observation error covariance of shape (Ny, Ny) (assume diagonal).
    y : np.ndarray
        Observed vector of shape (Ny,).
    Q : np.ndarray
        Process noise covariance of shape (Nx, Nx).
    loc_radius : float
        Localization radius (sigma in the Gaussian localization function).
    model : function
        Model propagation function that takes a state vector and returns the next state.
    alpha : float
        Inflation factor for the forecast covariance.
        
    Returns:
    --------
    updated_ensemble : np.ndarray
        The updated ensemble with shape (Nx, m).
    """
    Nx, m = ensemble.shape
    Ny = H.shape[0]
    
    # 1) Forecast step
    ens_forecast = np.zeros_like(ensemble)
    for i in range(m):
        ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)
    
    # 2) Mean, anomalies
    x_bar = np.mean(ens_forecast, axis=1)           # shape (Nx,)
    delta_Xb = ens_forecast - x_bar[:, None]        # shape (Nx, m)
      
    # 3) Project to observation space
    y_bar = H @ x_bar                      # shape (Ny,)
    delta_Yb = H @ delta_Xb                     # shape (Ny, m)
    d = y - y_bar                          # innovation, shape (Ny,)
    
    # 4) Define localization function (Gaussian with cutoff)
    def localization_function(distance, sigma=loc_radius):
        """Gaussian localization function with cutoff"""
        cutoff = 2 * np.sqrt(10.0/3.0) * sigma
        if distance < cutoff:
            return np.exp(-0.5 * (distance**2) / (sigma**2))
        else:
            return 0.0
    
     # 5) Local analysis
    analysis_ensemble = np.zeros_like(ens_forecast)
    for i in range(Nx):
        # Build localized R
        R_loc = np.diag(np.diag(R).copy())
        for j in range(Ny):
            dist = min(abs(i - j), Nx - abs(i - j))
            loc_factor = localization_function(dist)
            R_loc[j, j] = (R_loc[j, j] / loc_factor) if loc_factor > 0 else 1.0e10

        R_loc_inv = np.diag(1.0 / np.diag(R_loc))

        # ETKF style update
        Yb_rinv = delta_Yb.T @ R_loc_inv
        P_tilde_inv = np.eye(m) * (m - 1) / (1 + alpha) + Yb_rinv @ delta_Yb

        U, s, Vt = np.linalg.svd(P_tilde_inv)
        w = (U @ np.diag(1.0 / s) @ U.T) @ (Yb_rinv @ d)
        T = U @ np.diag(1.0 / np.sqrt(s)) @ U.T * np.sqrt(m - 1)

        # Local update
        x_a_i = x_bar[i] + delta_Xb[i, :] @ w
        delta_Xa_i = delta_Xb[i, :] @ T
        analysis_ensemble[i, :] = x_a_i + delta_Xa_i

    return analysis_ensemble


def mc_resampling_matrix(weights, m, n_samples=30):
    """
    Generate an ensemble transform (resampling) matrix T for LPF via Monte-Carlo sampling.
    weights: array of shape (m,) with sum(weights)=1
    m: ensemble size
    n_samples: number of random samplings to average
    """
    T_avg = np.zeros((m, m))
    # Precompute cumulative weights
    cumulative_w = np.zeros(m+1)
    for i in range(m):
        cumulative_w[i+1] = cumulative_w[i] + weights[i]
    # Monte-Carlo averaging
    for _ in range(n_samples):
        # Draw m random numbers in [0,1] and sort
        r = np.sort(np.random.rand(m))
        idx = 0
        for j in range(m):
            while r[j] > cumulative_w[idx+1]:
                idx += 1
            T_avg[idx, j] += 1
    T_avg /= n_samples
    return T_avg

def local_particle_filter(ensemble, H, R, y, Q, weights=None, loc_radius=5, model=rk4_step, 
                           alpha=0.0, beta=0.0, 
                          n_samples=100, Neff_threshold=None, tau=0.1):
    """
    Local Particle Filter with Monte-Carlo-based resampling matrix and weight succession.
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (Nx, m). Nx = state dimension, m = ensemble size.
    H : np.ndarray
        Observation operator of shape (Ny, Nx).
    R : np.ndarray
        Observation error covariance of shape (Ny, Ny).
    y : np.ndarray
        Observed vector of shape (Ny,).
    Q : np.ndarray
        Process noise covariance of shape (Nx, Nx).
    weights : np.ndarray or None
        Previous weights from last analysis, shape (Nx, m).
        If None, initialize with uniform weights.
    loc_radius : float
        Localization radius (sigma in the Gaussian localization function).
    model : function
        Model propagation function that takes a state vector and returns the next state.
    alpha : float
        RTPS inflation parameter for posterior ensemble spread (0 to 1).
    beta : float
        Additive inflation magnitude parameter.
    n_samples : int
        Number of Monte-Carlo samples to generate the transform matrix.
    Neff_threshold : float or None
        Effective ensemble size threshold for resampling.
    tau : float
        Forgetting factor for weight succession (0 ≤ τ ≤ 1).
        τ = 0: Full succession of posterior weights.
        τ = 1: Reset to uniform weights.
        
    Returns:
    --------
    updated_ensemble : np.ndarray
        The updated ensemble with shape (Nx, m).
    updated_weights : np.ndarray
        Updated weights with shape (Nx, m).
    """
    Nx, m = ensemble.shape
    Ny = H.shape[0]
    
    # Initialize weights if not provided
    if weights is None:
        weights = np.ones((Nx, m)) / m
    
    # 1) Forecast step
    ens_forecast = np.zeros_like(ensemble)
    for i in range(m):
        ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)
    
    # 2) Compute weighted forecast mean and spread
    x_bar_f = np.mean(ens_forecast, axis=1)  # shape (Nx,)
    spread_f = np.std(ens_forecast, axis=1)  # shape (Nx,)

    y_ens = H @ ens_forecast  # shape (Ny, m)
    obs_diff = y.reshape(-1, 1) - y_ens  # shape (Ny, m)
    
    # 3) Define localization function (Gaussian with cutoff)
    def localization_function(distance, sigma=loc_radius):
        """Gaussian localization function with cutoff"""
        cutoff = 2 * np.sqrt(10.0/3.0) * sigma
        if distance < cutoff:
            return np.exp(-0.5 * (distance**2) / (sigma**2))
        else:
            return 0.0
    
    # 4) Local analysis for each grid point
    updated_ensemble = np.zeros((Nx, m)) 
    updated_weights = np.zeros((Nx, m))
    
    for i in range(Nx):
        # Build localized R for this grid point
        R_loc = np.diag(np.diag(R).copy())
        for j in range(Ny):
            # Calculate distance with periodic boundary conditions
            dist = min(abs(i - j), Nx - abs(i - j))
            loc_factor = localization_function(dist)
            R_loc[j, j] = (R_loc[j, j] / loc_factor) if loc_factor > 0 else 1.0e10
        
        # Compute observation likelihood with localized R
        invR_loc = np.diag(1.0 / np.diag(R_loc))  # Inverse of diagonal R_loc
        
        # Calculate likelihood weights
        quad_form = np.sum((obs_diff**2) * np.diag(invR_loc)[:,None], axis=0)  # shape (m,)
        likelihood_values = np.exp(-0.5 * quad_form)  # shape (m,)
        # Proper Bayesian update: posterior ∝ likelihood × prior
        posterior_weights = weights[i, :] * likelihood_values  # shape (m,)
        posterior_weights /= np.sum(posterior_weights)  # normalize

        # Check effective ensemble size with proper posterior weights
        Neff = 1.0 / np.sum(posterior_weights**2)        
                
        if Neff_threshold is not None and Neff >= Neff_threshold:
            # Skip resampling for this grid point
            updated_ensemble[i, :] = ens_forecast[i, :]
            
            # Apply weight succession formula: w_t+1 = (1-τ)·w_t + τ/m
            updated_weights[i, :] = (1 - tau) * posterior_weights + tau/m
        else:
            # Generate transform matrix via Monte-Carlo resampling
            T = mc_resampling_matrix(posterior_weights, m, n_samples=n_samples)
            
            # Apply transform to this grid point
            updated_ensemble[i, :] = ens_forecast[i, :] @ T
            
            # Reset weights to uniform after resampling
            updated_weights[i, :] = np.ones(m) / m
    
   # Always apply RTPS first (if alpha>0), then additive inflation (if beta>0)
    if alpha > 0.0:
        x_bar_a = np.mean(updated_ensemble, axis=1)
        perturbations_a = updated_ensemble - x_bar_a[:, None]
        spread_a = np.std(updated_ensemble, axis=1)
        for k in range(Nx):
            if spread_a[k] > 1e-10:
                rtps_factor = (1.0 - alpha) + alpha * (spread_f[k] / spread_a[k])
                perturbations_a[k, :] *= rtps_factor
        updated_ensemble = x_bar_a[:, None] + perturbations_a
    
    if beta > 0.0:
        x_bar_a = np.mean(updated_ensemble, axis=1)
        for k in range(Nx):
            rand_perturbations = np.random.normal(0, np.sqrt(beta), m)
            rand_perturbations -= np.mean(rand_perturbations)
            updated_ensemble[k, :] += rand_perturbations
    
    
    return updated_ensemble, updated_weights

def local_particle_filter_gaussian_mixture(ensemble, H, R, y, Q, weights=None, loc_radius=5, 
                                          model=rk4_step, alpha=0.0, beta =0.0, gamma=1.0, 
                                          n_samples=100, Neff_threshold=None, tau=0.1):
    """
    Local Particle Filter with Gaussian Mixture approximation (LPFGM).
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (Nx, m). Nx = state dimension, m = ensemble size.
    H : np.ndarray
        Observation operator of shape (Ny, Nx).
    R : np.ndarray
        Observation error covariance of shape (Ny, Ny).
    y : np.ndarray
        Observed vector of shape (Ny,).
    Q : np.ndarray
        Process noise covariance of shape (Nx, Nx).
    weights : np.ndarray or None
        Previous weights from last analysis, shape (Nx, m).
        If None, initialize with uniform weights.
    loc_radius : float
        Localization radius (sigma in the Gaussian localization function).
    model : function
        Model propagation function that takes a state vector and returns the next state.
    alpha : float
        RTPS inflation parameter for posterior ensemble spread.
    beta : float
        Additive inflation magnitude parameter.
        If beta > 0, additive inflation is applied after RTPS.
    gamma : float
        Scaling factor for the covariance matrix in Gaussian mixture kernels.
        Larger gamma widens kernels and reduces peak amplitude.
    n_samples : int
        Number of Monte-Carlo samples to generate the transform matrix.
    Neff_threshold : float or None
        Effective ensemble size threshold for resampling.
    tau : float
        Forgetting factor for weight succession (0 ≤ τ ≤ 1).
        
    Returns:
    --------
    updated_ensemble : np.ndarray
        The updated ensemble with shape (Nx, m).
    updated_weights : np.ndarray
        Updated weights with shape (Nx, m).
    """
    Nx, m = ensemble.shape
    Ny = H.shape[0]
    
    # Initialize weights if not provided
    if weights is None:
        weights = np.ones((Nx, m)) / m
    
    # 1) Forecast step
    ens_forecast = np.zeros_like(ensemble)
    for i in range(m):
        ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)
    
    # 2) Compute forecast mean and anomalies (Z)
    x_bar_f = np.mean(ens_forecast, axis=1)  # shape (Nx,)
    Z = ens_forecast - x_bar_f[:, None]      # shape (Nx, m)
    
    # Calculate forecast spread for inflation later
    spread_f = np.std(ens_forecast, axis=1)  # shape (Nx,)
    
    # 3) Define localization function (Gaussian with cutoff)
    def localization_function(distance, sigma=loc_radius):
        """Gaussian localization function with cutoff"""
        cutoff = 2 * np.sqrt(10.0/3.0) * sigma
        if distance < cutoff:
            return np.exp(-0.5 * (distance**2) / (sigma**2))
        else:
            return 0.0
    
    # 4) Local analysis for each grid point
    updated_ensemble = np.zeros((Nx, m))
    updated_weights = np.zeros((Nx, m))
    
    for i in range(Nx):
        # Build localized R for this grid point
        R_loc = np.diag(np.diag(R).copy())
        for j in range(Ny):
            # Calculate distance with periodic boundary conditions
            dist = min(abs(i - j), Nx - abs(i - j))
            loc_factor = localization_function(dist)
            R_loc[j, j] = (R_loc[j, j] / loc_factor) if loc_factor > 0 else 1.0e10
        
        
        
        # ---------- STEP 1: Kalman update of kernel centers ----------
        
        # Construct scaled local covariance matrix 
        # We compute a simplified version for computational efficiency
        # P_hat = gamma * Z @ Z.T / (m-1)
        Z_loc = Z[i, :].reshape(1, m)  # Local anomalies for grid point i, shape (1, m)
        
        # Project to observation space
        HZ = H @ Z  # shape (Ny, m)
        
        # Compute innovation covariance in ensemble space (ETKF form)
        R_loc_inv = np.linalg.inv(R_loc)
        
        # P_tilde_inv combines the prior ((m-1)/gamma * I) with the observation impact
        P_tilde_inv = np.eye(m) * ((m-1) / gamma) + HZ.T @ R_loc_inv @ HZ
   
        # SVD to find transformation matrix
        U, s, Vt = np.linalg.svd(P_tilde_inv)
        
       # Compute innovation at grid point i
        d = y - H @ x_bar_f  # innovation vector, shape (Ny,)
        T = U @ np.diag(1.0 / s) @ U.T @ HZ.T @ np.linalg.inv(R_loc) @ d + np.eye(m)
        
        
        # Apply Kalman update for kernel centers 
        x_a_i = x_bar_f[i] + Z_loc @ T  # Updated 　vector　shape (1, m)
        
        # ---------- STEP 2: Resampling based on posterior weights ----------
        
        # Compute observation likelihood for each moved particle
        # obs_diff = y[i] - H[i, :] @ x_a_i  # shape (m,)
        obs_diff = y.reshape(-1, 1) - H @ ens_forecast  # shape (Ny,m)

        # Calculate quadratic forms (scaled distances in observation space)
        quad_forms = np.zeros(m)
        for j in range(m):
            quad_forms[j] = obs_diff[:, j].T @ R_loc_inv @ obs_diff[:, j]
        
        # Convert to likelihood weights
        likelihood = np.exp(-0.5 * quad_forms)
        
        # Normalize weights
        posterior_weights = likelihood / np.sum(likelihood)
     
        # Check effective ensemble size with proper posterior weights
        Neff = 1.0 / np.sum(posterior_weights**2)
                        
        if Neff_threshold is not None and Neff >= Neff_threshold:
            # Skip resampling for this grid point
            updated_ensemble[i, :] = x_a_i  # Keep Kalman-moved particles
            # Apply weight succession formula
            updated_weights[i, :] = (1 - tau) * posterior_weights + tau/m
        else:
            # Generate transform matrix via Monte-Carlo resampling
            T_q = mc_resampling_matrix(posterior_weights, m, n_samples=n_samples)
            # Apply transform matrix to grid point 
            updated_ensemble[i, :] = x_a_i @ T_q
            # Reset weights to uniform after resampling
            updated_weights[i, :] = np.ones(m) / m
    
    # 5) Apply RTPS inflation if needed
    if alpha > 0.0:
        # Compute analysis mean and perturbations
        x_bar_a = np.mean(updated_ensemble, axis=1)  # shape (Nx,)
        perturbations_a = updated_ensemble - x_bar_a[:, None]  # shape (Nx, m)
        
        # Compute analysis spread
        spread_a = np.std(updated_ensemble, axis=1)  # shape (Nx,)
        
        # Apply RTPS formula for each state variable
        for k in range(Nx):
            # Avoid division by zero
            if spread_a[k] > 1e-10:
                rtps_factor = (1.0 - alpha) + alpha * (spread_f[k] / spread_a[k])
                perturbations_a[k, :] *= rtps_factor
        
        # Reconstruct ensemble
        updated_ensemble = x_bar_a[:, None] + perturbations_a
    # 6) Apply additive inflation if needed
    if beta > 0.0:
        x_bar_a = np.mean(updated_ensemble, axis=1)  # shape (Nx,)
        for k in range(Nx):
            rand_perturbations = np.random.normal(0, np.sqrt(beta), m)
            rand_perturbations -= np.mean(rand_perturbations)
            updated_ensemble[k, :] += rand_perturbations
    
    return updated_ensemble, updated_weights

# def local_particle_filter_gaussian_mixture2(ensemble, H, R, y, Q, weights=None, loc_radius=5, 
#                                           model=rk4_step, alpha=0.0, beta=0.0,  gamma=1.0, 
#                                           n_samples=100, Neff_threshold=None, tau=0.1):
   
#     Nx, m = ensemble.shape
#     Ny = H.shape[0]

#     if weights is None:
#         weights = np.ones((Nx, m)) / m
    
#     ens_forecast = np.zeros_like(ensemble)
#     for i in range(m):
#         ens_forecast[:, i] = model(ensemble[:, i], dt=0.05, F=8)
    
#     x_bar_f = np.mean(ens_forecast, axis=1)
#     Z = ens_forecast - x_bar_f[:, None]
#     spread_f = np.std(ens_forecast, axis=1)
    
#     def localization_function(distance, sigma=loc_radius):
#         cutoff = 2 * np.sqrt(10.0/3.0) * sigma
#         return np.exp(-0.5 * (distance**2) / (sigma**2)) if distance < cutoff else 0.0
    
#     # --- Step 1: Compute kernel centers (Kalman update) for each grid point ---
#     x_a_array = np.zeros((Nx, m))
#     R_loc_list = []

#     for i in range(Nx):
#         R_loc = np.diag(np.diag(R).copy())
#         for j in range(Ny):
#             dist = min(abs(i - j), Nx - abs(i - j))
#             loc_factor = localization_function(dist)
#             R_loc[j, j] = (R_loc[j, j] / loc_factor) if loc_factor > 0 else 1.0e10
        
#         R_loc_inv = np.linalg.inv(R_loc)
#         HZ = H @ Z
#         P_tilde_inv = np.eye(m) * ((m-1) / gamma) + HZ.T @ R_loc_inv @ HZ

#         U, s, Vt = np.linalg.svd(P_tilde_inv)
#         d = y - H @ x_bar_f
#         T = U @ np.diag(1.0 / s) @ U.T @ HZ.T @ R_loc_inv @ d + np.eye(m)

#         # Store T for potential diagnostic or extensions
#         R_loc_list.append(R_loc_inv)
        
#         x_a_i = x_bar_f[i] + Z[i, :].reshape(1, m) @ T
#         x_a_array[i, :] = x_a_i
    
#     # --- Step 2: Resampling using x_a_array from Step 1 ---
#     updated_ensemble = np.zeros((Nx, m))
#     updated_weights = np.zeros((Nx, m))

#     obs_diff_full = y.reshape(-1, 1) - H @ x_a_array  # Use updated centers for likelihood

#     for i in range(Nx):
#         R_loc_inv = R_loc_list[i]
#         quad_forms = np.zeros(m)
#         for j in range(m):
#             quad_forms[j] = obs_diff_full[:, j].T @ R_loc_inv @ obs_diff_full[:, j]
        
#         likelihood = np.exp(-0.5 * quad_forms)
#         posterior_weights = likelihood / np.sum(likelihood)

#         T_q = mc_resampling_matrix(posterior_weights, m, n_samples=n_samples)
#         updated_ensemble[i, :] = x_a_array[i, :].reshape(1, m) @ T_q
#         updated_weights[i, :] = np.ones(m) / m
    
#     # RTPS inflation
#     if alpha > 0.0:
#         x_bar_a = np.mean(updated_ensemble, axis=1)
#         perturbations_a = updated_ensemble - x_bar_a[:, None]
#         spread_a = np.std(updated_ensemble, axis=1)
#         for k in range(Nx):
#             if spread_a[k] > 1e-10:
#                 rtps_factor = (1.0 - alpha) + alpha * (spread_f[k] / spread_a[k])
#                 perturbations_a[k, :] *= rtps_factor
#         updated_ensemble = x_bar_a[:, None] + perturbations_a
    
#     # Additive inflation
#     if beta > 0.0:
#         x_bar_a = np.mean(updated_ensemble, axis=1)
#         for k in range(Nx):
#             rand_perturbations = np.random.normal(0, np.sqrt(beta), m)
#             rand_perturbations -= np.mean(rand_perturbations)
#             updated_ensemble[k, :] += rand_perturbations

#     return updated_ensemble, updated_weights
