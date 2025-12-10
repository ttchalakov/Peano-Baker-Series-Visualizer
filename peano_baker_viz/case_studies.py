"""
Spacecraft Relative Motion Case Study

Implements the Tschauner-Hempel equations for relative motion in elliptical orbits.
This is a 6-dimensional Linear Time-Varying (LTV) system where the dynamics depend
on the orbital eccentricity and true anomaly.

Reference:
- Tschauner, J., Hempel, P. (1965). "Rendezvous zu einem in elliptischer Bahn umlaufenden Ziel"
- Yamanaka, K., Ankersen, F. (2002). "New State Transition Matrix for Relative Motion on an Arbitrary Elliptical Orbit"
"""

import numpy as np
from scipy.integrate import solve_ivp


def mean_to_true_anomaly(M, e, tol=1e-10, max_iter=100):
    """
    Convert mean anomaly M to true anomaly θ (theta) using Newton-Raphson.
    
    Args:
        M: Mean anomaly (radians)
        e: Orbital eccentricity (0 <= e < 1)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        theta: True anomaly (radians)
    """
    # Initial guess for eccentric anomaly E
    E = M if e < 0.8 else np.pi
    
    # Newton-Raphson to solve Kepler's equation: M = E - e*sin(E)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        if np.abs(E_new - E) < tol:
            E = E_new
            break
        E = E_new
    
    # Convert eccentric anomaly E to true anomaly theta
    theta = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    
    return theta


def true_anomaly_rate(theta, e, n):
    """
    Compute the rate of change of true anomaly dθ/dt.
    
    Args:
        theta: True anomaly (radians)
        e: Orbital eccentricity
        n: Mean motion (rad/s)
        
    Returns:
        theta_dot: Rate of true anomaly change
    """
    return n * (1 + e * np.cos(theta))**2 / (1 - e**2)**(3/2)


def tschauner_hempel_A(t, e, n, theta_func):
    """
    Construct the Tschauner-Hempel system matrix A(t) for relative motion.
    
    The state vector is: x = [x, y, z, x_dot, y_dot, z_dot]^T
    where (x, y, z) are relative positions in the LVLH (Local Vertical Local Horizontal) frame:
        - x: radial (outward from Earth center)
        - y: in-track (along velocity direction) 
        - z: cross-track (perpendicular to orbital plane)
    
    Args:
        t: Time (seconds)
        e: Orbital eccentricity
        n: Mean motion (rad/s)
        theta_func: Function that returns true anomaly at time t
        
    Returns:
        A: 6x6 system matrix
    """
    theta = theta_func(t)
    
    # Derived quantities
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Orbital radius factor
    rho = 1 + e * cos_theta  # r/a = (1-e^2) / rho, but we work in normalized form
    
    # True anomaly rate
    omega = true_anomaly_rate(theta, e, n)
    
    # Angular acceleration component
    omega_dot = -2 * n**2 * e * sin_theta * (1 + e * cos_theta) / (1 - e**2)**3
    
    # Coefficients for the system matrix
    # These come from the linearized equations of relative motion
    k = 1 + e * cos_theta
    
    # Build the 6x6 A matrix
    A = np.zeros((6, 6))
    
    # Position to velocity coupling (identity in upper right)
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    
    # Acceleration equations (Hill-Clohessy-Wiltshire-like with time-varying coefficients)
    # For elliptical orbits, we use the true anomaly parameterization
    
    # Radial equation: x_ddot = 2*omega*y_dot + omega_dot*y + (3*n^2/rho^3 - omega^2)*x + ... 
    # Simplified version using angular dynamics
    
    # Using the formulation from Yamanaka-Ankersen:
    # d^2x/dt^2 - 2*omega*dy/dt - omega_dot*y - omega^2*x - (2*mu/r^3)*x = 0
    # For relative motion, mu/r^3 = n^2 * (1-e^2)^3 / k^3
    
    mu_over_r3 = n**2 * (1 - e**2)**3 / k**3
    
    # x equation (radial)
    A[3, 0] = omega**2 + 2 * mu_over_r3  # coefficient of x
    A[3, 1] = omega_dot                   # coefficient of y
    A[3, 4] = 2 * omega                   # coefficient of y_dot
    
    # y equation (in-track)
    A[4, 0] = -omega_dot                  # coefficient of x  
    A[4, 1] = omega**2 - mu_over_r3       # coefficient of y
    A[4, 3] = -2 * omega                  # coefficient of x_dot
    
    # z equation (cross-track) - decoupled from x,y
    A[5, 2] = -mu_over_r3                 # coefficient of z
    
    return A


def get_theta_function(e, n, theta0=0):
    """
    Create a function that returns true anomaly at time t.
    
    Args:
        e: Orbital eccentricity
        n: Mean motion (rad/s)
        theta0: Initial true anomaly (radians)
        
    Returns:
        theta_func: Function theta(t)
    """
    # Convert initial true anomaly to mean anomaly
    E0 = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(theta0 / 2),
        np.sqrt(1 + e) * np.cos(theta0 / 2)
    )
    M0 = E0 - e * np.sin(E0)
    
    def theta_func(t):
        M = M0 + n * t  # Mean anomaly at time t
        M = np.mod(M, 2 * np.pi)  # Keep in [0, 2*pi)
        return mean_to_true_anomaly(M, e)
    
    return theta_func


def create_spacecraft_A_func(eccentricity, orbital_period):
    """
    Create the A(t) function for spacecraft relative motion.
    
    Args:
        eccentricity: Orbital eccentricity (0 to <1)
        orbital_period: Orbital period in seconds
        
    Returns:
        A_func: Function that returns 6x6 A matrix at time t
    """
    n = 2 * np.pi / orbital_period  # Mean motion
    theta_func = get_theta_function(eccentricity, n, theta0=0)
    
    def A_func(t):
        return tschauner_hempel_A(t, eccentricity, n, theta_func)
    
    return A_func


def solve_relative_motion_ode(A_func, x0, t_span, t_eval):
    """
    Solve the relative motion equations numerically using ODE solver.
    
    Args:
        A_func: Function returning 6x6 A matrix at time t
        x0: Initial state [x, y, z, x_dot, y_dot, z_dot]
        t_span: (t0, tf) time span
        t_eval: Times at which to evaluate solution
        
    Returns:
        x_solution: State trajectory (n_times, 6)
    """
    def dynamics(t, x):
        A = A_func(t)
        return A @ x
    
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y.T


# ISS orbit parameters (approximate)
ISS_ORBITAL_PERIOD = 92.68 * 60  # ~92.68 minutes in seconds  
ISS_ECCENTRICITY = 0.0001  # Nearly circular

# Example initial conditions for rendezvous scenario
# Chaser starts 1 km behind and 100m below the target
DEFAULT_INITIAL_STATE = np.array([
    -100.0,    # x: 100m below (radial inward)
    -1000.0,   # y: 1km behind (in-track)
    0.0,       # z: same plane
    0.0,       # x_dot
    0.0,       # y_dot  
    0.0        # z_dot
])
