import math
import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid, solve_ivp

def parse_matrix_func(matrix_str):
    """
    Parses a string representation of a matrix into a callable function A(t).
    
    Args:
        matrix_str (str): String representation of the matrix, e.g., "[[0, 1], [-1, -0.1*t]]"
        
    Returns:
        function: A function A(t) that takes a float t and returns a numpy array.
        
    Raises:
        ValueError: If the string cannot be parsed or is invalid.
    """
    try:
        # Define symbolic variable
        t = sp.symbols('t')
        
        # Safe parsing using sympify
        # We allow standard math functions
        matrix_sym = sp.sympify(matrix_str)
        
        if not isinstance(matrix_sym, sp.MutableDenseMatrix) and not isinstance(matrix_sym, sp.ImmutableDenseMatrix) and not isinstance(matrix_sym, list):
             # Try to convert list of lists to matrix to check dimensions if it's a list
             matrix_sym = sp.Matrix(matrix_sym)

        if not isinstance(matrix_sym, sp.MatrixBase):
             matrix_sym = sp.Matrix(matrix_sym)

        # Convert to a lambda function that returns a numpy array
        # 'numpy' module is used for math functions like sin, cos, etc.
        A_func_num = sp.lambdify(t, matrix_sym, modules='numpy')
        
        def A_func_wrapper(t_val):
            val = A_func_num(t_val)
            return np.array(val, dtype=float)

        return A_func_wrapper
        
    except Exception as e:
        raise ValueError(f"Failed to parse matrix string: {e}")

def compute_peano_series(A_func, t_eval, n_terms):
    """
    Computes the Peano-Baker series approximation of the state transition matrix.
    
    Args:
        A_func (function): Function A(t) returning (n, n) matrix.
        t_eval (array): Array of time points.
        n_terms (int): Number of terms in the series (including Identity).
        
    Returns:
        np.ndarray: Array of shape (len(t_eval), n, n) representing Phi(t, t0).
    """
    n_points = len(t_eval)
    
    # Determine matrix dimension n by calling A_func at t0
    A0 = A_func(t_eval[0])
    n_dim = A0.shape[0]
    
    # Initialize Phi_approx with Identity for all t
    # Shape: (n_points, n, n)
    Phi_approx = np.zeros((n_points, n_dim, n_dim))
    for i in range(n_points):
        Phi_approx[i] = np.eye(n_dim)
        
    # Initialize the previous term (Phi_0(t) = I)
    # Actually, the series is I + int(A) + int(A * int(A)) ...
    # So term 0 is I.
    # term 1 is int(A(tau) * term 0(tau)) dtau
    
    Phi_term_prev = np.zeros((n_points, n_dim, n_dim))
    for i in range(n_points):
        Phi_term_prev[i] = np.eye(n_dim)
        
    # We already have the I term in Phi_approx.
    # We need to add n_terms - 1 more terms.
    
    for k in range(1, n_terms):
        # Calculate integrand M(tau) = A(tau) * Phi_term_prev(tau)
        # We need to compute this for all t in t_eval
        
        # Pre-compute A(t) for all t to speed up
        # A_stack shape: (n_points, n, n)
        A_stack = np.array([A_func(t) for t in t_eval])
        
        # M_stack shape: (n_points, n, n)
        # Matrix multiplication at each time step
        M_stack = np.einsum('ijk,ikl->ijl', A_stack, Phi_term_prev)
        
        # Integrate cumulatively: Phi_term_new(t) = int_{t0}^t M(tau) dtau
        # cumulative_trapezoid works along axis 0 (time)
        Phi_term_new = cumulative_trapezoid(M_stack, t_eval, axis=0, initial=0)
        
        # Update approximation
        Phi_approx += Phi_term_new
        
        # Update prev term for next iteration
        Phi_term_prev = Phi_term_new
        
    return Phi_approx

def compute_peano_history(A_func, t_eval, n_terms):
    """
    Computes the history of Peano-Baker series approximations.
    
    Args:
        A_func (function): Function A(t).
        t_eval (array): Time points.
        n_terms (int): Number of terms N.
        
    Returns:
        np.ndarray: Array of shape (n_terms, len(t_eval), n, n) containing Phi_approx for each k=1..N.
    """
    n_points = len(t_eval)
    A0 = A_func(t_eval[0])
    n_dim = A0.shape[0]
    
    # History array
    # history[k] will hold the approximation using k+1 terms (indices 0 to k)
    history = np.zeros((n_terms, n_points, n_dim, n_dim))
    
    # Term 0: Identity
    Phi_approx = np.zeros((n_points, n_dim, n_dim))
    for i in range(n_points):
        Phi_approx[i] = np.eye(n_dim)
        
    history[0] = Phi_approx.copy()
    
    Phi_term_prev = Phi_approx.copy() # This is just I
    
    # Pre-compute A(t)
    A_stack = np.array([A_func(t) for t in t_eval])
    
    for k in range(1, n_terms):
        # M(tau) = A(tau) * Phi_term_prev(tau)
        M_stack = np.einsum('ijk,ikl->ijl', A_stack, Phi_term_prev)
        
        # Integrate
        Phi_term_new = cumulative_trapezoid(M_stack, t_eval, axis=0, initial=0)
        
        # Update approximation
        Phi_approx += Phi_term_new
        
        # Store in history
        history[k] = Phi_approx.copy()
        
        # Update prev term
        Phi_term_prev = Phi_term_new
        
    return history

def compute_peano_terms(A_func, t_eval, n_terms):
    """
    Computes the individual terms of the Peano-Baker series.
    
    Args:
        A_func (function): Function A(t).
        t_eval (array): Time points.
        n_terms (int): Number of terms N.
        
    Returns:
        np.ndarray: Array of shape (n_terms, len(t_eval), n, n) containing each Phi_k term.
    """
    n_points = len(t_eval)
    A0 = A_func(t_eval[0])
    n_dim = A0.shape[0]
    
    # Terms array
    terms = np.zeros((n_terms, n_points, n_dim, n_dim))
    
    # Term 0: Identity
    Phi_term_prev = np.zeros((n_points, n_dim, n_dim))
    for i in range(n_points):
        Phi_term_prev[i] = np.eye(n_dim)
        
    terms[0] = Phi_term_prev.copy()
    
    # Pre-compute A(t)
    A_stack = np.array([A_func(t) for t in t_eval])
    
    for k in range(1, n_terms):
        # M(tau) = A(tau) * Phi_term_prev(tau)
        M_stack = np.einsum('ijk,ikl->ijl', A_stack, Phi_term_prev)
        
        # Integrate
        Phi_term_new = cumulative_trapezoid(M_stack, t_eval, axis=0, initial=0)
        
        # Store term
        terms[k] = Phi_term_new.copy()
        
        # Update prev term
        Phi_term_prev = Phi_term_new
        
    return terms

def compute_true_solution(A_func, t_eval, x0):
    """
    Computes the true solution x(t) using scipy.integrate.solve_ivp.
    
    Args:
        A_func (function): Function A(t).
        t_eval (array): Time points.
        x0 (array): Initial state vector.
        
    Returns:
        np.ndarray: Array of shape (len(t_eval), n) representing x(t).
    """
    t_span = (t_eval[0], t_eval[-1])
    
    def odes(t, x):
        return A_func(t) @ x
        
    sol = solve_ivp(odes, t_span, x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    
    # sol.y is (n, n_points), we want (n_points, n)
    return sol.y.T

def compute_error_bound(A_func, t_eval, n_terms, x0_norm):
    """
    Computes the theoretical upper bound on the error of the Peano-Baker series approximation.
    
    Bound = (exp(L(t)) - sum_{k=0}^{N-1} L(t)^k / k!) * ||x0||
    where L(t) = int_{t0}^t ||A(tau)|| dtau
    
    Args:
        A_func (function): Function A(t).
        t_eval (array): Time points.
        n_terms (int): Number of terms N used in the approximation.
        x0_norm (float): Norm of the initial condition vector.
        
    Returns:
        np.ndarray: Error bound values for each time point in t_eval.
    """
    # 1. Compute norm of A(t) at each time point
    # We use Frobenius norm or 2-norm. Frobenius is easier/faster and is a valid consistent norm.
    A_norms = np.array([np.linalg.norm(A_func(t), ord='fro') for t in t_eval])
    
    # 2. Compute L(t) = cumulative integral of ||A(tau)||
    L_t = cumulative_trapezoid(A_norms, t_eval, initial=0)
    
    # 3. Compute the tail of the exponential series
    # Error <= (e^L - sum_{k=0}^{N-1} L^k/k!) * ||x0||
    # Note: n_terms includes Identity (k=0). So if n_terms=5, we have terms 0, 1, 2, 3, 4.
    # The approximation uses terms up to n_terms-1.
    
    exp_L = np.exp(L_t)
    
    sum_terms = np.zeros_like(L_t)
    for k in range(n_terms):
        # Add term L^k / k!
        # Use simple power and factorial for small k
        # For larger k, this might overflow, but n_terms is usually small (<100)
        # For safety with 1000 terms, we should be careful, but for visualization N is usually < 50.
        # Let's assume standard float64 precision is enough for typical usage.
        term = (L_t**k) / math.factorial(k)
        sum_terms += term
        
    bound = (exp_L - sum_terms) * x0_norm
    return bound
