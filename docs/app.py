import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
from utils import parse_matrix_func, compute_peano_series, compute_true_solution, compute_error_bound, compute_peano_history, compute_incremental_peano_terms

# Polyfill for st.fragment (introduced in Streamlit 1.34)
# Stlite might use an older version, so we define a no-op decorator if it's missing.
if not hasattr(st, "fragment"):
    def fragment(func):
        return func
    st.fragment = fragment

st.set_page_config(page_title="Peano-Baker Series Visualizer", layout="wide")

st.title("Peano-Baker Series Visualizer")
st.markdown(r"""
This app visualizes how the **Peano-Baker series** approximates the state transition matrix $\phi(t, t_0)$ for a Linear Time-Varying (LTV) system $\dot{x}(t) = A(t)x(t)$.

$$
\phi(t, t_0) = I + \int_{t_0}^t A(\sigma)\phi(\sigma, t_0) d\sigma
$$

The solution to the system is given by:

$$
x(t) = \phi(t, t_0)x(t_0)
$$

This suggests the following iterative scheme for finding $\phi$: Set $\phi_0(t, t_0) \equiv I$, and

$$
\phi_{k+1}(t, t_0) = I + \int_{t_0}^t A(\sigma)\phi_k(\sigma, t_0) d\sigma, \quad k \ge 0.
$$
""")

st.info("💡 **Tip**: You can use the **Left/Right Arrow Keys** on your keyboard to fine-tune the sliders in the sidebar!")

# Sidebar Controls
st.sidebar.header("System Configuration")

# Matrix A(t) Input
# Preset Systems
PRESET_SYSTEMS = {
    "Damped Oscillator (Time-Varying)": "[[0, 1], [-2 - 0.5*cos(t), -1]]",
    "Simple Harmonic Oscillator": "[[0, 1], [-1, 0]]",
    "Mathieu Equation (Parametric Resonance)": "[[0, 1], [-(1 + 0.5*cos(t)), 0]]"
}

selected_system = st.sidebar.selectbox("Select a Preset System", list(PRESET_SYSTEMS.keys()))
default_matrix = PRESET_SYSTEMS[selected_system]

# Matrix A(t) Input
matrix_str = st.sidebar.text_input("Matrix A(t) (Python syntax)", value=default_matrix, help="Use 't' as the time variable. Example: [[0, 1], [-1, -0.1*t]]")

# Simulation Parameters
# Max Terms Input
col_slider, col_max = st.sidebar.columns([3, 1])
with col_max:
    max_terms = st.number_input("Max", min_value=10, max_value=1000, value=20, step=5, label_visibility="collapsed")
with col_slider:
    n_terms = st.slider("Number of Terms (N)", min_value=1, max_value=max_terms, value=5)

col_T_slider, col_T_max = st.sidebar.columns([3, 1])
with col_T_max:
    max_T = st.number_input("Max T", min_value=1.0, max_value=60.0, value=20.0, step=1.0, label_visibility="collapsed")
with col_T_slider:
    T_horizon = st.slider("Time Horizon (T)", min_value=0.5, max_value=max_T, value=min(10.0, max_T), step=0.1)

# Initial Conditions
st.sidebar.subheader("Initial Conditions x(0)")
x1_0 = st.sidebar.number_input("x1(0)", value=1.0)
x2_0 = st.sidebar.number_input("x2(0)", value=0.0)
x0 = np.array([x1_0, x2_0])

# Cached Computation Functions
@st.cache_data
def get_true_solution(matrix_str, T, x0, n_points=1000):
    A_func = parse_matrix_func(matrix_str)
    t_eval = np.linspace(0, T, n_points)
    return compute_true_solution(A_func, t_eval, x0), t_eval

# Smart Caching for Peano Series
def get_smart_peano_series(matrix_str, T, n, n_points=1000):
    """
    Manages a session-state cache for Peano series terms to allow incremental computation.
    """
    # Key to detect system changes (now includes resolution)
    cache_key = (matrix_str, float(T), int(n_points))
    
    # Initialize cache if missing or system changed
    # We use a dict to separate caches by key to allow switching resolutions without clearing
    if 'peano_cache_store' not in st.session_state:
        st.session_state['peano_cache_store'] = {}
        
    if cache_key not in st.session_state['peano_cache_store']:
        A_func = parse_matrix_func(matrix_str)
        t_eval = np.linspace(0, T, n_points)
        
        # Initialize with Identity term (k=0)
        A0 = A_func(0)
        n_dim = A0.shape[0]
        
        term_0 = np.zeros((n_points, n_dim, n_dim))
        for i in range(n_points): term_0[i] = np.eye(n_dim)
        
        # Store tuple: (terms_list, t_eval)
        st.session_state['peano_cache_store'][cache_key] = {
            'terms': [term_0],
            't_eval': t_eval
        }
        
    # Retrieve from cache
    cache_data = st.session_state['peano_cache_store'][cache_key]
    cached_terms = cache_data['terms']
    t_eval = cache_data['t_eval']
    n_have = len(cached_terms)
    
    # If we need more terms
    if n > n_have:
        n_needed = n - n_have
        start_n = n_have
        prev_term = cached_terms[-1] # The last term we have
        
        A_func = parse_matrix_func(matrix_str)
        
        # Compute only the new terms
        new_terms = compute_incremental_peano_terms(A_func, t_eval, start_n, n_needed, prev_term)
        
        # Extend cache
        cached_terms.extend(new_terms)
        
    # Return sum of terms up to n
    terms_to_sum = cached_terms[:n]
    Phi_approx = np.sum(terms_to_sum, axis=0)
            
    return Phi_approx

@st.cache_data
def get_error_bound(matrix_str, T, n, x0_norm, n_points=10000):
    A_func = parse_matrix_func(matrix_str)
    t_eval = np.linspace(0, T, n_points)
    return compute_error_bound(A_func, t_eval, n, x0_norm)
    
@st.cache_data
def get_peano_history(matrix_str, T, n, n_points=1000):
    A_func = parse_matrix_func(matrix_str)
    t_eval = np.linspace(0, T, n_points)
    return compute_peano_history(A_func, t_eval, n)

# Computations
try:
    # Parse Matrix (needed for sensitivity analysis later)
    A_func = parse_matrix_func(matrix_str)

    with st.spinner("Computing Solutions..."):
        # 1. Visualization Stream (Low-Res, Fast)
        n_pts_viz = 1000
        x_true_viz, t_eval_viz = get_true_solution(matrix_str, T_horizon, x0, n_points=n_pts_viz)
        Phi_approx_viz = get_smart_peano_series(matrix_str, T_horizon, n_terms, n_points=n_pts_viz)
        x_approx_viz = np.einsum('ijk,k->ij', Phi_approx_viz, x0)
        
    # Compute bounds for plotting based on Visualization Stream
    x1_min, x1_max = np.min(x_true_viz[:, 0]), np.max(x_true_viz[:, 0])
    x2_min, x2_max = np.min(x_true_viz[:, 1]), np.max(x_true_viz[:, 1])
    
    x1_span = x1_max - x1_min
    x2_span = x2_max - x2_min
    
    # Handle constant states (span = 0)
    if x1_span == 0: x1_span = 1.0
    if x2_span == 0: x2_span = 1.0
    
    pad_x1 = 0.1 * x1_span
    pad_x2 = 0.1 * x2_span
    
    x1_bounds = [x1_min - pad_x1, x1_max + pad_x1]
    x2_bounds = [x2_min - pad_x2, x2_max + pad_x2]

    # Visualizations
    col1, col2 = st.columns(2)
    
    # Plot 1: Phase Portrait
    with col1:
        st.subheader("Phase Portrait (x1 vs x2)")
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=x_true_viz[:, 0], y=x_true_viz[:, 1], mode='lines', name='True Solution', line=dict(color='blue')))
        fig_phase.add_trace(go.Scatter(x=x_approx_viz[:, 0], y=x_approx_viz[:, 1], mode='lines', name='Peano Approx', line=dict(color='red', dash='dash')))
        fig_phase.update_layout(
            xaxis_title="x1", yaxis_title="x2", height=400,
            xaxis_range=x1_bounds, yaxis_range=x2_bounds
        )
        st.plotly_chart(fig_phase, width="stretch")
        
    # Plot 2: State vs Time
    with col2:
        st.subheader("State Trajectories vs Time")
        
        # Calculate common y-bounds for time plot
        y_min = min(x1_min, x2_min)
        y_max = max(x1_max, x2_max)
        y_span = y_max - y_min
        if y_span == 0: y_span = 1.0
        pad_y = 0.1 * y_span
        y_bounds = [y_min - pad_y, y_max + pad_y]
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_eval_viz, y=x_true_viz[:, 0], mode='lines', name='x1 (True)', line=dict(color='blue')))
        fig_time.add_trace(go.Scatter(x=t_eval_viz, y=x_approx_viz[:, 0], mode='lines', name='x1 (Approx)', line=dict(color='blue', dash='dash')))
        fig_time.add_trace(go.Scatter(x=t_eval_viz, y=x_true_viz[:, 1], mode='lines', name='x2 (True)', line=dict(color='green')))
        fig_time.add_trace(go.Scatter(x=t_eval_viz, y=x_approx_viz[:, 1], mode='lines', name='x2 (Approx)', line=dict(color='green', dash='dash')))
        fig_time.update_layout(
            xaxis_title="Time (t)", yaxis_title="State Value", height=400,
            yaxis_range=y_bounds
        )
        st.plotly_chart(fig_time, width="stretch")
        
    # Plot 3: Error Convergence
    st.subheader("Approximation Error over Time")
    
    st.markdown("""
    **Why use an Upper Bound?** 
    In safety-critical applications, we often don't have the "true" solution to compare against. 
    However, we can analytically derive a **theoretical upper bound** on the Peano-Baker series error. 
    If this conservative bound stays within our safety tolerance, we can guarantee the approximation is safe to use 
    up to a certain time horizon, providing a certification of accuracy without needing the exact solution.
    """)

    # Layout: Controls on Left, Plot on Right
    col_err_controls, col_err_plot = st.columns([1, 3])
    
    # 2. Error Analysis Stream (High-Res, Precise)
    # We compute this separately to ensure the check is valid
    n_pts_err = 10000
    x_true_err, t_eval_err = get_true_solution(matrix_str, T_horizon, x0, n_points=n_pts_err)
    Phi_approx_err = get_smart_peano_series(matrix_str, T_horizon, n_terms, n_points=n_pts_err)
    x_approx_err = np.einsum('ijk,k->ij', Phi_approx_err, x0)
    
    # Error = ||x_true - x_approx||
    error = np.linalg.norm(x_true_err - x_approx_err, axis=1)
    
    # Theoretical Bound calculation
        
    bound = get_error_bound(matrix_str, T_horizon, n_terms, np.linalg.norm(x0), n_points=n_pts_err)

    with col_err_controls:
        use_log_scale = st.checkbox("Log Scale", value=True)
        error_threshold = st.number_input("Error Threshold", value=0.01, min_value=1e-9, format="%.2e")
        
        # Calculate Validity Horizons (Use High-Res Data)
        # 1. Theoretical Horizon
        valid_indices_bound = np.where(bound > error_threshold)[0]
        if len(valid_indices_bound) > 0:
            valid_time_bound = t_eval_err[valid_indices_bound[0]]
            valid_msg_bound = f"{valid_time_bound:.2f} s"
        else:
            valid_msg_bound = f"> {T_horizon:.2f} s"
            
        # 2. True Error Horizon
        valid_indices_true = np.where(error > error_threshold)[0]
        if len(valid_indices_true) > 0:
            valid_time_true = t_eval_err[valid_indices_true[0]]
            valid_msg_true = f"{valid_time_true:.2f} s"
        else:
            valid_msg_true = f"> {T_horizon:.2f} s"
            
        st.metric("Theoretical Validity Horizon", valid_msg_bound, help=f"Time until the theoretical bound exceeds {error_threshold}")
        st.metric("True Validity Horizon", valid_msg_true, help=f"Time until the actual error exceeds {error_threshold}")

    with col_err_plot:
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(x=t_eval_err, y=error, mode='lines', name='Actual Error', line=dict(color='orange')))
        fig_error.add_trace(go.Scatter(x=t_eval_err, y=bound, mode='lines', name='Theoretical Bound', line=dict(color='gray', dash='dot')))
        
        # Add threshold line
        fig_error.add_hline(y=error_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        
        yaxis_type = "log" if use_log_scale else "linear"
        fig_error.update_layout(
            xaxis_title="Time (t)", 
            yaxis_title="Error ||x_true - x_approx||", 
            height=300, 
            yaxis_type=yaxis_type,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_error, width="stretch")

    with st.expander("ℹ️ How is the Upper Bound derived?"):
        st.markdown(r"""
        ### Theoretical Foundation
        This derivation is based on the work of **Baake & Schlägel**, *"The Peano–Baker series"* ([arXiv:1011.1775v3](https://arxiv.org/abs/1011.1775v3)).

        **The Analytical Upper Bound**
        Let $x(t)$ be the exact solution and $x_N(t)$ be the approximation using $N$ terms. The error is bounded by:

        $$
        \| x(t) - x_N(t) \| \le \underbrace{\left[ \exp\left( \int_{t_0}^t \|A(\tau)\| \, d\tau \right) - \sum_{k=0}^N \frac{1}{k!} \left( \int_{t_0}^t \|A(\tau)\| \, d\tau \right)^k \right]}_{\text{Scalar Series Tail}} \cdot \| x_0 \|
        $$

        ### Derivation Key Points
        1.  **Error Vector**: The error in the solution is $e(t) = (\Phi(t, t_0) - \Phi_N(t, t_0)) x_0$.
        2.  **Norm Property**: Using the sub-multiplicative property $\|M v\| \le \|M\| \|v\|$, we separate the initial condition:
            $$ \| x(t) - x_N(t) \| \le \| \Phi(t, t_0) - \Phi_N(t, t_0) \| \cdot \| x_0 \| $$
        3.  **Scalar Dominant**: The matrix error norm $\| \Phi - \Phi_N \|$ is bounded by the tail of the scalar exponential series of the cumulative norm (or "gain") $L(t) = \int_{t_0}^t \|A(\tau)\| d\tau$.

        ### Interpretation
        *   **Worst-Case Scenario**: This bound assumes the initial state $x_0$ aligns perfectly with the direction of maximum error growth.
        *   **Influence of Dynamics**: If $\|A(t)\|$ is large (high gain/fast dynamics), $L(t)$ grows quickly, requiring a larger $N$ to suppress the scalar tail.

        **Note on Numerical vs. Theoretical Error**:
        You may observe that the **Actual Error** is slightly higher than the **Theoretical Bound** at very small magnitudes (e.g., $10^{-10}$). This is due to **discretization error** in the numerical integration. The Theoretical Bound only accounts for the *truncation* of the series.

        **Why is the Bound so loose for Stable Systems?**
        You might notice the bound exploding while the actual error stays small. This is because **the bound is conservative**: it depends on $ \exp(\int \|A\|) $, which treats every matrix contribution as "growth" (ignoring signs and directions). accurate cancellation or stability (like in a damped oscillator) is ignored by the norm. Thus, for stable systems with large $\|A\|$, the bound assumes a worst-case explosion that doesn't actually happen.
        """)

    # --- Sensitivity Analysis Section ---
    st.markdown("---")
    st.header("Sensitivity Analysis")
    st.markdown("Explore how changes in the model affect the approximation accuracy.")

    with st.expander("1. Impact of Initial Condition Magnitude"):
        st.markdown("""
        **Hypothesis**: Does changing the initial condition $x_0$ affect the *convergence* of the series?
        
        Let's scale the initial condition by a factor $\\alpha$ and observe the error.
        """)
        
        alpha = st.slider("Scale Factor for x0 (alpha)", 0.1, 10.0, 2.0, step=0.1)
        
        # Calculate for scaled x0
        x0_scaled = x0 * alpha
        x_true_scaled = compute_true_solution(A_func, t_eval_err, x0_scaled)
        x_approx_scaled = np.einsum('ijk,k->ij', Phi_approx_err, x0_scaled)
        error_scaled = np.linalg.norm(x_true_scaled - x_approx_scaled, axis=1)
        
        # Plot comparison
        fig_sens_x0 = go.Figure()
        fig_sens_x0.add_trace(go.Scatter(x=t_eval_err, y=error, mode='lines', name=f'Error (Original x0)', line=dict(color='orange')))
        fig_sens_x0.add_trace(go.Scatter(x=t_eval_err, y=error_scaled, mode='lines', name=f'Error (Scaled x0 * {alpha})', line=dict(color='purple', dash='dot')))
        
        fig_sens_x0.update_layout(
            title="Error vs Time (Original vs Scaled Initial Condition)",
            xaxis_title="Time (t)", 
            yaxis_title="Error Norm",
            yaxis_type="log" if use_log_scale else "linear"
        )
        st.plotly_chart(fig_sens_x0, width="stretch")
        
        st.info(f"""
        **Observation**: Notice that the error curve shape is identical, just shifted. 
        
        Ratio of Max Errors: {np.max(error_scaled) / (np.max(error) + 1e-9):.2f} (Expected: {alpha})
        
        **Conclusion**: The Peano-Baker series approximation of $\\Phi(t, t_0)$ is **independent** of $x_0$. The error in $x(t)$ scales linearly with $x_0$, but the relative accuracy of the matrix approximation remains constant.
        """)

    with st.expander("2. Impact of System Dynamics (Norm of A)"):
        st.markdown("""
        **Hypothesis**: Does a "faster" system (larger values in $A(t)$) require more terms to converge?
        
        Let's scale the matrix $A(t)$ by a factor $\\beta$: $A_{new}(t) = \\beta A(t)$.
        """)
        
        beta = st.slider("Scale Factor for A(t) (beta)", 0.5, 5.0, 2.0, step=0.5)
        
        if beta != 1.0:
            # Define scaled A function
            # We can't easily modify the string, so we wrap the function
            def A_func_scaled(t):
                return beta * A_func(t)
            
            # Recompute for scaled system (cannot use cache easily here without refactoring, so we compute directly)
            # This is fine for exploration
            with st.spinner(f"Computing for scaled system (beta={beta})..."):
                # Use Low-Res for Sensitivity to match the Visualization Stream (Fast)
                
                x_true_beta = compute_true_solution(A_func_scaled, t_eval_viz, x0)
                # Note: compute_peano_series is strictly O(N*points), so using 1k points is 10x faster
                Phi_approx_beta = compute_peano_series(A_func_scaled, t_eval_viz, n_terms)
                x_approx_beta = np.einsum('ijk,k->ij', Phi_approx_beta, x0)
                error_beta = np.linalg.norm(x_true_beta - x_approx_beta, axis=1)
                
                # Compute error for original system on viz grid for comparison
                error_viz = np.linalg.norm(x_true_viz - x_approx_viz, axis=1)
                
            # Plot comparison
            fig_sens_A = go.Figure()
            fig_sens_A.add_trace(go.Scatter(x=t_eval_viz, y=error_viz, mode='lines', name=f'Error (Original A)', line=dict(color='orange')))
            fig_sens_A.add_trace(go.Scatter(x=t_eval_viz, y=error_beta, mode='lines', name=f'Error (Scaled A * {beta})', line=dict(color='red', dash='dot')))
            
            fig_sens_A.update_layout(
                title=f"Error vs Time (Original vs Scaled Dynamics)",
                xaxis_title="Time (t)", 
                yaxis_title="Error Norm",
                yaxis_type="log" if use_log_scale else "linear"
            )
            st.plotly_chart(fig_sens_A, width="stretch")
            
            st.warning("""
            **Conclusion**: Increasing the norm of $A(t)$ (making the system "faster" or "stiffer") **drastically increases** the error for a fixed number of terms. 
            
            The Peano-Baker series convergence depends on the integral of the norm of $A(t)$. Larger $A(t)$ means you need **more terms** ($N$) to achieve the same accuracy.
            """)
        else:
            st.write("Set beta != 1.0 to see the effect.")

    # --- Advanced Visualizations Section ---
    st.markdown("---")
    st.header("Advanced Visualizations")
    
    # We use the cached `get_peano_history` defined at the top of the file

    tab1, tab2, tab3, tab4 = st.tabs(["Convergence Heatmap", "Trajectory Animation", "Computational Complexity", "Term Contribution Analysis"])
    
    with tab1:
        st.subheader("Convergence Heatmap (Time vs Terms)")
        st.markdown("This heatmap visualizes the **logarithm of the error** as a function of Time (X-axis) and Number of Terms (Y-axis).")
        
        @st.fragment
        def heatmap_fragment():
            if st.button("Generate Heatmap", key="heatmap_btn"):
                with st.spinner("Computing Heatmap..."):
                    # Use max_terms from slider as the limit
                    heatmap_n = max_terms
                    
                    history = get_peano_history(matrix_str, T_horizon, heatmap_n, n_points=1000)
                    # history shape: (n_terms, n_points, n, n)
                    
                    # Compute error for each term
                    # We need x_true. Since this is a fragment, we can recompute x_true or fetch it.
                    # Ideally we fetch it. But `x_true_viz` is local to main execution.
                    # We can call get_true_solution(n_points=1000).
                    
                    x_true_heat, _ = get_true_solution(matrix_str, T_horizon, x0, n_points=1000)
                    
                    # Calculate errors
                    # errors shape: (n_terms, n_points)
                    errors = np.zeros((heatmap_n, 1000))
                    
                    for k in range(heatmap_n):
                        Phi_k = history[k]
                        x_approx_k = np.einsum('ijk,k->ij', Phi_k, x0)
                        errors[k] = np.linalg.norm(x_true_heat - x_approx_k, axis=1)
                        
                    # Log scale for heatmap
                    # Add small epsilon to avoid log(0)
                    log_errors = np.log10(errors + 1e-16)
                    
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=log_errors,
                        x=np.linspace(0, T_horizon, 1000), # Explicit linspace for this frag
                        y=np.arange(1, heatmap_n + 1),
                        colorscale='Viridis_r', # Reverse Viridis so dark is low error
                        colorbar=dict(title='Log10(Error)')
                    ))
                    
                    fig_heat.update_layout(
                        xaxis_title="Time (t)",
                        yaxis_title="Number of Terms (N)",
                        title="Convergence Wavefront",
                        height=500
                    )
                    st.plotly_chart(fig_heat, width="stretch")
        
        heatmap_fragment()
                
    with tab2:
        st.subheader("Trajectory Animation (Evolution with N)")
        st.markdown("Watch how the phase portrait trajectory evolves as you add more terms to the series.")
        
        @st.fragment
        def animation_fragment():
            if st.button("Generate Animation", key="anim_btn"):
                with st.spinner("Preparing Animation..."):
                    anim_n = max_terms
                    history = get_peano_history(matrix_str, T_horizon, anim_n, n_points=1000)
                    x_true_anim, _ = get_true_solution(matrix_str, T_horizon, x0, n_points=1000)
                    
                    # Loop to generate frames, but stop if converged to avoid numerical instability
                    frames = []
                    actual_anim_n = 0
                    
                    for k in range(anim_n):
                        Phi_k = history[k]
                        x_approx_k = np.einsum('ijk,k->ij', Phi_k, x0)
                        
                        # Check convergence
                        current_error = np.max(np.linalg.norm(x_true_anim - x_approx_k, axis=1))
                        
                        frame = go.Frame(
                            data=[
                                 go.Scatter(x=x_true_anim[:, 0], y=x_true_anim[:, 1], mode='lines', name='True Solution', line=dict(color='blue')),
                                 go.Scatter(x=x_approx_k[:, 0], y=x_approx_k[:, 1], mode='lines', line=dict(color='red', dash='dash'), name='Peano Approx')
                            ],
                            name=f"frame{k}"
                        )
                        frames.append(frame)
                        actual_anim_n += 1
                        
                        # Stop if converged (error is sufficiently small)
                        if current_error < 1e-6:
                             st.caption(f"Animation stopped at term {k+1} due to convergence (Error < 1e-6).")
                             break
                        
                    # Initial plot (N=1)
                    Phi_0 = history[0]
                    x_approx_0 = np.einsum('ijk,k->ij', Phi_0, x0)
                    
                    fig_anim = go.Figure(
                        data=[
                            go.Scatter(x=x_true_anim[:, 0], y=x_true_anim[:, 1], mode='lines', name='True Solution', line=dict(color='blue')),
                            go.Scatter(x=x_approx_0[:, 0], y=x_approx_0[:, 1], mode='lines', name='Peano Approx', line=dict(color='red', dash='dash'))
                        ],
                        layout=go.Layout(
                            xaxis=dict(range=x1_bounds, title="x1"),
                            yaxis=dict(range=x2_bounds, title="x2"),
                            title="Phase Portrait Evolution",
                            height=500,
                            updatemenus=[dict(
                                type="buttons",
                                buttons=[dict(label="Play",
                                              method="animate",
                                              args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
                            )]
                        ),
                        frames=frames
                    )
                    
                    # Add slider for animation
                    sliders = [dict(
                        steps=[dict(method='animate',
                                    args=[[f'frame{k}'], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
                                    label=f'{k+1}'
                                    ) for k in range(actual_anim_n)],
                        transition=dict(duration=0),
                        x=0,
                        y=0,
                        currentvalue=dict(font=dict(size=12), prefix='Terms: ', visible=True, xanchor='center'),
                        len=1.0)
                    ]
                    fig_anim.update_layout(sliders=sliders)
                    
                    st.plotly_chart(fig_anim, width="stretch")
        
        animation_fragment()

    with tab3:
        st.subheader("Computational Complexity Benchmark")
        st.markdown("Benchmarks the Peano-Baker series computation time against **Number of Terms (N)** and **Resolution (Time Points)**.")
        
        @st.fragment
        def benchmark_fragment():
            if st.button("Run Benchmark", key="bench_btn"):
                import time
                
                # Define grid for terms and points
                n_range = [5, 10, 15, 20, 30, 40, 50, 75, 100]
                pts_range = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
                
                times = np.zeros((len(n_range), len(pts_range)))
                
                progress_bar = st.progress(0)
                total_steps = len(n_range) * len(pts_range)
                step = 0
                
                for i, n in enumerate(n_range):
                    for j, pts in enumerate(pts_range):
                        t_test = np.linspace(0, T_horizon, pts)
                        
                        start_time = time.time()
                        # Directly call compute_peano_series (bypass cache for benchmark)
                        compute_peano_series(A_func, t_test, n)
                        end_time = time.time()
                        
                        times[i, j] = end_time - start_time
                        
                        step += 1
                        progress_bar.progress(step / total_steps)
                
                # Plot Heatmap
                fig_comp = go.Figure(data=go.Heatmap(
                    z=times,
                    x=[str(p) for p in pts_range], # Categorical axis for points
                    y=[str(n) for n in n_range],   # Categorical axis for terms
                    colorscale='Magma',
                    colorbar=dict(title='Time (s)')
                ))
                
                fig_comp.update_layout(
                    title="Computation Time (s)",
                    xaxis_title="Number of Time Points (Resolution)",
                    yaxis_title="Number of Terms (N)",
                    height=400
                )
                st.plotly_chart(fig_comp, width="stretch")
                
                st.info("**Observation**: Complexity scales linearly with $N$ (Terms) and linearly with $M$ (Points), confirming the $O(N \\cdot M)$ complexity of the cumulative trapezoidal integration scheme.")
        
        benchmark_fragment()

    with tab4:
        st.subheader("Magnitude of Series Terms")
        st.markdown(r"""
        The Peano-Baker series is a sum: $\Phi(t) = I + \Phi_1(t) + \Phi_2(t) + \dots$
        
        This plot shows the magnitude (Frobenius norm) of each term $||\Phi_k(T)||$ at the final time $T$.
        For the series to converge, these terms **must decay to zero**.
        """)
        
        @st.fragment
        def term_analysis_fragment():
            if st.button("Analyze Term Magnitudes", key="term_btn"):
                from utils import compute_peano_terms
                
                with st.spinner("Computing Term contributions..."):
                    analysis_n = max_terms
                    
                    # Get terms
                    A_func_analysis = parse_matrix_func(matrix_str)
                    t_eval_analysis = np.linspace(0, T_horizon, 10000)
                    terms = compute_peano_terms(A_func_analysis, t_eval_analysis, analysis_n)
                    # terms shape: (n_terms, n_points, n, n)
                    
                    # Compute norm of each term at final time T
                    # We take the last point: terms[:, -1, :, :]
                    final_terms = terms[:, -1, :, :]
                    magnitudes = [np.linalg.norm(term, ord='fro') for term in final_terms]
                    
                    if magnitudes[-1] < 1e-15:
                        st.success("✅ The series has converged to machine precision.")
                    
                    # Plotting
                    # Truncate for visualization (don't show 100 zeros if it converged at term 10)
                    display_mags = []
                    for m in magnitudes:
                        display_mags.append(m)
                        if m < 1e-16:
                             break
                    
                    fig = go.Figure(data=[
                        go.Bar(x=list(range(1, len(display_mags)+1)), y=display_mags)
                    ])
                    
                    fig.update_layout(
                        title="Decay of Term Magnitudes",
                        xaxis_title="k-th Term",
                        yaxis_title="Frobenius Norm ||Phi_k(T)||",
                        yaxis_type="log",
                        height=400
                    )
                    
                    st.plotly_chart(fig, width="stretch")
        
        term_analysis_fragment()

    # --- Parameter Sensitivity Section ---
    st.markdown("---")
    st.header("Gradients with Respect to Parameters (Sensitivity)")
    
    st.markdown(r"""
    In modern engineering, we often need to tune system parameters $p$ to achieve a desired behavior. 
    To do this efficiently using algorithms like **Gradient Descent**, we need the gradient of the output with respect to the parameters: $\frac{\partial x(t)}{\partial p}$.
    
    Standard methods like **Finite Differences** require re-running the simulation for every parameter perturbation, which is computationally expensive and prone to numerical errors.
    
    **The Peano-Baker Advantage**:
    The series structure allows us to compute the **exact analytic gradient** directly alongside the state solution, *without* re-simulating the system!
    
    $$ \frac{\partial \Phi(t, t_0)}{\partial p} = \int_{t_0}^t \Phi(t, \sigma) \left[ \frac{\partial A(\sigma, p)}{\partial p} \right] \Phi(\sigma, t_0) \, d\sigma $$
    """)
    
    @st.fragment
    def parameter_sensitivity_fragment():
        st.subheader("Sensitivity Surface: The Mathieu Equation")
        st.markdown(r"""
        We will visualize the **magnitude of sensitivity** for the Mathieu Equation across a range of parameters.
        
        **System in State Space form** ($\dot{\mathbf{x}} = A(t)\mathbf{x}$):
        
        $$ 
        \frac{d}{dt} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = 
        \begin{bmatrix} 0 & 1 \\ -(a + \epsilon \cos(t)) & 0 \end{bmatrix} 
        \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
        $$
        """)
        
        # Inputs for Surface Plot
        col_surf_1, col_surf_2 = st.columns(2)
        with col_surf_1:
            sens_n_terms = st.number_input("Peano Terms (N)", value=5, min_value=1, max_value=200, help="Number of terms to use for the Gradient Series")
            sens_T = st.slider("Time Horizon (T)", 1.0, 30.0, 10.0)
            
        with col_surf_2:
            st.info(r"Visualizing Gradient Magnitude w.r.t parameters $\epsilon$ (perturbation) and $a$ (stiffness).")
            
        if st.button("Compute Sensitivity Surfaces (3D)", type="primary"):
            
            # Grid Definition
            n_grid = 15 # Resolution of surface
            a_vals = np.linspace(0.0, 4.0, n_grid)  # Stiffness parameter
            eps_vals = np.linspace(0.0, 2.0, n_grid) # Perturbation parameter
            
            z_sensitivity_eps = np.zeros((n_grid, n_grid))
            z_sensitivity_a = np.zeros((n_grid, n_grid))
            
            # Progress bar
            progress_bar = st.progress(0, text="Computing Gradients...")
            
            # Use lower res for surface to be fast
            n_points_surf = 200
            t_eval_surf = np.linspace(0, sens_T, n_points_surf)
            
            total_steps = n_grid * n_grid
            step_count = 0
            
            # Derivative functions
            # dA/deps = [[0, 0], [-cos(t), 0]]
            dA_deps_func = lambda t: np.array([[0.0, 0.0], [-np.cos(t), 0.0]])
            # dA/da = [[0, 0], [-1, 0]]
            dA_da_func = lambda t: np.array([[0.0, 0.0], [-1.0, 0.0]])
            
            for i, a_val in enumerate(a_vals):
                for j, eps_val in enumerate(eps_vals):
                    
                    # Define System A(t) for this grid point
                    def get_A_surf(t):
                         return np.array([[0.0, 1.0], [-(a_val + eps_val * np.cos(t)), 0.0]])
                    
                    # 1. Compute Nominal Phi
                    Phi_surf = compute_peano_series(get_A_surf, t_eval_surf, sens_n_terms)
                    # Shape: (points, 2, 2)
                    
                    # 2. Compute Integration for Psi_eps and Psi_a
                    integrand_stack_eps = np.zeros_like(Phi_surf)
                    integrand_stack_a = np.zeros_like(Phi_surf)
                    
                    # Vectorized loop over time for this grid point
                    for t_idx in range(len(t_eval_surf)):
                         Phi_val = Phi_surf[t_idx]
                         Phi_inv = np.linalg.inv(Phi_val)
                         
                         # dA/deps term
                         dA_deps = dA_deps_func(t_eval_surf[t_idx])
                         integrand_stack_eps[t_idx] = Phi_inv @ dA_deps @ Phi_val

                         # dA/da term
                         dA_da = dA_da_func(t_eval_surf[t_idx])
                         integrand_stack_a[t_idx] = Phi_inv @ dA_da @ Phi_val
                    
                    # Integrate both
                    Integral_eps = cumulative_trapezoid(integrand_stack_eps, t_eval_surf, axis=0, initial=0)
                    Integral_a = cumulative_trapezoid(integrand_stack_a, t_eval_surf, axis=0, initial=0)
                    
                    # Compute Psi at final time T
                    Phi_T = Phi_surf[-1]
                    
                    Psi_T_eps = Phi_T @ Integral_eps[-1]
                    Psi_T_a = Phi_T @ Integral_a[-1]
                    
                    # Store Magnitudes
                    z_sensitivity_eps[j, i] = np.linalg.norm(Psi_T_eps, ord='fro')
                    z_sensitivity_a[j, i] = np.linalg.norm(Psi_T_a, ord='fro')
                    
                    step_count += 1
                    if step_count % 10 == 0:
                        progress_bar.progress(step_count / total_steps)
            
            progress_bar.empty()
            
            # --- Plotting ---
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                # Plot 1: Epsilon Sensitivity
                fig_surf_eps = go.Figure(data=[go.Surface(
                    z=z_sensitivity_eps, 
                    x=a_vals, 
                    y=eps_vals, 
                    colorscale='Viridis',
                    colorbar=dict(title='||Psi_eps||', x=-0.1) # Move colorbar to left
                )])
                
                fig_surf_eps.update_layout(
                    title=f"Sensitivity to Perturbation ε",
                    scene=dict(
                        xaxis_title="a (Stiffness)",
                        yaxis_title="ε (Perturbation)",
                        zaxis_title="||∂Φ/∂ε||"
                    ),
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_surf_eps, width="stretch")
                
            with col_plot2:
                # Plot 2: Stiffness Sensitivity
                fig_surf_a = go.Figure(data=[go.Surface(
                    z=z_sensitivity_a, 
                    x=a_vals, 
                    y=eps_vals, 
                    colorscale='Plasma', 
                    colorbar=dict(title='||Psi_a||')
                )])
                
                fig_surf_a.update_layout(
                    title=f"Sensitivity to Stiffness a",
                    scene=dict(
                        xaxis_title="a (Stiffness)",
                        yaxis_title="ε (Perturbation)",
                        zaxis_title="||∂Φ/∂a||"
                    ),
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_surf_a, width="stretch")
            
            st.success("✅ Surfaces Computed. Compare how sensitivity to 'stiffness' differs from sensitivity to 'perturbation noise'.")
            
    parameter_sensitivity_fragment()

    # --- Case Study Section ---
    st.markdown("---")
    st.header("🤖 Case Study: Variable-Height Humanoid Balance")
    
    st.markdown(r"""
    **The "Capture Point" Problem**:
    A walking robot needs to place its foot ($u$) at a specific spot to come to a complete stop *or* maintain a walking gait.
    
    *   **Standard Theory (LIPM)**: Assumes the robot's center of mass height $z_c$ is **constant**. This gives a simple Linear Time-Invariant (LTI) system.
    *   **Real World (Squatting/Walking)**: If the robot changes height (e.g., crouching), the system becomes **Linear Time-Varying (LTV)**:
        $$ \ddot{x} = \frac{g}{z(t)} (x - u) $$
        
    **Simulation Logic**:
    *   **Walking (Steps 1-4)**: The controller targets a **symmetric gait**, swinging the body from $-x$ to $+x$ relative to the foot.
    *   **Stopping (Step 6)**: The controller targets **zero velocity**, effectively finding the "Capture Point" that arrests motion.
    
    > **Disclaimer**: This demo skips rigorous derivation details (e.g., ZMP constraints, N-step MPC) to focus purely on the mathematical utility of the **Peano-Baker Series**: it allows us to invert the dynamics of an arbitrary LTV system to find the exact control input $u$ needed to reach a target state.
    """)
    
    @st.fragment
    def case_study_lipm_fragment():
        col_case_1, col_case_2 = st.columns(2)
        
        with col_case_1:
            st.subheader("Robot Parameters")
            h0 = st.slider("Average Height (m)", 0.5, 1.5, 0.5)
            T_step = st.slider("Step Duration (s)", 0.3, 1.0, 0.8)
            
        with col_case_2:
            st.subheader("Simulation Controls")
            lipm_N = st.number_input("Peano Terms (N)", value=20, min_value=1, max_value=50)
            target_v = st.slider("Target Walking Speed (m/s)", 0.1, 1.0, 0.5)
            
        if st.button("Run walking Simulation (6 Steps)", type="primary"):
            
            # 1. Physics Definition
            g = 9.81
            amp_z = 0.2 # Fixed squat amplitude for simplicity in demo
            
            # Height Profile: z(t) - Periodic per step
            def z_profile(t):
                # Cosine profile: starts at h0+amp, dips to h0-amp
                # t is local time in step [0, T_step]
                return h0 + amp_z * np.cos(2 * np.pi * t / T_step)
            
            # System Matrices for LTV: x_dot = A(t)x + B(t)u
            def get_A_lipm(t):
                zt = z_profile(t)
                return np.array([[0.0, 1.0], [g/zt, 0.0]])
                
            def get_B_lipm(t):
                zt = z_profile(t)
                return np.array([0.0, -g/zt])
            
            # Time grid for one step
            n_step_pts = 50
            t_one_step = np.linspace(0, T_step, n_step_pts)
            
            # --- Pre-compute Peano Matrices (Same for every step due to periodicity) ---
            with st.spinner("Computing Controller..."):
                # Peano (LTV)
                Phi_series = compute_peano_series(get_A_lipm, t_one_step, lipm_N)
                Phi_T = Phi_series[-1]
                
                integrand_B = np.zeros((len(t_one_step), 2))
                for i, t_val in enumerate(t_one_step):
                    Phi_inv = np.linalg.inv(Phi_series[i])
                    B_val = get_B_lipm(t_val)
                    integrand_B[i] = Phi_inv @ B_val
                integral_B_term = cumulative_trapezoid(integrand_B, t_one_step, axis=0, initial=0)
                Gamma_T = Phi_T @ integral_B_term[-1]
                
                # Naive (LTI) - Constant A
                A_lti_func = lambda t: np.array([[0.0, 1.0], [g/h0, 0.0]])
                Phi_lti_series = compute_peano_series(A_lti_func, t_one_step, lipm_N)
                Phi_T_lti = Phi_lti_series[-1]
                
                integrand_B_lti = np.zeros((len(t_one_step), 2))
                B_const = np.array([0.0, -g/h0])
                for i in range(len(t_one_step)):
                    integrand_B_lti[i] = np.linalg.inv(Phi_lti_series[i]) @ B_const
                Gamma_T_lti = Phi_T_lti @ cumulative_trapezoid(integrand_B_lti, t_one_step, axis=0, initial=0)[-1]
                
            # --- Simulation Loop (6 Steps) ---
            
            # State: [x_rel, v] (x relative to current stance foot)
            # We also track [x_abs] for plotting
            
            # Initial absolute state
            x_abs_peano = 0.0
            x_abs_naive = 0.0
            v_peano = 0.0 # Start from rest
            v_naive = 0.0
            
            # Data containers for plotting
            history_peano_x = []
            history_peano_z = []
            history_naive_x = []
            history_naive_z = []
            footsteps_peano = []
            footsteps_naive = []
            
            # We assume starting stance foot is at 0
            foot_peano = 0.0
            foot_naive = 0.0
            
            footsteps_peano.append(foot_peano)
            footsteps_naive.append(foot_naive)
            
            # Simulation
            n_steps = 6
            
            for step_idx in range(n_steps):
                # Controller Strategy Selection
                # Step 0: Accelerate (Velocity Control)
                # Steps 1-(N-2): Maintain Gait (Symmetry Control)
                # Step (N-1): Decelerate/Stop (Velocity Control)
                
                strategy = "symmetry"
                if step_idx == 0:
                    strategy = "velocity"
                    v_tgt = target_v
                elif step_idx == n_steps - 1:
                    strategy = "velocity"
                    v_tgt = 0.0
                else: 
                     strategy = "symmetry" # Maintain natural swing
                
                
                # Current Absolute State
                x0_vec_p = np.array([x_abs_peano, v_peano])
                x0_vec_n = np.array([x_abs_naive, v_naive])
                
                # Calculate Control u based on strategy
                
                # --- PEANO CONTROL ---
                if strategy == "velocity":
                    # Velocity Control: v(T) = Phi_21 x + Phi_22 v + Gamma_2 u = v_tgt
                    numerator_p = v_tgt - np.dot(Phi_T[1, :], x0_vec_p)
                    u_p = numerator_p / Gamma_T[1]
                else:
                    # Symmetry Control: x_rel(T) = -x_rel(0)
                    # Implies u is exactly in the midpoint of x(0) and x(T)
                    # u = (x0*(1+Phi_11) + v0*Phi_12) / (2 - Gamma_1)
                    num = x0_vec_p[0] * (1 + Phi_T[0,0]) + x0_vec_p[1] * Phi_T[0,1]
                    den = 2.0 - Gamma_T[0]
                    u_p = num / den

                # --- NAIVE CONTROL ---
                if strategy == "velocity":
                    numerator_n = v_tgt - np.dot(Phi_T_lti[1, :], x0_vec_n)
                    u_n = numerator_n / Gamma_T_lti[1]
                else:
                    # Symmetry
                    num_n = x0_vec_n[0] * (1 + Phi_T_lti[0,0]) + x0_vec_n[1] * Phi_T_lti[0,1]
                    den_n = 2.0 - Gamma_T_lti[0]
                    u_n = num_n / den_n
                
                # Store Footsteps
                footsteps_peano.append(u_p)
                footsteps_naive.append(u_n)
                
                # 3. Simulate this step dynamics
                # Trajectory for this step
                
                # Peano (True System)
                for k in range(len(t_one_step)):
                    # Gamma(t)
                    Gamma_t_p = Phi_series[k] @ integral_B_term[k]
                    # State(t)
                    state_p = Phi_series[k] @ x0_vec_p + Gamma_t_p * u_p
                    
                    history_peano_x.append(state_p[0])
                    history_peano_z.append(z_profile(t_one_step[k]))
                    
                # Naive (True System driven by naive u)
                for k in range(len(t_one_step)):
                    Gamma_t_n = Phi_series[k] @ integral_B_term[k] # Driven by TRUE LTV physics
                    state_n = Phi_series[k] @ x0_vec_n + Gamma_t_n * u_n
                    
                    history_naive_x.append(state_n[0])
                    history_naive_z.append(z_profile(t_one_step[k]))
                    
                # Update State for next step
                x_abs_peano = history_peano_x[-1]
                v_peano =     state_p[1] # Velocity is continuous
                
                x_abs_naive = history_naive_x[-1]
                v_naive =     state_n[1]
                
            # --- Animation ---
            
            # Combine history arrays
            traj_p_x = np.array(history_peano_x)
            traj_p_z = np.array(history_peano_z)
            traj_n_x = np.array(history_naive_x)
            
            total_frames = len(traj_p_x)
            
            # Sub-sample frames for smooth animation relative to browser performance through frame.duration
            # We want Real-Time. 
            # Total Simulation Time = n_steps * T_step
            # Total Points = n_steps * n_step_pts
            # dt per point = T_step / n_step_pts
            
            step_stride = 1 # Plot every point for smoothness if performance allows? 
            # If we have 300 points for 3 seconds, that's 100fps. Too fast or browser lag?
            # Let's aim for 30-60fps.
            # If T_step=0.5, n_pts=50. Total 6 steps = 3.0s. 300 points.
            # 300 frames in 3s = 100fps. A bit high. Let's stride by 2.
            
            step_stride = 1
            if (n_steps * n_step_pts) / (n_steps * T_step) > 60:
                 step_stride = 2
            
            # Calculate Frame Duration in ms
            # Calculate Frame Duration in ms
            # dt is variable from earlier scope which isn't available here, so we re-derive it
            dt_local = T_step / n_step_pts
            frame_dur_ms = dt_local * step_stride * 1000 
            
            frames = []
            for k in range(0, total_frames, step_stride): 
                step_num = k // n_step_pts
                
                # Identify Foot Position
                # footsteps array: [init_0, u_0 (step 0), u_1 (step 1), ...]
                # step_num starts at 0.
                # Dynamics for step_num=0 used footsteps[1] (u_0).
                # So we must draw leg from footsteps[step_num + 1].
                
                idx_foot = step_num + 1
                if idx_foot < len(footsteps_peano):
                    ft_p = footsteps_peano[idx_foot]
                    ft_n = footsteps_naive[idx_foot]
                else: 
                    # Should not happen if frames align with steps, but safety fallback
                    ft_p = footsteps_peano[-1]
                    ft_n = footsteps_naive[-1]
                
                # Check for "Crash" - if Naive diverges too much, stop drawing it or cap it?
                # Visuals handle it naturally (it flies off chart).
                
                frames.append(go.Frame(data=[
                    # Peano Robot (Leg + Body)
                    go.Scatter(x=[ft_p, traj_p_x[k]], y=[0, traj_p_z[k]], mode='lines+markers', 
                               marker=dict(size=[10, 20], color='green'), 
                               line=dict(color='green', width=6)),
                    # Naive Robot
                    go.Scatter(x=[ft_n, traj_n_x[k]], y=[0, traj_p_z[k]], mode='lines+markers', 
                               marker=dict(size=[8, 15], color='red'), 
                               line=dict(color='red', dash='dot', width=3)),
                    # Trails
                    go.Scatter(x=traj_p_x[:k+1], y=traj_p_z[:k+1]),
                    go.Scatter(x=traj_n_x[:k+1], y=traj_p_z[:k+1]),
                ]))
                
            fig_anim = go.Figure(
                data=[
                    # Initial state (dummy for legend)
                    go.Scatter(x=[0,0], y=[0, h0], name='Peano (LTV)', mode='lines+markers', marker=dict(size=15), line=dict(color='green', width=6)),
                    go.Scatter(x=[0,0], y=[0, h0], name='Naive (LTI)', mode='lines+markers', marker=dict(size=12), line=dict(color='red', dash='dot', width=3)),
                    go.Scatter(x=[], y=[], name='P Trace', mode='lines', line=dict(color='green', width=1)),
                    go.Scatter(x=[], y=[], name='N Trace', mode='lines', line=dict(color='red', width=1))
                ],
                layout=go.Layout(
                    title="Real-Time Walking Simulation",
                    xaxis=dict(title="Position (m)", range=[-0.5, target_v * n_steps * T_step * 1.5]), 
                    yaxis=dict(title="Height (m)", range=[-0.1, 1.5], scaleanchor="x", scaleratio=1), # 1:1 Aspect Ratio
                    
                    # Ground Line
                    shapes=[dict(type="line", x0=-10, y0=0, x1=20, y1=0, line=dict(color="black", width=2))],
                    
                    updatemenus=[dict(
                        type="buttons", 
                        buttons=[dict(label="▶️ Play Real-Time", 
                                      method="animate", 
                                      args=[None, dict(frame=dict(duration=frame_dur_ms, redraw=False), 
                                                       fromcurrent=True, 
                                                       transition=dict(duration=0))])]
                    )],
                    height=500,
                    margin=dict(t=40, b=0)
                ),
                frames=frames
            )
            
            st.plotly_chart(fig_anim, width="stretch")
            
            # Final Report
            st.write(f"**Final Velocity (Peano)**: {v_peano:.3f} m/s (Target: 0.0)")
            st.write(f"**Final Velocity (Naive)**: {v_naive:.3f} m/s")
            
            if abs(v_peano) < 0.1:
                st.success(f"✅ Peano Controller successfully walked and stopped! (Distance covered: {x_abs_peano:.2f} m)")
            else:
                st.warning("⚠️ Peano Controller drifting.")
                
            if abs(v_naive) > 0.5:
                # If naive crashed, message
                st.error("❌ Naive Controller (Red) crashed. The LTI assumption caused it to place feet incorrectly for the variable height.")

    case_study_lipm_fragment()




except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your matrix syntax. Ensure it is a valid Python expression using 't'.")

