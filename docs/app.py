import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import parse_matrix_func, compute_peano_series, compute_true_solution

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

T_horizon = st.sidebar.slider("Time Horizon (T)", min_value=1.0, max_value=20.0, value=10.0)

# Initial Conditions
st.sidebar.subheader("Initial Conditions x(0)")
x1_0 = st.sidebar.number_input("x1(0)", value=1.0)
x2_0 = st.sidebar.number_input("x2(0)", value=0.0)
x0 = np.array([x1_0, x2_0])

# Cached Computation Functions
@st.cache_data
def get_true_solution(matrix_str, T, x0):
    A_func = parse_matrix_func(matrix_str)
    t_eval = np.linspace(0, T, 1000)
    return compute_true_solution(A_func, t_eval, x0), t_eval

@st.cache_data
def get_peano_series(matrix_str, T, n):
    A_func = parse_matrix_func(matrix_str)
    t_eval = np.linspace(0, T, 1000)
    return compute_peano_series(A_func, t_eval, n)

# Computations
try:
    # Parse Matrix (needed for sensitivity analysis later)
    A_func = parse_matrix_func(matrix_str)

    # Compute True Solution
    with st.spinner("Computing True Solution..."):
        x_true, t_eval = get_true_solution(matrix_str, T_horizon, x0)
        
    # Compute Peano Approximation
    # We don't need a spinner for this if it's fast/cached
    Phi_approx = get_peano_series(matrix_str, T_horizon, n_terms)
    
    # x_approx(t) = Phi(t, t0) * x0
    x_approx = np.einsum('ijk,k->ij', Phi_approx, x0)
        
    # Compute bounds based on True Solution for scaling
    x1_min, x1_max = np.min(x_true[:, 0]), np.max(x_true[:, 0])
    x2_min, x2_max = np.min(x_true[:, 1]), np.max(x_true[:, 1])
    
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
        fig_phase.add_trace(go.Scatter(x=x_true[:, 0], y=x_true[:, 1], mode='lines', name='True Solution', line=dict(color='blue')))
        fig_phase.add_trace(go.Scatter(x=x_approx[:, 0], y=x_approx[:, 1], mode='lines', name='Peano Approx', line=dict(color='red', dash='dash')))
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
        fig_time.add_trace(go.Scatter(x=t_eval, y=x_true[:, 0], mode='lines', name='x1 (True)', line=dict(color='blue')))
        fig_time.add_trace(go.Scatter(x=t_eval, y=x_approx[:, 0], mode='lines', name='x1 (Approx)', line=dict(color='blue', dash='dash')))
        fig_time.add_trace(go.Scatter(x=t_eval, y=x_true[:, 1], mode='lines', name='x2 (True)', line=dict(color='green')))
        fig_time.add_trace(go.Scatter(x=t_eval, y=x_approx[:, 1], mode='lines', name='x2 (Approx)', line=dict(color='green', dash='dash')))
        fig_time.update_layout(
            xaxis_title="Time (t)", yaxis_title="State Value", height=400,
            yaxis_range=y_bounds
        )
        st.plotly_chart(fig_time, width="stretch")
        
    # Plot 3: Error Convergence
    st.subheader("Approximation Error over Time")
    use_log_scale = st.checkbox("Log Scale", value=True)
    
    # Error = ||x_true - x_approx||
    error = np.linalg.norm(x_true - x_approx, axis=1)
    
    # Theoretical Bound
    # We need compute_error_bound from utils
    from utils import compute_error_bound
    
    @st.cache_data
    def get_error_bound(matrix_str, T, n, x0_norm):
        A_func = parse_matrix_func(matrix_str)
        t_eval = np.linspace(0, T, 1000)
        return compute_error_bound(A_func, t_eval, n, x0_norm)
        
    bound = get_error_bound(matrix_str, T_horizon, n_terms, np.linalg.norm(x0))
    
    # Calculate Validity Horizon (time until error bound > tolerance)
    tolerance = 0.1 # 10% error relative to unit norm (arbitrary but useful)
    # Find first index where bound > tolerance
    valid_indices = np.where(bound > tolerance)[0]
    if len(valid_indices) > 0:
        valid_time = t_eval[valid_indices[0]]
        valid_msg = f"{valid_time:.2f} s"
    else:
        valid_time = T_horizon
        valid_msg = f"> {T_horizon:.2f} s"

    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Max Theoretical Error", f"{np.max(bound):.2e}")
    col_metric2.metric("Validity Horizon (Bound < 0.1)", valid_msg)
    
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(x=t_eval, y=error, mode='lines', name='Actual Error', line=dict(color='orange')))
    fig_error.add_trace(go.Scatter(x=t_eval, y=bound, mode='lines', name='Theoretical Bound', line=dict(color='gray', dash='dot')))
    
    yaxis_type = "log" if use_log_scale else "linear"
    fig_error.update_layout(xaxis_title="Time (t)", yaxis_title="Error ||x_true - x_approx||", height=300, yaxis_type=yaxis_type)
    st.plotly_chart(fig_error, width="stretch")

    with st.expander("ℹ️ How is the Upper Bound derived?"):
        st.markdown(r"""
        The **Theoretical Bound** (dotted gray line) represents the worst-case error for a Peano-Baker series truncated after $N$ terms. It is derived by comparing the norm of the matrix series to the scalar exponential series.

        **The Bound Formula:**
        $$
        ||\text{Error}(t)|| \le \left( e^{L(t)} - \sum_{k=0}^{N-1} \frac{L(t)^k}{k!} \right) ||x_0||
        $$

        Where $L(t)$ is the cumulative integral of the system's "speed" (matrix norm):
        $$
        L(t) = \int_{t_0}^t ||A(\sigma)|| d\sigma
        $$

        **Why does the error diverge?**
        The Peano-Baker series behaves like a "matrix Taylor series". Just like the Taylor series for $e^x$ ($1 + x + x^2/2! + \dots$) converges for all $x$ but requires more terms as $x$ gets larger, the Peano-Baker series requires more terms as the "cumulative magnitude" $L(t)$ grows.
        
        *   **Small $L(t)$** (short time or slow dynamics): The first few terms capture almost all the behavior.
        *   **Large $L(t)$** (long time or fast dynamics): The truncated series fails to keep up with the true exponential growth, causing the error to explode (often exponentially) once $L(t)$ exceeds the "capacity" of the $N$ terms you selected.
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
        x_true_scaled = compute_true_solution(A_func, t_eval, x0_scaled)
        x_approx_scaled = np.einsum('ijk,k->ij', Phi_approx, x0_scaled)
        error_scaled = np.linalg.norm(x_true_scaled - x_approx_scaled, axis=1)
        
        # Plot comparison
        fig_sens_x0 = go.Figure()
        fig_sens_x0.add_trace(go.Scatter(x=t_eval, y=error, mode='lines', name=f'Error (Original x0)', line=dict(color='orange')))
        fig_sens_x0.add_trace(go.Scatter(x=t_eval, y=error_scaled, mode='lines', name=f'Error (Scaled x0 * {alpha})', line=dict(color='purple', dash='dot')))
        
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
                x_true_beta = compute_true_solution(A_func_scaled, t_eval, x0)
                Phi_approx_beta = compute_peano_series(A_func_scaled, t_eval, n_terms)
                x_approx_beta = np.einsum('ijk,k->ij', Phi_approx_beta, x0)
                error_beta = np.linalg.norm(x_true_beta - x_approx_beta, axis=1)
                
            # Plot comparison
            fig_sens_A = go.Figure()
            fig_sens_A.add_trace(go.Scatter(x=t_eval, y=error, mode='lines', name=f'Error (Original A)', line=dict(color='orange')))
            fig_sens_A.add_trace(go.Scatter(x=t_eval, y=error_beta, mode='lines', name=f'Error (Scaled A * {beta})', line=dict(color='red', dash='dot')))
            
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
    
    # We need compute_peano_history
    from utils import compute_peano_history
    
    @st.cache_data
    def get_peano_history(matrix_str, T, n):
        A_func = parse_matrix_func(matrix_str)
        t_eval = np.linspace(0, T, 1000)
        return compute_peano_history(A_func, t_eval, n)

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
                    
                    history = get_peano_history(matrix_str, T_horizon, heatmap_n)
                    # history shape: (n_terms, n_points, n, n)
                    
                    # Compute error for each term
                    # We need x_true
                    # x_true shape: (n_points, n)
                    
                    # Calculate errors
                    # errors shape: (n_terms, n_points)
                    errors = np.zeros((heatmap_n, len(t_eval)))
                    
                    for k in range(heatmap_n):
                        Phi_k = history[k]
                        x_approx_k = np.einsum('ijk,k->ij', Phi_k, x0)
                        errors[k] = np.linalg.norm(x_true - x_approx_k, axis=1)
                        
                    # Log scale for heatmap
                    # Add small epsilon to avoid log(0)
                    log_errors = np.log10(errors + 1e-16)
                    
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=log_errors,
                        x=t_eval,
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
                    history = get_peano_history(matrix_str, T_horizon, anim_n)
                    
                    # Loop to generate frames, but stop if converged to avoid numerical instability
                    frames = []
                    actual_anim_n = 0
                    
                    for k in range(anim_n):
                        Phi_k = history[k]
                        x_approx_k = np.einsum('ijk,k->ij', Phi_k, x0)
                        
                        # Check convergence
                        current_error = np.max(np.linalg.norm(x_true - x_approx_k, axis=1))
                        
                        frame = go.Frame(
                            data=[
                                 go.Scatter(x=x_true[:, 0], y=x_true[:, 1], mode='lines', name='True Solution', line=dict(color='blue')),
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
                            go.Scatter(x=x_true[:, 0], y=x_true[:, 1], mode='lines', name='True Solution', line=dict(color='blue')),
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
                    t_eval_analysis = np.linspace(0, T_horizon, 1000)
                    terms = compute_peano_terms(A_func_analysis, t_eval_analysis, analysis_n)
                    # terms shape: (n_terms, n_points, n, n)
                    
                    # Compute norm of each term at final time T
                    # We take the last point: terms[:, -1, :, :]
                    final_terms = terms[:, -1, :, :]
                    
                    # Frobenius norm for each k
                    magnitudes = np.array([np.linalg.norm(term, ord='fro') for term in final_terms])
                    
                    # Create Bar Plot
                    fig_terms = go.Figure(data=go.Bar(
                        x=np.arange(analysis_n),
                        y=magnitudes,
                        marker_color='teal'
                    ))
                    
                    fig_terms.update_layout(
                        title=f"Magnitude of Terms at t={T_horizon}",
                        xaxis_title="Term Index (k)",
                        yaxis_title="Norm ||Phi_k(T)||",
                        yaxis_type="log", # Log scale is crucial to see decay
                        height=500
                    )
                    
                    st.plotly_chart(fig_terms, width="stretch")
                    
                    # Analysis
                    max_term_idx = np.argmax(magnitudes)
                    st.write(f"**Dominant Term**: k={max_term_idx} (Magnitude: {magnitudes[max_term_idx]:.2e})")
                    
                    if magnitudes[-1] < 1e-15:
                        st.success("✅ The series has converged to machine precision.")
                    elif magnitudes[-1] < 1e-6:
                        st.success("✅ The series is converging well.")
                    elif magnitudes[-1] > magnitudes[0]:
                        st.error("⚠️ The series is diverging! The terms are getting larger.")
                    else:
                        st.warning("⚠️ The series has not fully converged yet.")
        
        term_analysis_fragment()

    # --- Case Study Section ---
    st.markdown("---")
    st.header("🚀 Case Study: Satellite Rendezvous in Elliptical Orbits")
    
    st.markdown(r"""
    This case study applies the Peano-Baker series to a **real-world 6-dimensional LTV system**: 
    the **Tschauner-Hempel equations** for spacecraft relative motion.
    
    ### The Problem
    When a "chaser" satellite approaches a "target" satellite (like ISS resupply missions), 
    we need to predict the relative trajectory. In elliptical orbits, the dynamics are 
    **time-varying** because the orbital angular rate changes with position.
    
    ### State Vector (6D)
    $$\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$$
    
    Where $(x, y, z)$ are relative positions in the **LVLH frame**:
    - **x**: Radial (outward from Earth)
    - **y**: In-track (along velocity)  
    - **z**: Cross-track (out of orbital plane)
    
    ### Why LTV?
    The system matrix $A(t)$ depends on the **true anomaly** $\theta(t)$, which varies with 
    orbital position. Higher eccentricity → more time variation → harder to approximate!
    """)
    
    # Import case study functions
    from case_studies import (
        create_spacecraft_A_func, 
        solve_relative_motion_ode,
        ISS_ORBITAL_PERIOD,
        DEFAULT_INITIAL_STATE
    )
    from utils import compute_peano_series
    
    # Controls in columns
    case_col1, case_col2 = st.columns(2)
    
    with case_col1:
        st.subheader("Orbital Parameters")
        case_eccentricity = st.slider(
            "Eccentricity (e)", 
            min_value=0.0, 
            max_value=0.7, 
            value=0.1, 
            step=0.05,
            help="0 = circular, higher = more elliptical (more time-varying)"
        )
        
        case_orbits = st.slider(
            "Simulation Duration (orbits)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        case_n_terms = st.slider(
            "Peano-Baker Terms",
            min_value=3,
            max_value=30,
            value=10
        )
    
    with case_col2:
        st.subheader("Initial Relative Position")
        init_x = st.number_input("x₀ (radial, m)", value=-100.0)
        init_y = st.number_input("y₀ (in-track, m)", value=-1000.0)
        init_z = st.number_input("z₀ (cross-track, m)", value=50.0)
        
        st.caption("Initial velocities are set to zero (co-orbiting)")
    
    case_x0 = np.array([init_x, init_y, init_z, 0.0, 0.0, 0.0])
    T_case = case_orbits * ISS_ORBITAL_PERIOD
    
    @st.fragment
    def case_study_fragment():
        if st.button("🛰️ Run Rendezvous Simulation", key="case_study_btn"):
            with st.spinner("Computing spacecraft trajectories..."):
                # Create A(t) function
                A_func_case = create_spacecraft_A_func(case_eccentricity, ISS_ORBITAL_PERIOD)
                
                # Time evaluation points
                n_points = 500
                t_eval_case = np.linspace(0, T_case, n_points)
                
                # Compute true solution using ODE solver
                x_true_case = solve_relative_motion_ode(
                    A_func_case, case_x0, (0, T_case), t_eval_case
                )
                
                # Compute Peano-Baker approximation
                Phi_approx = compute_peano_series(A_func_case, t_eval_case, case_n_terms)
                # Apply to initial condition: x(t) = Phi(t) @ x0
                x_approx_case = np.einsum('ijk,k->ij', Phi_approx, case_x0)
                
                # Compute errors
                errors_case = np.linalg.norm(x_true_case - x_approx_case, axis=1)
                
                # --- Visualizations ---
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # 3D Trajectory Plot
                    st.subheader("3D Relative Trajectory")
                    
                    fig_3d = go.Figure()
                    
                    # True trajectory
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_true_case[:, 1],  # y = in-track
                        y=x_true_case[:, 0],  # x = radial
                        z=x_true_case[:, 2],  # z = cross-track
                        mode='lines',
                        name='True (ODE)',
                        line=dict(color='blue', width=4)
                    ))
                    
                    # Peano-Baker approximation
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_approx_case[:, 1],
                        y=x_approx_case[:, 0],
                        z=x_approx_case[:, 2],
                        mode='lines',
                        name=f'Peano-Baker (N={case_n_terms})',
                        line=dict(color='red', width=3, dash='dash')
                    ))
                    
                    # Target satellite at origin
                    fig_3d.add_trace(go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode='markers',
                        name='Target Satellite',
                        marker=dict(size=10, color='gold', symbol='diamond')
                    ))
                    
                    # Initial position
                    fig_3d.add_trace(go.Scatter3d(
                        x=[case_x0[1]], y=[case_x0[0]], z=[case_x0[2]],
                        mode='markers',
                        name='Initial Position',
                        marker=dict(size=8, color='green')
                    ))
                    
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title="In-track (m)",
                            yaxis_title="Radial (m)",
                            zaxis_title="Cross-track (m)",
                            aspectmode='data'
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with viz_col2:
                    # Error over time
                    st.subheader("Approximation Error")
                    
                    t_minutes = t_eval_case / 60
                    
                    fig_err = go.Figure()
                    fig_err.add_trace(go.Scatter(
                        x=t_minutes,
                        y=errors_case,
                        mode='lines',
                        name='||x_true - x_approx||',
                        line=dict(color='red')
                    ))
                    
                    fig_err.update_layout(
                        xaxis_title="Time (minutes)",
                        yaxis_title="Error (m)",
                        yaxis_type="log",
                        height=250
                    )
                    st.plotly_chart(fig_err, use_container_width=True)
                    
                    # State components over time
                    st.subheader("Position Components")
                    
                    fig_states = go.Figure()
                    labels = ['x (radial)', 'y (in-track)', 'z (cross-track)']
                    colors = ['blue', 'green', 'purple']
                    
                    for i in range(3):
                        fig_states.add_trace(go.Scatter(
                            x=t_minutes,
                            y=x_true_case[:, i],
                            mode='lines',
                            name=f'{labels[i]} (true)',
                            line=dict(color=colors[i])
                        ))
                        fig_states.add_trace(go.Scatter(
                            x=t_minutes,
                            y=x_approx_case[:, i],
                            mode='lines',
                            name=f'{labels[i]} (approx)',
                            line=dict(color=colors[i], dash='dash'),
                            opacity=0.7
                        ))
                    
                    fig_states.update_layout(
                        xaxis_title="Time (minutes)",
                        yaxis_title="Position (m)",
                        height=250
                    )
                    st.plotly_chart(fig_states, use_container_width=True)
                
                # Summary metrics
                max_error = np.max(errors_case)
                mean_error = np.mean(errors_case)
                final_error = errors_case[-1]
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Max Error", f"{max_error:.2e} m")
                with metric_cols[1]:
                    st.metric("Mean Error", f"{mean_error:.2e} m")
                with metric_cols[2]:
                    st.metric("Final Error", f"{final_error:.2e} m")
                
                # Analysis
                if max_error < 1.0:
                    st.success(f"✅ Excellent approximation! Max error < 1 meter over {case_orbits:.1f} orbit(s).")
                elif max_error < 100.0:
                    st.info(f"ℹ️ Good approximation for mission planning. Consider more terms for higher precision.")
                else:
                    st.warning(f"⚠️ Large errors detected. Try increasing the number of terms or reducing simulation duration.")
                
                st.markdown(f"""
                **Observations:**
                - Eccentricity = **{case_eccentricity}** ({"circular" if case_eccentricity < 0.01 else "elliptical"} orbit)
                - The higher the eccentricity, the more time-varying the dynamics
                - More terms are needed for accurate approximation of elliptical orbits
                - This 6D LTV system demonstrates Peano-Baker on a real aerospace problem!
                """)
    
    case_study_fragment()


except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your matrix syntax. Ensure it is a valid Python expression using 't'.")

