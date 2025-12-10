# Peano-Baker Series Visualizer

An interactive Streamlit web application that visualizes the **Peano-Baker series**, a powerful mathematical tool for approximating the state transition matrix $\Phi(t, t_0)$ of Linear Time-Varying (LTV) systems.

This tool aims to bridge the gap between abstract control theory and intuitive understanding by providing real-time visualizations, error bounds, and practical control case studies.

## 🚀 Key Features

### 1. Interactive Simulation
*   **Custom LTV Systems**: Define your own time-varying matrix $A(t)$ using Python syntax (e.g., `[[0, 1], [-2 - 0.5*cos(t), -1]]`) or choose from preset physical systems like the Damped Oscillator or Mathieu Equation.
*   **Real-Time Phase Portraits**: Visualize 2D state trajectories ($x_1$ vs $x_2$) and time-series evolution.
*   **Visual Convergence**: Adjust the number of series terms ($N$) and watch the approximation converge to the true solution in real-time.

### 2. Rigorous Error Analysis
*   ** Theoretical Upper Bounds**: Compares the actual numerical error against the theoretical analytical upper bound derived from the work of Baake & Schlägel.
*   **Validity Horizon**: Automatically calculates the time horizon for which the approximation is guaranteed to stay within a specified error threshold.

### 3. Sensitivity & Stability
*   **Parameter Sensitivity**: Explore how changes in initial conditions ($x_0$) and system dynamics magnitude ($\|A\|$) affect convergence.
*   **Gradient Visualization**: Compute and visualize 3D surfaces showing the sensitivity of the solution to system parameters (e.g., stiffness vs. perturbation).

### 4. Advanced Analytics
*   **Convergence Heatmap**: A "Wavefront" visualization showing the error magnitude across Time vs. Number of Terms.
*   **Term Magnitude**: Analyze the Frobenius norm of individual series terms to verify decay and convergence.
*   **Computation Benchmarks**: Analyze the $O(N \cdot M)$ complexity of the cumulative integration scheme.

### 5. Case Study: Robot Walking Control
*   **Variable-Height LIPM**: A demonstration of LTV control applied to a walking humanoid robot that changes height (squatting).
*   **LTI vs LTV**: Visually compares a standard "Naive" controller (assuming constant height) vs. the "Peano" controller (accounting for time-varying height), showing why LTV theory is critical for dynamic robots.

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   pip

### 1. Install Dependencies
Clone the repository, create a virtual environment, and install the required Python packages:

```bash
git clone https://github.com/yourusername/Peano-Baker-Series-Visualizer.git
cd Peano-Baker-Series-Visualizer

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r peano_baker_viz/requirements.txt
```

### 2. Run Locally
Start the Streamlit development server:

```bash
streamlit run peano_baker_viz/app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🌐 Deployment to GitHub Pages

This project is deployed to **GitHub Pages** using **stlite** (Serverless Streamlit). 

### How it works
Unlike traditional Streamlit apps that require a backend Python server, this project uses [stlite](https://github.com/whitphx/stlite) to run the entire application directly in the user's browser using WebAssembly (Pyodide).

1.  **Static Files**: The `docs/` directory contains a static `index.html` file and copies of the Python source code (`app.py`, `utils.py`, `case_studies.py`).
2.  **WebAssembly Runtime**: When a user visits the page, `index.html` loads the `stlite` runtime, which downloads the Pyodide Python interpreter.
3.  **Browser Execution**: The Python code is executed locally within the browser tab. This eliminates the need for a dedicated backend server, allowing free hosting on GitHub Pages.

GitHub Pages is configured to serve the `/docs` folder from the main branch.

---

## 📂 Project Structure

*   `peano_baker_viz/`: Source code for local development.
    *   `app.py`: Main application entry point.
    *   `utils.py`: Core mathematical engine.
    *   `case_studies.py`: Robot simulation logic.
*   `docs/`: Static site assets for GitHub Pages deployment (contains `index.html` and mirrors the python files).

The pages webiste hosts a preview of the streamlit app. It is not exactly the same as what is seen if running the streamlit app itself.

You can see the app in action at [Peano-Baker-Series-Visualizer](https://ttchalakov.github.io/Peano-Baker-Series-Visualizer/).