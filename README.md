# Peano-Baker Series Visualizer

An interactive Streamlit web app that visualizes how the Peano-Baker series approximates the state transition matrix $\Phi(t, t_0)$ for a Linear Time-Varying (LTV) system.

## Setup & Run

1. Install dependencies:
   ```bash
   pip install -r peano_baker_viz/requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run peano_baker_viz/app.py
   ```

## Features

- **Interactive Matrix Input**: Define your own LTV system matrix $A(t)$ using Python syntax (e.g., `[[0, 1], [-1, -0.1*t]]`).
- **Visual Convergence**: Adjust the number of terms in the Peano-Baker series and watch the approximation converge to the true solution.
- **Phase Portrait**: Visualize the state trajectory in 2D.
- **Error Analysis**: Real-time plot of the approximation error over time.

## Project Structure

- `peano_baker_viz/app.py`: Main Streamlit application.
- `peano_baker_viz/utils.py`: Core mathematical logic (parsing, integration, solving).
- `peano_baker_viz/requirements.txt`: Python dependencies.