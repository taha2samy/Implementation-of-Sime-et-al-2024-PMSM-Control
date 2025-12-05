# High-Performance PMSM Speed Control for Electric Vehicles


[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


![Uploading image.png…]()

This repository contains the implementation and simulation of an advanced intelligent control system for a Permanent Magnet Synchronous Motor (PMSM) drive, specifically tailored for Electric Vehicle (EV) applications. The project is developed as an interactive web-based dashboard using **Marimo**.

The core of the project is the implementation of a **Switched Adaptive Fuzzy Fractional Order PID (AFFOPID)** controller, inspired by the methodologies presented in the paper by Sime, T.L., et al. (2024). The system is automatically tuned using a **Genetic Algorithm (GA)** to achieve optimal performance.

---

## 🚀 Key Features

*   **Advanced Control Strategy:** Implements a state-of-the-art switched AFFOPID controller that combines Fractional Order calculus, a dual-stage Adaptive Fuzzy Logic system, and an intelligent switching mechanism for superior performance in both transient and steady states.
*   **Interactive Dashboard:** A fully interactive `marimo` dashboard allows for real-time tuning of controller parameters, simulation scenarios, and visualization of results.
*   **Automated Tuning:** Includes a built-in Genetic Algorithm (GA) console for automatically optimizing the controller's numerous parameters to minimize error and improve response time.
*   **Comprehensive Analysis:** The dashboard provides in-depth analysis tools, including dynamic performance metrics, intelligent controller behavior visualization, GA convergence history, and system stability plots (Phase Plane).

---

## 🔧 Getting Started

### Prerequisites

*   Python 3.9+
*   Poetry (or `pip`) for dependency management.

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/taha2samy/Implementation-of-Sime-et-al-2024-PMSM-Control
    cd Implementation-of-Sime-et-al-2024-PMSM-Control
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Marimo dashboard:**
    ```bash
    marimo run pmsm_project.py
    ```

4.  Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:2718`).

---

## 📚 Detailed Documentation

For a complete technical breakdown of the system modeling, controller architecture, GA methodology, and a full analysis of the results, please refer to the detailed documentation directory:

**[➡️ Go to Detailed Documentation](./docs/README.md)**

---
