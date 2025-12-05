import marimo

__generated_with = "0.18.1"
app = marimo.App(width="columns", layout_file="layouts/pmsm_project.grid.json")


@app.cell(column=0, hide_code=True)
def _():

    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.special as special  # Required for Gamma functions in Fractional Calculus
    from dataclasses import dataclass
    from typing import Dict, List, Tuple

    # --- Visualization Configuration ---
    # Set a professional academic style for all plots generated in this notebook
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 120,                 # High resolution for standard screens
        'figure.figsize': (10, 6),         # Default aspect ratio
        'font.size': 10,
        'lines.linewidth': 2,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'        # Clean, modern font
    })

    # --- Project Header ---
    # Renders the dashboard title and abstract information
    mo.md(
        """
        <div style="
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #ffffff;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.1);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            text-align: center;
            margin-bottom: 2rem;
        ">
            <!-- Pre-title Tag -->
            <div style="
                display: inline-block;
                background: rgba(255,255,255,0.1);
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                font-weight: 600;
                margin-bottom: 1.2rem;
                border: 1px solid rgba(255,255,255,0.15);
            ">
                Advanced Control Simulation
            </div>

            <!-- Main Title -->
            <h1 style="
                margin: 0;
                font-size: 2.8rem;
                font-weight: 800;
                letter-spacing: -1px;
                background: linear-gradient(to right, #ffffff, #e0e0e0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 2px 10px rgba(0,0,0,0.3);
            ">
                GA-AFFOPID Control System
            </h1>

            <!-- Subtitle -->
            <p style="
                margin: 1rem 0 2rem 0;
                font-size: 1.15rem;
                font-weight: 300;
                color: #d1d5db;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            ">
                High-performance speed control simulation for <strong>Permanent Magnet Synchronous Motors (PMSM)</strong> 
                in Electric Vehicles, utilizing Genetic Algorithm optimization, Adaptive Fuzzy Logic, and Fractional Order calculus.
            </p>

            <!-- Tech Stack Badges -->
            <div style="
                display: flex;
                justify-content: center;
                gap: 12px;
                flex-wrap: wrap;
            ">
                <span style="background: rgba(56, 178, 172, 0.15); border: 1px solid rgba(56, 178, 172, 0.3); color: #81e6d9; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 6px;">
                    ⚡ EV Dynamics
                </span>
                <span style="background: rgba(66, 153, 225, 0.15); border: 1px solid rgba(66, 153, 225, 0.3); color: #90cdf4; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 6px;">
                    🧬 Genetic Algorithm
                </span>
                <span style="background: rgba(236, 201, 75, 0.15); border: 1px solid rgba(236, 201, 75, 0.3); color: #f6e05e; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 6px;">
                    🧠 Adaptive Fuzzy
                </span>
                <span style="background: rgba(159, 122, 234, 0.15); border: 1px solid rgba(159, 122, 234, 0.3); color: #d6bcfa; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 6px;">
                    ∫ Fractional Order PID
                </span>
            </div>
        </div>
        """
    )
    return go, make_subplots, mo, np


@app.cell
def _(mo):
    # 1. Constants
    OPT_PID = "Conventional PID Controller"
    OPT_FOPID = "Fractional Order PID (FOPID)"
    OPT_GA = "GA-Optimized Adaptive Fuzzy FOPID" # Name remains unchanged as requested

    # 2. Vault definition (This structure will remain UNCHANGED)
    # The parameters inside OPT_GA are essential for the Genetic Algorithm's search space.
    vault = {
        OPT_PID: mo.ui.dictionary({
            "Kp": mo.ui.number(0.0, 500.0, 0.000001, 10.0, label=r"Proportional ($K_p$)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.000001, 20.0, label=r"Integral ($K_i$)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.000001, 0.0, label=r"Derivative ($K_d$)", full_width=True),
        }),
        OPT_FOPID: mo.ui.dictionary({
            "Kp": mo.ui.number(0.0, 500.0, 0.000001, 5.0, label=r"Proportional ($K_p$)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.000001, 10.0, label=r"Integral ($K_i$)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.000001, 0.01, label=r"Derivative ($K_d$)", full_width=True),
            "Lambda": mo.ui.number(0.01, 2.0, 0.000001, 0.9, label=r"Int. Order ($\lambda$)", full_width=True),
            "Mu": mo.ui.number(0.01, 2.0, 0.000001, 0.8, label=r"Diff. Order ($\mu$)", full_width=True),
        }),
        OPT_GA: mo.ui.dictionary({
            # These sliders now primarily define the GA's search space bounds, not direct simulation values.
            "Kp": mo.ui.number(0.0, 500.0, 0.000001, 5.0, label=r"GA Kp (Search Space)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.000001, 10.0, label=r"GA Ki (Search Space)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.000001, 0.01, label=r"GA Kd (Search Space)", full_width=True),
            "Lambda": mo.ui.number(0.01, 2.0, 0.000001, 0.978, label=r"GA Lambda (Search Space)", full_width=True),
            "Mu": mo.ui.number(0.01, 2.0, 0.000001, 0.862, label=r"GA Mu (Search Space)", full_width=True),
            "Fuzzy_Scale": mo.ui.number(0.1, 10.0, 0.000001, 1.0, label=r"GA Fuzzy Scale (Search Space)", full_width=True),
        })
    }

    # --- NEW ADDITION ---
    # 3. Define the parameters for the Switch logic and the steady-state FOPID controller.
    # These will be displayed in the sidebar ONLY when the OPT_GA mode is active.
    switch_and_fopid_params = mo.ui.dictionary({
        "Kp_fixed": mo.ui.number(10.0, 500.0, 0.1, 50.0, label=r"Fixed Kp (Steady-State)", full_width=True),
        "Ki_fixed": mo.ui.number(10.0, 500.0, 0.1, 100.0, label=r"Fixed Ki (Steady-State)", full_width=True),
        # The Kd can be shared, so we don't need a separate fixed Kd unless specified.
        "Threshold": mo.ui.number(0.1, 10.0, 0.1, 0.5, label=r"Switching Threshold |E|", full_width=True)
    })
    return OPT_FOPID, OPT_GA, OPT_PID, switch_and_fopid_params, vault


@app.cell
def _(OPT_FOPID, OPT_GA, OPT_PID, mo):
    controller_selector = mo.ui.dropdown(
        options=[OPT_PID, OPT_FOPID, OPT_GA],
        value=OPT_PID,
        label="Architecture",
        full_width=True
    )
    return (controller_selector,)


@app.cell
def _(OPT_GA, controller_selector, mo, switch_and_fopid_params, vault):
    # The cell starts here
    # 1. Retrieve values from the vault based on selection
    active_params = vault[controller_selector.value]

    # 2. Other settings (Simulation & Motor)
    sim_settings = mo.ui.dictionary({
        "Ref": mo.ui.number(0, 1000, 1, 250, label=r"Ref Speed $\omega^*$ (rad/s)", full_width=True),
        "Load": mo.ui.number(0, 500, 0.1, 50, label="Load Torque $T_L$ (N.m)", full_width=True),
        "T_load": mo.ui.number(0, 5, 0.01, 0.1, label="Step Load Time (sec)", full_width=True),
        "Time": mo.ui.number(0.1, 5.0, 0.01, 0.2, label="Total Time (sec)", full_width=True)
    })

    motor_ui = mo.ui.dictionary({
        "Rs": mo.ui.number(0, 100, 0.01, 2.85, label="Stator Res $R_s$ ($\Omega$)", full_width=True),
        "Ld": mo.ui.number(0, 1, 0.0001, 0.004, label="Inductance $L_d$ (H)", full_width=True),
        "Lq": mo.ui.number(0, 1, 0.0001, 0.004, label="Inductance $L_q$ (H)", full_width=True),
        "Psi": mo.ui.number(0, 10, 0.0001, 0.1252, label="Flux $\psi_m$ (Wb)", full_width=True),
        "P": mo.ui.number(1, 20, 1, 4, label="Pole Pairs $P$", full_width=True),
        "J": mo.ui.number(0, 1, 0.0001, 0.008, label="Inertia $J$ ($kg.m^2$)", full_width=True),
        "B": mo.ui.number(0, 1, 0.0001, 0.001, label="Friction $B$", full_width=True)
    })

    vehicle_ui = mo.ui.dictionary({
        "Mass": mo.ui.number(1, 5000, 1, 600.0, label="Mass $M$ (kg)", full_width=True),
        "Rw": mo.ui.number(0.1, 2.0, 0.01, 0.3, label="Wheel Radius $r_w$ (m)", full_width=True),
        "Gear": mo.ui.number(1, 50, 0.1, 10.0, label="Gear Ratio $G$", full_width=True),
        "Cd": mo.ui.number(0, 2, 0.001, 0.335, label="Drag $C_d$", full_width=True),
        "Area": mo.ui.number(0, 10, 0.01, 2.0, label="Front Area $A$ ($m^2$)", full_width=True),
        "Cr": mo.ui.number(0, 1, 0.001, 0.02, label="Rolling Res $C_r$", full_width=True),
        "Rho": mo.ui.number(0, 2, 0.01, 1.2, label="Air Density $\rho$", full_width=True)
    })


    # --- START OF MODIFICATIONS ---
    # 3. Create a conditional stack to show/hide the switch parameters
    # This will hold the parameters for the switch and fixed FOPID
    conditional_switch_params = mo.vstack([
        mo.md("### ⚙️ Switch & FOPID Tuning"),
        switch_and_fopid_params,
    ]) if controller_selector.value == OPT_GA else mo.vstack([]) # Show if OPT_GA is selected, otherwise show nothing

    # 4. Display the Sidebar with the new conditional element
    mo.sidebar(
        mo.vstack([
            mo.md("# ⚙️ Control Station"),
            mo.md("---"),
            mo.md("### 🧠 Strategy"),
            controller_selector,
            mo.md("### 🎛️ Tuning"),
            active_params,
            conditional_switch_params, # <-- The new conditional UI element is added here
            mo.md("### 📉 Scenario"),
            sim_settings,
            mo.md("---"),
            mo.accordion({
                "🔌 PMSM Parameters": motor_ui,
                "🚗 Vehicle Dynamics": vehicle_ui
            })
        ])
    )

    # The cell ends here
    return active_params, motor_ui, sim_settings, vehicle_ui


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ##  Interactive Simulation & Performance Analysis Dashboard

        This is the main control station for the PMSM drive system. Use the **sidebar (⚙️)** to configure the controller architecture, tuning parameters, and simulation scenario. The results will be updated in real-time below, providing a comprehensive analysis of the system's dynamic performance and stability.
    """)
    return


@app.cell(hide_code=True)
def _(go, make_subplots, mo, np):
    # Assume 'np', 'go', 'make_subplots', 'mo', and 'OustaloupFilter' are pre-defined.

    # The fractional order to be tested (e.g., 0.5 represents a half-derivative).
    test_order = 0.5
    filter_viz = OustaloupFilter(test_order, freq_low=0.01, freq_high=100.0)

    # ------------------------------------------
    # Part A: Calculate Frequency Response (for Bode Plot)
    # ------------------------------------------
    # This part recalculates the continuous-time transfer function to plot the ideal response.
    freqs = np.logspace(-3, 3, 500)
    magnitudes = []
    phases = []

    for w in freqs:
        s = 1j * w
        h_s = filter_viz.gain
        wb, wh, N, alpha = 0.01, 100.0, 3, test_order # Oustaloup params
        for k in range(-N, N + 1):
            w_z = wb * (wh/wb)**((k + N + 0.5*(1 - alpha))/(2*N + 1))
            w_p = wb * (wh/wb)**((k + N + 0.5*(1 + alpha))/(2*N + 1))
            h_s *= (s + w_z) / (s + w_p)

        magnitudes.append(20 * np.log10(np.abs(h_s)))
        phases.append(np.angle(h_s, deg=True))

    # ------------------------------------------
    # Part B: Calculate Time Domain Response
    # ------------------------------------------
    # This part uses the discrete-time filter's 'compute' method.
    time_sim = OustaloupFilter(test_order)
    t_vec = np.linspace(0, 10, 1000)
    input_step = np.ones_like(t_vec)
    output_response = [time_sim.compute(val) for val in input_step]

    # ------------------------------------------
    # Plotting Configuration & Theming
    # ------------------------------------------
    c_mag = '#00d2ff'   # Bright Cyan for Magnitude
    c_phase = '#e74c3c' # Bright Red for Phase
    c_resp = '#54a0ff'  # Bright Blue for Time Response
    c_text = '#aaa'     # Light gray for text, visible on light/dark backgrounds
    c_grid = 'rgba(170, 170, 170, 0.2)' # Subtle grid lines

    # --- Key Subplot Configuration ---
    # To create a chart with two y-axes (for magnitude and phase), we must use 'specs'
    # to explicitly enable a secondary y-axis on the first subplot.
    fig_filter = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"<b style='color:{c_mag}'>Bode Plot (Frequency Domain)</b>",
            f"<b style='color:{c_resp}'>Step Response (Time Domain)</b>"
        ),
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {}]] # <-- This enables the dual-axis plot
    )

    # Plot 1a: Add the Magnitude trace to the primary y-axis.
    fig_filter.add_trace(go.Scatter(
        x=freqs, y=magnitudes,
        name="Magnitude (dB)",
        line=dict(color=c_mag, width=2.5)
    ), row=1, col=1, secondary_y=False)

    # Plot 1b: Add the Phase trace to the secondary y-axis.
    fig_filter.add_trace(go.Scatter(
        x=freqs, y=phases,
        name="Phase (Deg)",
        line=dict(color=c_phase, width=2.5)
    ), row=1, col=1, secondary_y=True) # <-- This assigns the trace to the second axis

    # Plot 2a: Add the filter's time response to the second subplot.
    fig_filter.add_trace(go.Scatter(
        x=t_vec, y=output_response,
        name=f"Response of s<sup>{test_order}</sup>",
        line=dict(color=c_resp, width=2.5)
    ), row=1, col=2)

    # Plot 2b: Add the input step signal for reference.
    fig_filter.add_trace(go.Scatter(
        x=t_vec, y=input_step,
        name="Input Step",
        line=dict(color=c_text, dash='dash', width=1.5),
        opacity=0.7
    ), row=1, col=2)

    # ------------------------------------------
    # General Styling for Dark/Light Mode
    # ------------------------------------------
    base_axis_style = dict(
        showgrid=True, gridcolor=c_grid, zerolinecolor=c_grid, tickfont=dict(color=c_text)
    )

    fig_filter.update_layout(
        title=dict(
            text=f"<b>Fractional Filter Analysis (Oustaloup Method)</b><br><span style='font-size:12px; color:{c_text};'>Verifying that s<sup>{test_order}</sup> behaves as a half-derivative</span>",
            y=0.92, x=0.05, xanchor='left', yanchor='top'
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        font=dict(color=c_text),
        hovermode="x unified",
        margin=dict(t=100, l=80, r=80, b=50),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),

        # Bode Plot Axes
        xaxis=dict(type="log", title="Frequency (rad/s)", **base_axis_style),
        yaxis=dict(title="Magnitude (dB)", title_font=dict(color=c_mag), **base_axis_style),
        yaxis2=dict(
            title="Phase (Deg)", title_font=dict(color=c_phase),
            range=[0, 90], showgrid=False, tickfont=dict(color=c_phase)
        ),

        # Step Response Axes
        xaxis2=dict(title="Time (s)", **base_axis_style),
        yaxis3=dict(title="Amplitude", **base_axis_style)
    )

    # Add a horizontal line showing the theoretical target phase for this order.
    theoretical_phase = test_order * 90
    fig_filter.add_hline(
        y=theoretical_phase, line_dash="dot", line_color=c_phase,
        annotation_text=f"Target: {theoretical_phase}°",
        annotation_font=dict(color=c_phase),
        annotation_position="bottom right",
        row=1, col=1, secondary_y=True
    )

    # --- Render Final Output in Marimo ---
    mo.vstack([
        mo.md("### 📉 Oustaloup Filter Verification"),
        mo.md(f"""
        This analysis validates the behavior of the fractional-order operator, **s<sup>{test_order}</sup>**. A key property of a fractional operator is its constant phase shift across a wide frequency range.

        - The **Bode Plot** confirms this: the phase (red line) holds steady around the theoretical target of **{theoretical_phase}°**.
        - The **Step Response** visualizes how this unique operator behaves in the time domain when subjected to a sudden input.
        """),
        mo.ui.plotly(fig_filter)
    ])
    return


@app.class_definition
class OustaloupFilter:
    """
    Implements a fractional-order calculus filter using Oustaloup's Recursive Approximation.
    This method approximates the fractional operator s^alpha as a cascade of first-order
    zero-pole filters, making it suitable for digital implementation.
    """
    def __init__(self, alpha, num_poles=3, freq_low=0.01, freq_high=1000.0, dt=0.0001):
        """
        Initializes the fractional-order filter.

        Args:
            alpha (float): The fractional order. For a FOPID controller, this would be
                           lambda for the integral part or mu for the derivative part.
            num_poles (int): The number of poles and zeros used in the approximation (N).
                             Higher numbers increase accuracy but also computational cost.
            freq_low (float): The lower frequency bound (wb) of the approximation range in Hz.
            freq_high (float): The upper frequency bound (wh) of the approximation range in Hz.
            dt (float): The sampling time (time step) for discretization.
        """
        self.alpha = alpha
        self.dt = dt
        # The filter is only active if the fractional order is non-zero.
        self.active = (abs(alpha) > 1e-6)

        # State storage for each sub-filter in the cascade.
        # The main filter is broken down into a series of smaller, stable filters.
        self.filters_state_x = [] # Stores previous input values (x[n-1]) for each filter.
        self.filters_state_y = [] # Stores previous output values (y[n-1]) for each filter.
        self.coeffs = []          # Stores the computed coefficients (b0, b1, a1) for each filter.
        self.gain = 1.0           # The overall filter gain.

        if self.active:
            self._compute_coefficients(alpha, num_poles, freq_low, freq_high)
            # Initialize the state storage with zeros.
            num_filters = len(self.coeffs)
            self.filters_state_x = [0.0] * num_filters
            self.filters_state_y = [0.0] * num_filters

    def _compute_coefficients(self, alpha, N, wb, wh):
        # This method translates the continuous-time Oustaloup approximation
        # into discrete-time coefficients for real-time processing.

        # 1. Calculate continuous-time poles and zeros based on Oustaloup's equations.
        zeros = []
        poles = []

        for k in range(-N, N + 1):
            # Calculate the frequency for the k-th zero.
            w_z = wb * (wh / wb)**((k + N + 0.5 * (1 - alpha)) / (2 * N + 1))
            # Calculate the frequency for the k-th pole.
            w_p = wb * (wh / wb)**((k + N + 0.5 * (1 + alpha)) / (2 * N + 1))
            zeros.append(w_z)
            poles.append(w_p)

        # The overall gain of the continuous-time filter.
        self.gain = (wh)**alpha

        # 2. Discretize each zero-pole pair using the Tustin (Bilinear) Transform.
        # This converts each H_k(s) = (s + w_z) / (s + w_p) into a discrete-time
        # difference equation: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1].
        # The Tustin transform substitutes s with (2/dt) * (1-z^-1)/(1+z^-1).

        for w_z, w_p in zip(zeros, poles):
            # Tustin transformation constant.
            k_tustin = (2.0 / self.dt)

            # Coefficients for the denominator: (s + w_p) -> a0 + a1*z^-1
            a0 = k_tustin + w_p
            a1 = w_p - k_tustin

            # Coefficients for the numerator: (s + w_z) -> b0 + b1*z^-1
            b0 = k_tustin + w_z
            b1 = w_z - k_tustin

            # Normalize the coefficients by a0 to isolate y[n] in the difference equation.
            self.coeffs.append({
                'b0': b0 / a0,
                'b1': b1 / a0,
                'a1': a1 / a0
            })

    def compute(self, input_val):
        """
        Processes a single input value through the fractional-order filter.

        Args:
            input_val (float): The current input signal value x[n].

        Returns:
            float: The filtered output signal value y[n].
        """
        if not self.active:
            return input_val

        # Start with the input signal scaled by the filter's overall gain.
        current_signal = input_val * self.gain

        # Pass the signal through the cascade of first-order filters.
        for i, coeff in enumerate(self.coeffs):
            # Retrieve the previous input and output states for this specific filter.
            x_prev = self.filters_state_x[i]
            y_prev = self.filters_state_y[i]

            # Apply the difference equation for this filter:
            # y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
            output = (coeff['b0'] * current_signal) + (coeff['b1'] * x_prev) - (coeff['a1'] * y_prev)

            # Update the state memory for the next time step.
            self.filters_state_x[i] = current_signal
            self.filters_state_y[i] = output

            # The output of this filter becomes the input for the next one in the cascade.
            current_signal = output

        return current_signal


@app.cell
def _(FuzzyScalerController, OPT_FOPID, OPT_GA, RealFuzzyController, np):
    # The cell starts here

    def simulate_pmsm_system(ctrl_params, scenario, motor_phys, vehicle_phys, switch_and_fopid_params, strategy_type):
        """
        Simulates the PMSM-based EV powertrain.
        Handles all strategies: PID, FOPID, and the switched AFFOPID-FOPID method.
        Logs all relevant data for comprehensive analysis.
        """
        # 1. Unpack Simulation, Physics, and Controller Parameters
        t_end = scenario["Time"]
        w_target = scenario["Ref"]
        t_load = scenario["T_load"]
        load_val = scenario["Load"]

        Rs, P, Psi_m, J, B = motor_phys["Rs"], motor_phys["P"], motor_phys["Psi"], motor_phys["J"], motor_phys["B"]
        Mass, Rw, Gear, Cd, Area, Rho = vehicle_phys["Mass"], vehicle_phys["Rw"], vehicle_phys["Gear"], vehicle_phys["Cd"], vehicle_phys["Area"], vehicle_phys["Rho"]

        # Unpack parameters for the switch and the fixed FOPID mode
        kp_fixed = switch_and_fopid_params["Kp_fixed"]
        ki_fixed = switch_and_fopid_params["Ki_fixed"]
        error_threshold = switch_and_fopid_params["Threshold"]

        # Unpack general controller parameters
        kd = ctrl_params.get("Kd", 0.0)
        lam = ctrl_params.get("Lambda", 1.0)
        mu = ctrl_params.get("Mu", 1.0)
        fuzzy_master_scale = ctrl_params.get("Fuzzy_Scale", 1.0)

        # 2. Initialize Controller Components
        is_fuzzy_mode = (strategy_type == OPT_GA) 

        if is_fuzzy_mode:
            flc1_tuner = RealFuzzyController()
            flc2_scaler = FuzzyScalerController()
    
        use_fractional = (strategy_type == OPT_FOPID) or (strategy_type == OPT_GA)
        frac_integrator = OustaloupFilter(-lam) if use_fractional else None
        frac_differentiator = OustaloupFilter(mu) if use_fractional else None

        # 3. Initialize Simulation State and Logging Variables
        dt = 0.0001
        steps = int(t_end / dt)
        time = np.linspace(0, t_end, steps)
    
        speed_arr, torque_arr, iq_arr = np.zeros(steps), np.zeros(steps), np.zeros(steps)
        kp_log, ki_log, sf_log = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    
        w_mech, iq_current, error_sum, prev_error = 0.0, 0.0, 0.0, 0.0

        # 4. Main Simulation Loop
        for i in range(steps):
            t = time[i]
            ref = w_target if t > 0.005 else 0.0
            error = ref - w_mech
            delta_error = (error - prev_error) / dt if dt > 0 else 0.0

            # --- Block A: SWITCH LOGIC & GAIN DETERMINATION ---
            final_kp = 0.0
            final_ki = 0.0
            final_kd = 0.0 # Default initialization
            current_sf = 1.0

            if is_fuzzy_mode and abs(error) > error_threshold:
                # --- TRANSIENT STATE (High Error) ---
                # Use Intelligent Control (AFFOPID)
                base_kp, base_ki = flc1_tuner.compute_base_gains(error, delta_error)
                raw_scaling_factor = flc2_scaler.compute_scaling_factor(error, delta_error)
                current_sf = raw_scaling_factor * fuzzy_master_scale
            
                final_kp = base_kp * current_sf
                final_ki = base_ki * current_sf
                final_kd = kd # Use the optimized Kd during transient
        
            else:
                # --- STEADY STATE (Low Error) ---
                if strategy_type == OPT_GA:
                    # *** CRITICAL FIX HERE ***
                    # Use Fixed Gains and DISABLE Derivative term to stop oscillations
                    final_kp = kp_fixed
                    final_ki = ki_fixed
                    final_kd = 0.0 # Force Kd to 0 in steady state for smooth response
                else:
                    # Standard PID/FOPID behavior
                    final_kp = ctrl_params["Kp"]
                    final_ki = ctrl_params["Ki"]
                    final_kd = ctrl_params["Kd"]
        
            # --- Logging ---
            kp_log[i] = final_kp
            ki_log[i] = final_ki
            sf_log[i] = current_sf

            # --- Block B: Core Controller Logic ---
            p_term = final_kp * error

            if use_fractional:
                i_signal = frac_integrator.compute(error)
                d_signal = frac_differentiator.compute(error)
                i_term = final_ki * i_signal
                d_term = final_kd * d_signal
            else: 
                error_sum += error * dt
                i_term = final_ki * error_sum
                d_term = final_kd * delta_error

            iq_ref = p_term + i_term + d_term

            # Saturation and Anti-windup
            if iq_ref > 200:
                iq_ref = 200
                if not use_fractional: error_sum -= error * dt
            elif iq_ref < -200:
                iq_ref = -200
                if not use_fractional: error_sum -= error * dt

            # --- Block C: Plant Model ---
            iq_current += (iq_ref - iq_current) * (dt / 0.001)
            Te = 1.5 * P * Psi_m * iq_current
            T_ext = load_val if t >= t_load else 0.0
            v = (w_mech / Gear) * Rw
            F_drag = 0.5 * Rho * Cd * Area * v * abs(v)
            T_drag = (F_drag * Rw) / Gear
            dw_dt = (Te - T_ext - T_drag - B * w_mech) / J
            w_mech += dw_dt * dt
            if w_mech < 0: w_mech = 0

            # Log state variables
            speed_arr[i] = w_mech
            torque_arr[i] = Te
            iq_arr[i] = iq_current
            prev_error = error

        return time, speed_arr, torque_arr, iq_arr, kp_log, ki_log, sf_log, ref
    return (simulate_pmsm_system,)


@app.cell(hide_code=True)
def _(
    active_params,
    controller_selector,
    go,
    make_subplots,
    mo,
    motor_ui,
    sim_settings,
    simulate_pmsm_system,
    switch_and_fopid_params,
    vehicle_ui,
):
    # --- Cell 5: Main Dashboard (Fixed Layout) ---

    # 1. Run Simulation (Global Data)
    # --- START OF MODIFICATION ---
    # The function now returns more logs. We need to unpack them into new variables.
    t_data, w_data, te_data, iq_data, kp_log, ki_log, sf_log, ref_data = simulate_pmsm_system(
        active_params.value,
        sim_settings.value,
        motor_ui.value,
        vehicle_ui.value,
        switch_and_fopid_params.value,
        controller_selector.value
    )
    # --- END OF MODIFICATION ---


    # 2. Define Plotting Function (Encapsulated & Clean) - NO CHANGES HERE
    def generate_dashboard_plot(t_arr, w_arr, te_arr, iq_arr, ref_val, load_config):
        """
        Plotting function with fixed overlapping titles
        """
        # Create Local Figure
        _fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1, # Increased spacing between subplots
            subplot_titles=("🚀 Speed Response", "⚙️ Torque (Mechanical)", "⚡ Stator Current (Iq)")
        )

        # A. Speed Trace
        _fig.add_trace(go.Scatter(x=t_arr, y=w_arr, name="Speed", line=dict(color="#00d2ff", width=3)), row=1, col=1)
        # Handle Reference
        # --- Minor fix: ref_data can be a single value, not an array ---
        ref_val_corrected = ref_val if isinstance(ref_val, (int, float)) else ref_val[0]
        _ref_line = [ref_val_corrected]*len(t_arr)
        _fig.add_trace(go.Scatter(x=t_arr, y=_ref_line, name="Target", line=dict(dash="dash", color="#ff4b1f")), row=1, col=1)

        # B. Torque Trace
        _fig.add_trace(go.Scatter(x=t_arr, y=te_arr, name="Torque", line=dict(color="#00b09b", width=2)), row=2, col=1)

        # C. Current Trace
        _fig.add_trace(go.Scatter(x=t_arr, y=iq_arr, name="Current", line=dict(color="#f7971e", width=2)), row=3, col=1)

        # D. Load Marker Logic (Fixed Position)
        _load_val = load_config["Load"]
        _load_time = load_config["T_load"]

        if _load_val > 0.01:
            # Draw line
            _fig.add_vline(x=_load_time, line_width=2, line_dash="dot", line_color="#f1c40f", opacity=0.8)
            # Annotation (adjusted position)
            _fig.add_annotation(
                x=_load_time, y=1.12, yref="paper", text=" ", showarrow=False,
                font=dict(size=10, color="#f1c40f"), xanchor="left", xshift=5
            )

        # E. Styling & Margins
        _fig.update_layout(
            height=750, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#555"), showlegend=False, margin=dict(t=80, b=30, l=50, r=20)
        )

        _grid_style = dict(showgrid=True, gridcolor='rgba(64, 160, 255, 0.1)')
        _fig.update_xaxes(**_grid_style, row=1, col=1); _fig.update_yaxes(**_grid_style, row=1, col=1)
        _fig.update_xaxes(**_grid_style, row=2, col=1); _fig.update_yaxes(**_grid_style, row=2, col=1)
        _fig.update_xaxes(**_grid_style, row=3, col=1); _fig.update_yaxes(**_grid_style, row=3, col=1)

        return _fig

    # 3. Call Function & Display - NO CHANGES HERE
    final_dashboard_fig = generate_dashboard_plot(t_data, w_data, te_data, iq_data, ref_data, sim_settings.value)

    # 4. Render - NO CHANGES HERE
    mo.vstack([
        mo.md("### 🏎️ Vehicle Performance"),
        mo.ui.plotly(final_dashboard_fig)
    ])
    return iq_data, ki_log, kp_log, ref_data, t_data, te_data, w_data


@app.cell(hide_code=True)
def _(mo, np, ref_data, t_data, w_data):

    def calculate_metrics(t, y, ref_val):
        # Convert inputs to numpy arrays
        y = np.array(y)
        t = np.array(t)

        # Handle reference value type
        target = ref_val if isinstance(ref_val, (int, float)) else ref_val
        if target == 0: return 0.0, 0.0, 0.0, 0.0

        # 1. Max Overshoot (%)
        max_val = np.max(y)
        overshoot = ((max_val - target) / target) * 100

        # 2. Rise Time (10% to 90%)
        lower_bound = 0.1 * target
        upper_bound = 0.9 * target
        t_10_idx = np.where(y >= lower_bound)[0]
        t_90_idx = np.where(y >= upper_bound)[0]

        if len(t_10_idx) > 0 and len(t_90_idx) > 0:
            rise_time = t[t_90_idx[0]] - t[t_10_idx[0]]
        else:
            rise_time = 0.0

        # 3. Settling Time (2% tolerance band)
        threshold = 0.02 * target
        error = np.abs(y - target)
        unstable_indices = np.where(error > threshold)[0]

        if len(unstable_indices) == 0:
            settling_time = 0.0
        else:
            settling_time = t[unstable_indices[-1]]

        # 4. Steady State Error
        # Average of last 50 points to filter noise
        final_val = np.mean(y[-50:])
        ess = np.abs(target - final_val)

        return overshoot, rise_time, settling_time, ess

    # Calculate values
    os_val, tr_val, ts_val, ess_val = calculate_metrics(t_data, w_data, ref_data)

    # Define CSS Styles (Neutral colors safe for all modes)
    # Text Color: #2c3e50 (Dark Slate Blue) - Visible on white and dark backgrounds
    # Value Color: #16a085 (Green Sea) for good metrics, #c0392b (Red) for errors
    # Border Color: #bdc3c7 (Silver)

    card_style = """
        flex: 1;
        text-align: center;
        padding: 10px;
        margin: 5px;
        border-right: 1px solid #bdc3c7;
    """

    label_style = """
        font-size: 0.9rem;
        color: #566573; 
        font-weight: 600;
        margin-bottom: 5px;
        font-family: 'Segoe UI', sans-serif;
    """

    value_style = """
        font-size: 1.4rem;
        font-weight: bold;
        color: #1a5276;
        font-family: 'Consolas', monospace;
    """

    # Render HTML Dashboard
    mo.Html(f"""
    <div style="
        display: flex; 
        flex-direction: row; 
        justify-content: space-around; 
        background-color: transparent; 
        width: 100%; 
        padding: 15px 0;">

        <!-- Overshoot -->
        <div style="{card_style}">
            <div style="{label_style}">Max Overshoot (M<sub>p</sub>)</div>
            <div style="{value_style}">{os_val:.4f}%</div>
        </div>

        <!-- Rise Time -->
        <div style="{card_style}">
            <div style="{label_style}">Rise Time (T<sub>r</sub>)</div>
            <div style="{value_style}">{tr_val:.4f} s</div>
        </div>

        <!-- Settling Time -->
        <div style="{card_style}">
            <div style="{label_style}">Settling Time (T<sub>s</sub>)</div>
            <div style="{value_style}">{ts_val:.4f} s</div>
        </div>

        <!-- Steady State Error -->
        <div style="{card_style} border-right: none;">
            <div style="{label_style}">Steady State Error (e<sub>ss</sub>)</div>
            <div style="{value_style} color: #c0392b;">{ess_val:.4f} rad/s</div>
        </div>

    </div>
    """)
    return


@app.cell
def _(iq_data, mo, motor_ui, np, t_data, te_data, vehicle_ui, w_data):
    # This is the final output variable for the cell.
    output_energy = None 

    # --- Case 1: No simulation data is available ---
    if 't_data' not in globals() or t_data is None or len(t_data) == 0:
        output_energy = mo.callout(
            mo.md("⏳ **Waiting for simulation data...** Run a simulation from the main dashboard to calculate advanced metrics."),
            kind="neutral"
        )
    # --- Case 2: Simulation data exists ---
    else:
        # Helper function to calculate the advanced metrics.
        def calculate_advanced_metrics(time, w_mech, torque, iq, motor_phys, vehicle_phys):
            dt = time[1] - time[0]

            # --- 1. Energy Consumption ---
            Rs = motor_phys["Rs"]
            P_mech = torque * w_mech
            P_loss = 1.5 * Rs * (iq ** 2)
            P_elec_in = P_mech + P_loss
            total_energy_joules = np.sum(P_elec_in[P_elec_in > 0]) * dt
            total_energy_Wh = total_energy_joules / 3600

            # --- 2. Control Effort ---
            control_effort = np.sum(np.abs(np.diff(iq, prepend=iq[0])))

            # --- 3. Ride Comfort ---
            # Calculation for vehicle speed is still needed for Jerk.
            Rw, Gear = vehicle_phys["Rw"], vehicle_phys["Gear"]
            vehicle_speed_mps = (w_mech / Gear) * Rw
            accel = np.gradient(vehicle_speed_mps, dt)
            jerk = np.gradient(accel, dt)
            integral_jerk_squared = np.sum(jerk ** 2) * dt

            return {
                "Total Energy (Wh)": total_energy_Wh,
                "Control Effort": control_effort,
                "Ride Comfort (Jerk)": integral_jerk_squared
            }

        # Run the calculation.
        metrics = calculate_advanced_metrics(t_data, w_data, te_data, iq_data, motor_ui.value, vehicle_ui.value)

        # HTML and CSS for the display cards.
        html_content = f"""
        <style>
            .adv-kpi-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
            .adv-kpi-card {{ background-color: rgba(128, 128, 128, 0.05); border-radius: 8px; padding: 20px; border-left: 5px solid; }}
            .adv-kpi-title {{ display: flex; align-items: center; font-size: 0.9rem; font-weight: 600; color: #8899a6; margin-bottom: 8px; }}
            .adv-kpi-title span {{ margin-right: 8px; font-size: 1.2rem; }}
            .adv-kpi-value {{ font-size: 2.0rem; font-weight: 700; color: #2c3e50; font-family: 'Consolas', 'Menlo', monospace; }}
            .adv-kpi-desc {{ font-size: 0.8rem; color: #aaa; margin-top: 4px; }}
            /* Specific border colors for each card */
            .energy-card {{ border-color: #f39c12; }} /* Orange */
            .effort-card {{ border-color: #2980b9; }} /* Blue */
            .comfort-card {{ border-color: #8e44ad; }} /* Purple */
            @media (prefers-color-scheme: dark) {{ .adv-kpi-value {{ color: #ecf0f1; }} .adv-kpi-card {{ background-color: rgba(255, 255, 255, 0.07); }} }}
        </style>
        <div class="adv-kpi-container">
            <!-- Card 1: Total Energy Consumed -->
            <div class="adv-kpi-card energy-card">
                <div class="adv-kpi-title"><span>⚡️</span> TOTAL ENERGY</div>
                <div class="adv-kpi-value">{metrics['Total Energy (Wh)']:.4f}</div>
                <div class="adv-kpi-desc">Watt-hours (Wh)</div>
            </div>

            <!-- Card 2: Control Effort -->
            <div class="adv-kpi-card effort-card">
                <div class="adv-kpi-title"><span>🕹️</span> CONTROL EFFORT</div>
                <div class="adv-kpi-value">{metrics['Control Effort']:.2f}</div>
                <div class="adv-kpi-desc">Lower is smoother</div>
            </div>

            <!-- Card 3: Ride Comfort -->
            <div class="adv-kpi-card comfort-card">
                <div class="adv-kpi-title"><span>🛋️</span> RIDE COMFORT</div>
                <div class="adv-kpi-value">{metrics['Ride Comfort (Jerk)']:.2f}</div>
                <div class="adv-kpi-desc">Lower is less jerky</div>
            </div>
        </div>
        """

        # Assemble the final Marimo object to be displayed.
        output_energy = mo.vstack([
            mo.md("### 📈 Advanced Performance & Efficiency Analysis"),
            mo.md("Beyond tracking error, these metrics evaluate the controller's real-world viability by measuring its energy consumption, control smoothness, and impact on ride comfort."),
            mo.Html(html_content)
        ])

    # This final line displays the correct output for the cell.
    output_energy
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧩 Fuzzy Logic: Visualized

    Understanding the non-linear behavior of a fuzzy controller is best done visually.
    This panel breaks down the controller's decision-making process into its fundamental visual components:

    - **Membership Functions:** How the controller categorizes inputs.
    - **Control Surface:** The complete map of all possible decisions.
    - **Rule Matrix:** The core logic table that drives the system.
    """)
    return


@app.cell
def _(np):
    # The cell starts here


    # --- Controller 1: The Gain Generator (FLC1) ---
    class RealFuzzyController:
        def __init__(self):
            self.terms = ['NB', 'NS', 'Z', 'PS', 'PB']
            self.centers = {'NB': -1.0, 'NS': -0.5, 'Z': 0.0, 'PS': 0.5, 'PB': 1.0}

            # --- Outputs and Rule Bases for Kp and Ki ---
            # These are example values and will likely need tuning.
            self.kp_outputs = {'S': 10,  'M': 40,  'B': 80,  'VB': 120}
            self.ki_outputs = {'S': 20,  'M': 80,  'B': 150, 'VB': 250}

            # Rule base for Kp: Determines the proportional response.
            self.kp_rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['VB', 'B',  'B',  'M',  'S'],  # e: NB
                ['B',  'B',  'M',  'S',  'S'],  # e: NS
                ['B',  'M',  'S',  'M',  'B'],  # e: Z
                ['S',  'S',  'M',  'B',  'B'],  # e: PS
                ['S',  'M',  'B',  'B',  'VB']  # e: PB
            ]
            # Rule base for Ki: Determines the integral response to eliminate steady-state error.
            self.ki_rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['VB', 'VB', 'B',  'M',  'S'],  # e: NB
                ['VB', 'B',  'M',  'S',  'M'],  # e: NS
                ['B',  'M',  'S',  'M',  'B'],  # e: Z
                ['M',  'S',  'M',  'B',  'VB'],  # e: PS
                ['S',  'M',  'B',  'VB', 'VB']  # e: PB
            ]

        def triangle_mf(self, x, center, width=0.5):
            return max(0, 1 - abs(x - center) / width)

        def compute_base_gains(self, error, delta_error):
            e_norm = np.clip(error / 250.0 * 5.0, -1, 1)
            de_norm = np.clip(delta_error / 10.0, -1, 1)

            kp_numerator = 0.0
            ki_numerator = 0.0
            denominator = 0.0

            for i, e_term in enumerate(self.terms):
                mu_e = self.triangle_mf(e_norm, self.centers[e_term])
                if mu_e == 0: continue

                for j, de_term in enumerate(self.terms):
                    mu_de = self.triangle_mf(de_norm, self.centers[de_term])
                    if mu_de == 0: continue

                    firing_strength = min(mu_e, mu_de)
                    if firing_strength == 0: continue

                    # Calculate numerator for Kp
                    kp_output_term = self.kp_rule_matrix[i][j]
                    kp_output_val = self.kp_outputs[kp_output_term]
                    kp_numerator += firing_strength * kp_output_val

                    # Calculate numerator for Ki
                    ki_output_term = self.ki_rule_matrix[i][j]
                    ki_output_val = self.ki_outputs[ki_output_term]
                    ki_numerator += firing_strength * ki_output_val

                    denominator += firing_strength

            if denominator == 0:
                return (self.kp_outputs['M'], self.ki_outputs['M']) # Return medium default values

            final_kp = kp_numerator / denominator
            final_ki = ki_numerator / denominator

            return (final_kp, final_ki)

    # --- Controller 2: The Scaling Factor Generator (FLC2) ---
    class FuzzyScalerController:
        def __init__(self):
            self.terms = ['NB', 'NS', 'Z', 'PS', 'PB']
            self.centers = {'NB': -1.0, 'NS': -0.5, 'Z': 0.0, 'PS': 0.5, 'PB': 1.0}

            # Outputs are scaling factors, numbers around 1.0
            self.outputs = {'S': 0.9, 'M': 1.0, 'B': 1.1, 'VB': 1.2}

            # This rule base determines how to fine-tune the gains from FLC1.
            self.rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['VB', 'B',  'M',  'S',  'S'],  # e: NB
                ['B',  'B',  'M',  'S',  'S'],  # e: NS
                ['M',  'M',  'S',  'M',  'M'],  # e: Z
                ['S',  'S',  'M',  'B',  'B'],  # e: PS
                ['S',  'S',  'M',  'B',  'VB']  # e: PB
            ]

        def triangle_mf(self, x, center, width=0.5):
            return max(0, 1 - abs(x - center) / width)

        def compute_scaling_factor(self, error, delta_error):
            e_norm = np.clip(error / 250.0 * 5.0, -1, 1)
            de_norm = np.clip(delta_error / 10.0, -1, 1)

            numerator = 0.0
            denominator = 0.0

            for i, e_term in enumerate(self.terms):
                mu_e = self.triangle_mf(e_norm, self.centers[e_term])
                if mu_e == 0: continue

                for j, de_term in enumerate(self.terms):
                    mu_de = self.triangle_mf(de_norm, self.centers[de_term])
                    if mu_de == 0: continue

                    firing_strength = min(mu_e, mu_de)
                    output_term = self.rule_matrix[i][j]
                    output_val = self.outputs[output_term]
                    numerator += firing_strength * output_val
                    denominator += firing_strength

            if denominator == 0:
                return 1.0 # Default scaling factor is 1.0 (no change)

            return numerator / denominator


    # The cell ends here
    return FuzzyScalerController, RealFuzzyController


@app.cell
def _(FuzzyScalerController, RealFuzzyController, mo):
    # The cell starts here

    # --- CONSTANTS FOR OPTIONS ---
    OPT_KP = "Kp Output (from FLC1)"
    OPT_KI = "Ki Output (from FLC1)"
    OPT_SF = "Scaling Factor Output (from FLC2)"

    # --- UI ELEMENT FOR SELECTION ---
    # This cell ONLY defines the UI element. It does not access its .value
    fuzzy_viz_selector = mo.ui.dropdown(
        options=[OPT_KP, OPT_KI, OPT_SF],
        value=OPT_KP,
        label="Select Control Surface to Visualize:"
    )

    # We also need the controller instances for the next cell
    flc1_viz_instance = RealFuzzyController()
    flc2_viz_instance = FuzzyScalerController()

    # The final output of this cell is just the UI element itself and the instances
    mo.vstack([
        mo.md("## 🧩 Fuzzy Controller Internals"),
        mo.md("""
        This dashboard visualizes the internal logic of the fuzzy controllers.
        1.  **Fuzzification:** The top plot shows the shared membership functions for inputs.
        2.  **Control Surface:** The bottom 3D plot is the decision-making map. Use the dropdown to switch between the different fuzzy outputs.
        """),
        fuzzy_viz_selector # Display the dropdown
    ])

    # The cell ends here
    return (
        OPT_KI,
        OPT_KP,
        flc1_viz_instance,
        flc2_viz_instance,
        fuzzy_viz_selector,
    )


@app.cell
def _(
    OPT_KI,
    OPT_KP,
    flc1_viz_instance,
    flc2_viz_instance,
    fuzzy_viz_selector,
    go,
    mo,
    np,
):
    # The cell starts here

    # --- DYNAMIC VISUALIZATION LOGIC ---
    # This cell DEPENDS on fuzzy_viz_selector from the previous cell.
    # Here, we are allowed to access .value
    selected_surface = fuzzy_viz_selector.value

    # =======================================================
    # PART 1: Fuzzification Visualization (Membership Functions)
    # =======================================================
    x_range = np.linspace(-1.5, 1.5, 300)
    fig_mf = go.Figure()
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    for idx, term in enumerate(flc1_viz_instance.terms):
        y_values = [flc1_viz_instance.triangle_mf(x, flc1_viz_instance.centers[term]) for x in x_range]
        fig_mf.add_trace(go.Scatter(
            x=x_range, y=y_values, name=f"Term: {term}", fill='tozeroy',
            line=dict(color=colors[idx], width=2.5), opacity=0.7
        ))
    fig_mf.update_layout(
        title=dict(text="<b>1. Fuzzification Stage (Shared Membership Functions)</b>", y=0.9, x=0.01, xanchor='left', yanchor='top'),
        xaxis_title="Normalized Input (e.g., Error)", yaxis_title="Degree of Membership (μ)", height=400,
        margin=dict(l=40, r=20, t=120, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#aaa"
    )

    # =================================================================
    # PART 2: Dynamic Control Surface Visualization
    # =================================================================
    res = 40
    e_vec = np.linspace(-250, 250, res)
    de_vec = np.linspace(-10, 10, res)
    z_surface = np.zeros((res, res))
    z_axis_title = ""

    for i in range(res):
        for j in range(res):
            if selected_surface == OPT_KP or selected_surface == OPT_KI:
                kp_val, ki_val = flc1_viz_instance.compute_base_gains(e_vec[i], de_vec[j])
                if selected_surface == OPT_KP:
                    z_surface[j][i] = kp_val
                    z_axis_title = "Base Kp Output"
                else: # OPT_KI
                    z_surface[j][i] = ki_val
                    z_axis_title = "Base Ki Output"
            else: # OPT_SF
                z_surface[j][i] = flc2_viz_instance.compute_scaling_factor(e_vec[i], de_vec[j])
                z_axis_title = "Scaling Factor (SF) Output"

    fig_surf = go.Figure(data=[go.Surface(
        z=z_surface, x=e_vec, y=de_vec, colorscale='Plasma',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="#00FFFF", project_z=True),
        opacity=0.9
    )])
    fig_surf.update_layout(
        title=dict(text=f"<b>2. Control Surface for: {z_axis_title}</b>", y=0.92, x=0.01, xanchor='left', yanchor='top'),
        scene=dict(
            xaxis_title='Error (E)', yaxis_title='Change of Error (dE)', zaxis_title=z_axis_title,
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))),
        height=600, margin=dict(l=10, r=10, t=100, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font_color="#aaa"
    )

    # ==========================================
    # PART 3: Render Final Plots in Marimo
    # ==========================================
    mo.vstack([
        mo.ui.plotly(fig_mf),
        mo.ui.plotly(fig_surf)
    ])

    # The cell ends here
    return


@app.cell
def _(
    OPT_KI,
    OPT_KP,
    flc1_viz_instance,
    flc2_viz_instance,
    fuzzy_viz_selector,
    go,
    mo,
):
    # The cell starts here

    # This cell DEPENDS on fuzzy_viz_selector from a PREVIOUS cell.
    # We access its .value to decide which heatmap to display.
    selected_display_hm = fuzzy_viz_selector.value # Using a unique variable name

    # --- SELECT THE CORRECT CONTROLLER, RULE MATRIX, AND TITLE ---
    # Based on the value from the shared dropdown
    if selected_display_hm == OPT_KP:
        controller_to_display_hm = flc1_viz_instance # FLC1
        matrix_to_display_hm = controller_to_display_hm.kp_rule_matrix
        title_text_hm = "Rule Matrix for Kp (from FLC1)"
        description_hm = """
        This heatmap shows the logic for the Proportional Gain (`Kp`).
        Notice the aggressive response (`VB`, `B`) when the error is large, designed for rapid correction.
        """
    elif selected_display_hm == OPT_KI:
        controller_to_display_hm = flc1_viz_instance # FLC1
        matrix_to_display_hm = controller_to_display_hm.ki_rule_matrix
        title_text_hm = "Rule Matrix for Ki (from FLC1)"
        description_hm = """
        This heatmap shows the logic for the Integral Gain (`Ki`).
        The rules here focus on eliminating steady-state error, often becoming aggressive (`VB`) when the error persists over time.
        """
    else: # OPT_SF
        controller_to_display_hm = flc2_viz_instance # FLC2
        matrix_to_display_hm = controller_to_display_hm.rule_matrix
        title_text_hm = "Rule Matrix for Scaling Factor (from FLC2)"
        description_hm = """
        This heatmap shows the fine-tuning logic. The values are typically conservative, centered around 'M' (Medium),
        to provide small adjustments to the primary gains from FLC1.
        """

    # =======================================================
    # PART 1: Prepare Data for the Heatmap
    # =======================================================
    rule_map_val_hm = {'VB': 4, 'B': 3, 'M': 2, 'S': 1}
    rule_map_text_hm = []
    rule_map_num_hm = []
    rows_hm = len(controller_to_display_hm.terms)
    cols_hm = len(controller_to_display_hm.terms)

    # --- USING UNIQUE LOOP COUNTERS ---
    for row_idx in range(rows_hm): # Changed 'i' to 'row_idx'
        row_text = []
        row_num = []
        for col_idx in range(cols_hm): # Changed 'j' to 'col_idx'
            rule_str = matrix_to_display_hm[row_idx][col_idx]
            row_text.append(rule_str)
            row_num.append(rule_map_val_hm.get(rule_str, 0))
        rule_map_text_hm.append(row_text)
        rule_map_num_hm.append(row_num)

    # =======================================================
    # PART 2: Create the Heatmap Figure
    # =======================================================
    fig_rules_hm = go.Figure(data=go.Heatmap(
        z=rule_map_num_hm,
        x=controller_to_display_hm.terms,
        y=controller_to_display_hm.terms,
        text=rule_map_text_hm,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[
            [0.0, '#27ae60'], [0.33, '#f39c12'],
            [0.66, '#e67e22'], [1.0, '#c0392b']
        ],
        xgap=4, ygap=4, showscale=False
    ))

    # =======================================================
    # PART 3: Update Figure Layout
    # =======================================================
    fig_rules_hm.update_layout(
        xaxis_title="Change of Error (dE)",
        yaxis_title="Error (E)",
        height=550,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#aaa"
    )

    # =======================================================
    # PART 4: Render the final Marimo output with dynamic titles
    # =======================================================
    mo.vstack([
        mo.md(f"## 🚦 The Controller's Brain: {title_text_hm}"),
        mo.md(description_hm),
        mo.ui.plotly(fig_rules_hm)
    ])

    # The cell ends here
    return


@app.cell(hide_code=True)
def _(
    OPT_GA,
    controller_selector,
    go,
    ki_log,
    kp_log,
    make_subplots,
    mo,
    np,
    ref_data,
    sim_settings,
    switch_and_fopid_params,
    t_data,
    w_data,
):
    # The cell starts here

    # 1. Define default variables
    output_view = None

    # 2. Check which controller mode is active
    is_fuzzy_mode = (controller_selector.value == OPT_GA)

    # Prepare data for plotting regardless of mode
    ref_val = ref_data if isinstance(ref_data, (int, float)) else ref_data[0]
    if 'w_data' in globals() and w_data is not None:
        error_signal = np.array([ref_val - w for w in w_data])
    else:
        error_signal = np.array([])


    # 3. Create the plots
    fig_analysis = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "👁️ System Input: Tracking Error",
            "🧠 Final Gain Output: Proportional Gain (Kp)",
            "🧠 Final Gain Output: Integral Gain (Ki)"
        )
    )

    # Plot 1: Error Signal (Always visible)
    fig_analysis.add_trace(go.Scatter(
        x=t_data, y=error_signal, name="Error Signal",
        line=dict(color="#d35400", width=2.5),
        fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.15)'
    ), row=1, col=1)

    # Plot 2: Final Kp Gain over time
    fig_analysis.add_trace(go.Scatter(
        x=t_data, y=kp_log, name="Final Kp",
        line=dict(color="#2980b9", width=2.5)
    ), row=2, col=1)

    # Plot 3: Final Ki Gain over time
    fig_analysis.add_trace(go.Scatter(
        x=t_data, y=ki_log, name="Final Ki",
        line=dict(color="#27ae60", width=2.5)
    ), row=3, col=1)


    # 4. Add annotations and styling
    # Add a vertical line for the load disturbance
    load_time = sim_settings.value["T_load"]
    if sim_settings.value["Load"] > 0.01:
        fig_analysis.add_vline(x=load_time, line_width=2, line_dash="dot", line_color="#f1c40f", opacity=0.8)

    # If in fuzzy mode, add a horizontal rectangle for the switching threshold zone
    if is_fuzzy_mode:
        threshold = switch_and_fopid_params.value["Threshold"]
        fig_analysis.add_hrect(
            y0=-threshold, y1=threshold,
            fillcolor="rgba(128, 128, 128, 0.2)",
            line_width=0,
            layer="below", # Draw the shape below the data line
            row=1, col=1,
            annotation_text="Switching Zone",
            annotation_position="outside top right"
        )

    # --- CORRECTION FOR NameError ---
    # Define the grid_params dictionary before using it
    grid_params = dict(showgrid=True, gridcolor='rgba(84, 110, 122, 0.15)', gridwidth=1)

    # General layout adjustments
    fig_analysis.update_layout(
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#546e7a", family="Segoe UI, sans-serif", size=13),
        showlegend=False,
        margin=dict(t=80, b=30, l=60, r=20)
    )
    # Now, apply the defined grid_params
    fig_analysis.update_xaxes(**grid_params, row=1, col=1); fig_analysis.update_yaxes(**grid_params, row=1, col=1)
    fig_analysis.update_xaxes(**grid_params, row=2, col=1); fig_analysis.update_yaxes(**grid_params, row=2, col=1)
    fig_analysis.update_xaxes(**grid_params, row=3, col=1); fig_analysis.update_yaxes(**grid_params, row=3, col=1)


    # 5. Assemble the final display
    title_text = "🧠 Intelligent Controller Analysis" if is_fuzzy_mode else "📈 Standard Controller Gain Analysis"
    description_text = """
    These plots show the final `Kp` and `Ki` values used by the controller at each moment.
    In `GA-Optimized` mode, you can see how the gains change dynamically when the error is large (outside the grey switching zone)
    and then settle to the fixed steady-state values when the error is small (inside the zone).
    """ if is_fuzzy_mode else """
    These plots show the constant `Kp` and `Ki` gains used by the standard PID/FOPID controller.
    """

    output_view = mo.vstack([
        mo.md(f"#### {title_text}"),
        mo.md(description_text),
        mo.ui.plotly(fig_analysis)
    ])

    # 6. Render the final view
    output_view

    # The cell ends here
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧬 Evolutionary Optimization Engine
    """)
    return


@app.cell
def _(OPT_GA, np, simulate_pmsm_system):
    # The cell starts here

    class GeneticOptimizerEngine:
        """
        A Genetic Algorithm (GA) engine designed to find the optimal controller parameters
        for the transient response of the AFFOPID system.
        """
        def __init__(self, motor_phys, vehicle_phys, sim_setup, switch_params, custom_bounds=None):
            """
            Initializes the optimizer with all necessary simulation parameters.
            """
            self.motor_phys = motor_phys
            self.vehicle_phys = vehicle_phys
            self.sim_setup = sim_setup
            self.switch_params = switch_params

            if custom_bounds:
                self.bounds = custom_bounds
            else:
                self.bounds = [
                    (0.1, 150.0), (0.1, 300.0), (0.0, 1.0),
                    (0.1, 1.5), (0.1, 1.5), (0.1, 5.0)
                ]

        def _evaluate_fitness(self, individual_genes):
            """
            Calculates the fitness of a single set of controller parameters.
            """
            controller_candidate = {
                "Kp": individual_genes[0], "Ki": individual_genes[1], "Kd": individual_genes[2],
                "Lambda": individual_genes[3], "Mu": individual_genes[4], "Fuzzy_Scale": individual_genes[5]
            }
            fast_scenario = self.sim_setup.copy()
            fast_scenario["Time"] = 0.5

            time, speed, _, _, _, _, _, ref = simulate_pmsm_system(
                controller_candidate, fast_scenario, self.motor_phys, self.vehicle_phys, 
                self.switch_params, strategy_type=OPT_GA
            )

            dt = time[1] - time[0]
            ref_val = ref if isinstance(ref, (int, float)) else ref[0]
            error = ref_val - speed
            ise_cost = np.sum(np.square(error)) * dt

            is_unstable = np.isnan(ise_cost) or np.isinf(ise_cost) or np.max(speed) > (ref_val * 2.5)
            if is_unstable:
                return 1e6 

            return ise_cost

        # --- START OF CORRECTION ---
        # Added 'crossover_rate' to the function definition
        def run(self, pop_size=20, generations=10, mutation_rate=0.1, crossover_rate=0.7):
        # --- END OF CORRECTION ---
            """
            Executes the main Genetic Algorithm loop.
            """
            # 1. Initialization
            population = [[np.random.uniform(L, H) for L, H in self.bounds] for _ in range(pop_size)]
            best_solution = None
            best_fitness = float('inf')

            convergence_history = []
            full_trial_history = []

            # 2. Main Evolutionary Loop
            for gen in range(generations):
                fitness_scores = [self._evaluate_fitness(individual) for individual in population]

                for i, individual in enumerate(population):
                    full_trial_history.append({
                        'Generation': gen + 1, 'Kp': individual[0], 'Ki': individual[1], 'Kd': individual[2],
                        'Lambda': individual[3], 'Mu': individual[4], 'Alpha': individual[5],
                        'Cost': fitness_scores[i]
                    })

                min_fitness_in_gen = min(fitness_scores)
                if min_fitness_in_gen < best_fitness:
                    best_fitness = min_fitness_in_gen
                    best_solution = population[fitness_scores.index(min_fitness_in_gen)]

                convergence_history.append(best_fitness)

                # 3. Evolution: Create the next generation
                new_population = [best_solution] # Elitism

                while len(new_population) < pop_size:
                    # Parent Selection (Tournament)
                    p1_idx, p2_idx = np.random.choice(range(pop_size), 2, replace=False)
                    parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p2_idx]

                    p3_idx, p4_idx = np.random.choice(range(pop_size), 2, replace=False)
                    parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p4_idx]

                    # Crossover (Arithmetic)
                    if np.random.random() < crossover_rate:
                        alpha = np.random.random()
                        child = [p1 * alpha + p2 * (1 - alpha) for p1, p2 in zip(parent1, parent2)]
                    else:
                        child = parent1[:]

                    # Mutation
                    for i in range(len(child)):
                        if np.random.random() < mutation_rate:
                            child[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])

                    new_population.append(child)

                population = new_population

            return best_solution, best_fitness, convergence_history, full_trial_history
    return (GeneticOptimizerEngine,)


@app.cell(hide_code=True)
def _(mo):
    # The cell starts here

    flowchart = mo.mermaid("""
    graph TD
        %% --- Styling Definitions ---
        classDef startend fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white,rx:10,ry:10;
        classDef proc fill:#ecf0f1,stroke:#34495e,stroke-width:2px,color:#2c3e50;
        classDef decision fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:#d35400,rx:5,ry:5;
        classDef sim fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
        classDef loop fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,stroke-dasharray: 5 5,color:white;
        classDef note fill:#fef9e7,stroke:#f39c12,stroke-width:1px,color:#b06a00;


        %% --- Nodes ---
        Start([Start Optimization]):::startend

        Init[Initialize Population<br/>Generate random candidate solutions for:<br/><b>[Kp, Ki, Kd, λ, µ, Fuzzy_Scale]</b>]:::proc

        subgraph "Fitness Evaluation Loop (for each candidate)"
            direction LR
            Sim[Run PMSM Simulation<br/>(Using the full Switched AFFOPID-FOPID model)]:::sim
            -->
            Calc[Calculate Fitness Cost<br/>Cost = ISE = ∫ e²(t) dt]:::sim
        end

        Check{All Generations<br/>Completed?}:::decision

        subgraph "Evolution: Create New Generation"
            direction TB
            Sel[Selection<br/>(Elitism + Tournament)]:::loop
            -->
            Cross[Crossover<br/>(Arithmetic Combination)]:::loop
            -->
            Mut[Mutation<br/>(Random Perturbation)]:::loop
        end

        Stop([Output Optimal Parameters]):::startend

        Note[Note: The GA optimizes the 6 controller parameters.<br/>The Switch Threshold and Fixed Gains<br/>are treated as fixed conditions during the simulation.]:::note


        %% --- Connections ---
        Start --> Init
        Init --> Sim
        Calc --> Check

        Check -- No --> Sel
        Sel --> Cross
        Cross --> Mut
        Mut -- New Population --> Sim

        Check -- Yes --> Stop

        Sim -.-> Note

    """)

    mo.vstack([
        mo.md("### 🔄 Optimization Process Flowchart"),
        mo.md("*This diagram illustrates the step-by-step logic executed by the `GeneticOptimizerEngine`.*"),
        flowchart
    ])

    # The cell ends here
    return


@app.cell(hide_code=True)
def _(mo):
    # The cell starts here

    mo.md("### 🧬 Evolutionary Optimization Console")

    # 1. Initialize State Management for the ENTIRE GA section
    get_ga_run_count, set_ga_run_count = mo.state(0)
    get_ga_cached_view, set_ga_cached_view = mo.state(
        mo.md("ℹ️ *System Ready. Configure GA parameters and click Start.*")
    )
    # --- START OF CORRECTION ---
    # Define all the new state variables needed to store the GA results.
    # This makes them globally available to other cells.
    get_ga_full_history, set_ga_full_history = mo.state(None)
    get_ga_best_genes, set_ga_best_genes = mo.state(None)
    get_ga_final_cost, set_ga_final_cost = mo.state(None)
    get_ga_convergence, set_ga_convergence = mo.state(None)
    # --- END OF CORRECTION ---


    # 2. General Algorithm Settings
    ga_controls = mo.ui.dictionary({
        "pop": mo.ui.number(10, 100, value=20, step=1, label="Population Size"),
        "gen": mo.ui.number(5, 200, value=10, step=1, label="Generations"),
        "mut": mo.ui.number(0.01, 0.5, value=0.1, step=0.01, label="Mutation Rate"),
        "cross": mo.ui.number(0.5, 1.0, value=0.7, step=0.05, label="Crossover Rate")
    })

    # 3. Search Space Bounds Configuration
    ga_bounds_config = {
        "Kp":   {"label": "Prop. Gain (Kp)",   "min": 0.1, "max": 500.0, "start_min": 0.1,  "start_max": 150.0},
        "Ki":   {"label": "Integ. Gain (Ki)",   "min": 0.1, "max": 500.0, "start_min": 0.1,  "start_max": 300.0},
        "Kd":   {"label": "Deriv. Gain (Kd)",   "min": 0.0, "max": 100.0, "start_min": 0.0,  "start_max": 2.0},
        "Lam":  {"label": "Int. Order (λ)",     "min": 0.1, "max": 2.0,   "start_min": 0.8,  "start_max": 1.2},
        "Mu":   {"label": "Diff. Order (µ)",    "min": 0.1, "max": 2.0,   "start_min": 0.8,  "start_max": 1.2},
        "Fuz":  {"label": "Fuzzy Scale (α)",    "min": 0.1, "max": 10.0,  "start_min": 0.5,  "start_max": 3.0}
    }

    # Create the UI elements from the configuration dictionary
    bounds_ui_elements = {
        key: mo.ui.array([
            mo.ui.number(conf["min"], conf["max"], value=conf["start_min"], full_width=True),
            mo.ui.number(conf["min"], conf["max"], value=conf["start_max"], full_width=True)
        ]) for key, conf in ga_bounds_config.items()
    }

    # 4. Layout Helper for Bounds Table
    def create_bound_rows():
        rows = [
            mo.hstack([
                mo.md("Parameter").style({"width": "140px", "font-weight": "bold"}),
                mo.md("Lower Bound").style({"flex": "1", "text-align": "center", "font-weight": "bold"}),
                mo.md("").style({"width": "40px"}),
                mo.md("Upper Bound").style({"flex": "1", "text-align": "center", "font-weight": "bold"}),
            ]),
            mo.md("---")
        ]
        for key, conf in ga_bounds_config.items():
            rows.append(mo.hstack([
                mo.md(f"**{conf['label']}**").style({"width": "140px", "align-self": "center"}),
                bounds_ui_elements[key][0],
                mo.md("↔️").style({"padding": "0 10px", "align-self": "center"}),
                bounds_ui_elements[key][1]
            ], align="center"))
        return mo.vstack(rows)


    # 5. Accordion Widget for Advanced Settings
    bounds_widget = mo.accordion({
        "⚙️ Configure Search Space (Bounds)": create_bound_rows()
    })

    # 6. Action Button
    run_ga_btn = mo.ui.button(
        label="🚀 Start Evolution Process",
        kind="success",
        value=0,
        on_click=lambda value: value + 1
    )

    # 7. Final Panel Assembly
    mo.vstack([
        mo.callout(
            mo.vstack([
                mo.md("### 🛠️ Optimization Configuration"),
                ga_controls,
                mo.md("---"),
                bounds_widget,
                mo.md(""), # Spacer
                run_ga_btn
            ]),
            kind="neutral"
        )
    ])

    # The cell ends here
    return (
        bounds_ui_elements,
        ga_controls,
        get_ga_best_genes,
        get_ga_cached_view,
        get_ga_convergence,
        get_ga_full_history,
        get_ga_run_count,
        run_ga_btn,
        set_ga_best_genes,
        set_ga_cached_view,
        set_ga_convergence,
        set_ga_final_cost,
        set_ga_full_history,
        set_ga_run_count,
    )


@app.cell
def _(
    GeneticOptimizerEngine,
    bounds_ui_elements,
    ga_controls,
    get_ga_cached_view,
    get_ga_run_count,
    mo,
    motor_ui,
    run_ga_btn,
    set_ga_best_genes,
    set_ga_cached_view,
    set_ga_convergence,
    set_ga_final_cost,
    set_ga_full_history,
    set_ga_run_count,
    sim_settings,
    switch_and_fopid_params,
    vehicle_ui,
):
    # The cell starts here

    # This cell depends on 'run_ga_btn' and other UI elements from the console cell.
    # Its main purpose is to EXECUTE the GA and STORE the results in the state.

    raw_val = run_ga_btn.value
    current_clicks = raw_val if raw_val is not None else 0
    last_clicks = get_ga_run_count()
    is_new_click = (current_clicks > last_clicks)

    if is_new_click:
        # --- This block runs ONLY when the "Start" button is clicked ---
    
        # 1. Prepare and Run the GA
        user_defined_bounds = [
            tuple(bounds_ui_elements["Kp"].value), tuple(bounds_ui_elements["Ki"].value),
            tuple(bounds_ui_elements["Kd"].value), tuple(bounds_ui_elements["Lam"].value),
            tuple(bounds_ui_elements["Mu"].value), tuple(bounds_ui_elements["Fuz"].value),
        ]

        optimizer = GeneticOptimizerEngine(
            motor_phys=motor_ui.value,
            vehicle_phys=vehicle_ui.value,
            sim_setup=sim_settings.value,
            switch_params=switch_and_fopid_params.value,
            custom_bounds=user_defined_bounds
        )

        with mo.status.spinner(title="🧬 Processing Genome... Analyzing generations..."):
            best_genes, final_cost, history_scores, full_history = optimizer.run(
                pop_size=int(ga_controls.value["pop"]),
                generations=int(ga_controls.value["gen"]),
                mutation_rate=ga_controls.value["mut"],
                crossover_rate=ga_controls.value.get("cross", 0.7)
            )
    
        # 2. Store ALL results in state variables for other cells to use.
        set_ga_full_history(full_history)
        set_ga_best_genes(best_genes)
        set_ga_final_cost(final_cost)
        set_ga_convergence(history_scores)
    
        # 3. Prepare a simple confirmation message as the output for this cell
        if best_genes:
            new_view = mo.md(f"**✅ Evolution Complete!** Best cost found: `{final_cost:.6f}`. Results are now available for analysis in the cells below.")
        else:
            new_view = mo.md("⚠️ **Optimization failed. Check console for potential errors.**")

        # 4. Update the run counter and cache the confirmation message
        set_ga_run_count(current_clicks)
        set_ga_cached_view(new_view)
        final_output = new_view

    else:
        # If the button was not clicked, just show the last cached message
        final_output = get_ga_cached_view()

    # Render the final output for this cell (which is just the message)
    final_output

    # The cell ends here
    return


@app.cell
def _(mo):
    # Cell 1: Filter Slider (No changes needed)
    ga_result_filter_slider = mo.ui.range_slider(
        0, 100, value=[0, 75], step=1,full_width=True,
        label="Filter Results by Cost (Show best %):"
    )

    return (ga_result_filter_slider,)


@app.cell
def _(
    ga_result_filter_slider,
    get_ga_best_genes,
    get_ga_convergence,
    get_ga_full_history,
    go,
    mo,
):
    # The cell starts here - VISUALIZATION CELL

    # This cell is for VISUALIZATION. It depends on the filter slider and the results stored in the state.
    # It will re-run reactively whenever the slider's value changes.

    # 1. Retrieve all necessary data from the global state
    full_history_data = get_ga_full_history()
    best_genes_data = get_ga_best_genes()
    history_scores_data = get_ga_convergence()

    # 2. Main display logic: Check if data from the GA is available
    if full_history_data and best_genes_data and history_scores_data:
        # --- This block runs if GA results are available ---
    
        # a. Prepare and Filter Data using the slider
        import pandas as pd
        history_df = pd.DataFrame(full_history_data).sort_values(by='Cost').reset_index(drop=True)
        min_percent, max_percent = ga_result_filter_slider.value
        start_idx = int(len(history_df) * (min_percent / 100.0))
        # Make sure end_idx is at least start_idx
        end_idx = max(start_idx, int(len(history_df) * (max_percent / 100.0)))
        if start_idx == end_idx and start_idx < len(history_df): # Ensure slice is not empty if possible
            end_idx += 1
        filtered_df = history_df.iloc[start_idx:end_idx]

        # b. Create Static Summary View (Table + Convergence Curve)
        results_table = f"""
        | Parameter | Symbol | Best Value |
        | :--- | :---: | :---: |
        | Proportional | $K_p$ | **{best_genes_data[0]:.5f}** |
        | Integral | $K_i$ | **{best_genes_data[1]:.5f}** |
        | Derivative | $K_d$ | **{best_genes_data[2]:.5f}** |
        | Int. Order | $\lambda$ | **{best_genes_data[3]:.4f}** |
        | Diff. Order | $\mu$ | **{best_genes_data[4]:.4f}** |
        | Fuzzy Scale | $\\alpha$ | **{best_genes_data[5]:.4f}** |
        """
        fig_conv = go.Figure(data=go.Scatter(
            x=list(range(1, len(history_scores_data) + 1)), y=history_scores_data,
            mode='lines+markers', line=dict(color='#2ecc71', width=3)
        ))
        fig_conv.update_layout(title="<b>Overall Convergence History</b>", height=350, template="plotly_white")
    
        summary_view = mo.vstack([
            mo.md("### 🏆 Best Solution & Convergence"),
            mo.hstack([mo.md(results_table), mo.ui.plotly(fig_conv)], align="center"),
            mo.md("---")
        ])

        # c. Create Filtered Visualizations
        if not filtered_df.empty:
            # Parallel Coordinates Plot
            fig_par = go.Figure(data=go.Parcoords(
                line=dict(color=filtered_df['Cost'], colorscale='Turbo_r', showscale=True),
                dimensions=[{'label': col, 'values': filtered_df[col]} for col in filtered_df.columns]
            ))
            fig_par.update_layout(title="<b>Parallel Coordinates (Filtered)</b>", height=500)

            # 3D Scatter Plot
            fig_3d = go.Figure(data=go.Scatter3d(
                x=filtered_df['Kp'], y=filtered_df['Ki'], z=filtered_df['Cost'], mode='markers',
                marker=dict(size=4, color=filtered_df['Cost'], colorscale='Viridis_r')
            ))
            fig_3d.update_layout(title="<b>3D Landscape (Filtered)</b>", height=500)
        
            # Detailed History Table
            history_table = mo.ui.table(filtered_df.round(4).to_dict('records'), pagination=True, page_size=10, label="Filtered Trials Log")

            # Assemble the interactive/filtered part of the view
            analysis_view = mo.vstack([
                mo.ui.plotly(fig_par),
                mo.accordion({
                    "View 3D Landscape (Filtered)": mo.ui.plotly(fig_3d),
                    "View Detailed Log (Filtered)": history_table
                })
            ])
        else:
            analysis_view = mo.md("ℹ️ *No data in the selected filter range. Adjust the slider.*")

        # d. Render the complete view for this cell
        l_l=mo.vstack([ga_result_filter_slider,
            summary_view,
            analysis_view
        ])

    else:
        # This message is shown before the GA has been run for the first time
        l_l=mo.md("⏳ *Run the Genetic Algorithm to generate results for analysis.*")

    l_l
    return


if __name__ == "__main__":
    app.run()
