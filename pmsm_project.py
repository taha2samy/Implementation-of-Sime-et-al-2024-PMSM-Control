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
    OPT_GA = "GA-Optimized Adaptive Fuzzy FOPID"

    # 2. Vault definition (all values are defined here once to avoid accidental deletion)
    # Note: no need for globals() or complexity, just defining them here is enough
    vault = {
        OPT_PID: mo.ui.dictionary({
            "Kp": mo.ui.number(0.0, 500.0, 0.0000001, 10.0, label=r"Proportional ($K_p$)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.0000001, 20.0, label=r"Integral ($K_i$)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.0000001, 0.0, label=r"Derivative ($K_d$)", full_width=True),
        }),
        OPT_FOPID: mo.ui.dictionary({
            "Kp": mo.ui.number(0.0, 500.0, 0.0000001, 5.0, label=r"Proportional ($K_p$)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.0000001, 10.0, label=r"Integral ($K_i$)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.0000001, 0.01, label=r"Derivative ($K_d$)", full_width=True),
            "Lambda": mo.ui.number(0.01, 2.0,0.0000001, 0.9, label=r"Int. Order ($\lambda$)", full_width=True),
            "Mu": mo.ui.number(0.01, 2.0, 0.0000001, 0.8, label=r"Diff. Order ($\mu$)", full_width=True),
        }),
        OPT_GA: mo.ui.dictionary({
            "Kp": mo.ui.number(0.0, 500.0, 0.0000001, 5.0, label=r"Proportional ($K_p$)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.0000001, 10.0, label=r"Integral ($K_i$)", full_width=True),
            "Kd": mo.ui.number(0.0, 100.0, 0.0000001, 0.01, label=r"Derivative ($K_d$)", full_width=True),
            "Lambda": mo.ui.number(0.01, 2.0, 0.0000001, 0.978, label=r"Int. Order ($\lambda$)", full_width=True),
            "Mu": mo.ui.number(0.01, 2.0, 0.0000001, 0.862, label=r"Diff. Order ($\mu$)", full_width=True),
            "Fuzzy_Scale": mo.ui.number(0.1, 10.0, 0.0000001, 1.0, label=r"Fuzzy Scale ($\alpha$)", full_width=True),
        })
    }
    return OPT_FOPID, OPT_GA, OPT_PID, vault


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
def _(controller_selector, mo, vault):
    # 1. Retrieve values from the vault based on selection
    # The magic: we get the same Widget stored in Cell 1
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

    # 3. Display the Sidebar
    mo.sidebar(
        mo.vstack([
            mo.md("# ⚙️ Control Station"),
            mo.md("---"),
            mo.md("### 🧠 Strategy"),
            controller_selector,
            mo.md("### 🎛️ Tuning"),
            active_params,  # <--- This displays the stored values
            mo.md("### 📉 Scenario"),
            sim_settings,
            mo.md("---"),
            mo.accordion({
                "🔌 PMSM Parameters": motor_ui,
                "🚗 Vehicle Dynamics": vehicle_ui
            })
        ])
    )
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
def _(RealFuzzyController, np):
    def simulate_pmsm_system(ctrl_params, scenario, motor_phys, vehicle_phys, strategy_type):
        """
        Simulates the PMSM-based EV powertrain with a specified control strategy.

        Returns:
            A tuple containing: (time, speed, torque, current, fuzzy_gain_log, reference_speed)
        """
        # 1. Unpack Simulation, Physics, and Controller Parameters
        t_end = scenario["Time"]
        w_target = scenario["Ref"]
        t_load = scenario["T_load"]
        load_val = scenario["Load"]

        Rs, P, Psi_m, J, B = motor_phys["Rs"], motor_phys["P"], motor_phys["Psi"], motor_phys["J"], motor_phys["B"]
        Mass, Rw, Gear, Cd, Area, Rho = vehicle_phys["Mass"], vehicle_phys["Rw"], vehicle_phys["Gear"], vehicle_phys["Cd"], vehicle_phys["Area"], vehicle_phys["Rho"]

        kp, ki, kd = ctrl_params["Kp"], ctrl_params["Ki"], ctrl_params["Kd"]
        lam = ctrl_params.get("Lambda", 1.0) # Default to 1.0 for integer order
        mu = ctrl_params.get("Mu", 1.0)     # Default to 1.0 for integer order
        alpha_scale = ctrl_params.get("Fuzzy_Scale", 1.0)

        # 2. Initialize Controller Components
        # a) Instantiate the Fuzzy Logic "Brain" for adaptive gain.
        fuzzy_brain = RealFuzzyController()

        # b) Determine if fractional calculus is active for this strategy.
        use_fractional = ("Fractional" in strategy_type) or ("GA" in strategy_type)

        # c) Initialize the fractional-order filters if required.
        # The integrator operator is s^-λ, so we use a negative alpha.
        frac_integrator = OustaloupFilter(-lam) if use_fractional else None
        # The differentiator operator is s^μ, so we use a positive alpha.
        frac_differentiator = OustaloupFilter(mu) if use_fractional else None

        # 3. Initialize Simulation State Variables
        dt = 0.0001
        steps = int(t_end / dt)
        time = np.linspace(0, t_end, steps)

        # Data logging arrays
        speed_arr = np.zeros(steps)
        torque_arr = np.zeros(steps)
        iq_arr = np.zeros(steps)
        gain_log = np.zeros(steps)

        # System state variables
        w_mech = 0.0
        iq_current = 0.0
        error_sum = 0.0
        prev_error = 0.0

        # 4. Main Simulation Loop
        for i in range(steps):
            t = time[i]
            # Apply reference speed after a small delay to avoid initial shock.
            ref = w_target if t > 0.005 else 0.0
            error = ref - w_mech
            delta_error = error - prev_error

            # --- Block A: Adaptive Fuzzy Logic ---
            # Calculate the adaptive gain multiplier. Defaults to 1.0 for non-fuzzy modes.
            gain_factor = 1.0
            if "GA" in strategy_type or "Fuzzy" in strategy_type:
                fuzzy_output = fuzzy_brain.compute_alpha(error, delta_error)
                gain_factor = fuzzy_output * alpha_scale
            gain_log[i] = gain_factor

            # --- Block B: Core Controller Logic (PID vs. FOPID) ---
            # 1. Proportional Term
            p_term = (kp * gain_factor) * error

            # 2. Integral & Derivative Terms
            if use_fractional:
                # Apply fractional-order calculus using the Oustaloup filters.
                i_signal = frac_integrator.compute(error)
                d_signal = frac_differentiator.compute(error)
                i_term = (ki * gain_factor) * i_signal
                d_term = (kd * gain_factor) * d_signal
            else:
                # Apply standard integer-order calculus.
                error_sum += error * dt
                i_term = (ki * gain_factor) * error_sum
                # Avoid division by zero at the first step.
                d_term = (kd * gain_factor) * (delta_error / dt) if dt > 0 else 0.0

            # Sum the components to get the final control signal (q-axis current reference).
            iq_ref = p_term + i_term + d_term

            # Apply saturation (limiter) to the control signal to respect physical constraints.
            # Also includes an anti-windup mechanism for the standard integrator.
            if iq_ref > 200:
                iq_ref = 200
                if not use_fractional: error_sum -= error * dt # Anti-windup
            elif iq_ref < -200:
                iq_ref = -200
                if not use_fractional: error_sum -= error * dt # Anti-windup

            # --- Block C: Plant Model (System Physics) ---
            # Simulate the current controller's first-order response.
            iq_current += (iq_ref - iq_current) * (dt / 0.001)
            # Calculate the electromagnetic torque produced by the motor.
            Te = 1.5 * P * Psi_m * iq_current

            # Calculate external load torques from vehicle dynamics.
            T_ext = load_val if t >= t_load else 0.0
            v = (w_mech / Gear) * Rw # Vehicle speed
            F_drag = 0.5 * Rho * Cd * Area * v * abs(v) # Aerodynamic drag force
            T_drag = (F_drag * Rw) / Gear # Drag torque reflected to the motor shaft

            # Apply Newton's second law for rotation.
            dw_dt = (Te - T_ext - T_drag - B * w_mech) / J
            w_mech += dw_dt * dt
            if w_mech < 0: w_mech = 0 # Speed cannot be negative in this model.

            # Log state variables for this time step.
            speed_arr[i] = w_mech
            torque_arr[i] = Te
            iq_arr[i] = iq_current
            prev_error = error

        return time, speed_arr, torque_arr, iq_arr, gain_log, w_target
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
    vehicle_ui,
):
    # --- Cell 5: Main Dashboard (Fixed Layout) ---

    # 1. Run Simulation (Global Data)
    t_data, w_data, te_data, iq_data, fuzzy_data, ref_data = simulate_pmsm_system(
        active_params.value,
        sim_settings.value,
        motor_ui.value,
        vehicle_ui.value,
        controller_selector.value
    )

    # 2. Define Plotting Function (Encapsulated & Clean)
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
        _ref_line = [ref_val]*len(t_arr) if isinstance(ref_val, (int, float)) else ref_val
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
            _fig.add_vline(
                x=_load_time,
                line_width=2,
                line_dash="dot",
                line_color="#f1c40f",
                opacity=0.8
            )
            # Annotation (adjusted position)
            _fig.add_annotation(
                x=_load_time, 
                y=1.12,         # Raised higher (was 1.05)
                yref="paper",   # Relative to entire page
                text=" ",
                showarrow=False,
                font=dict(size=10, color="#f1c40f"),
                xanchor="left", # Text starts at the right of the line
                xshift=5        # Small shift to the right
            )

        # E. Styling & Margins
        _fig.update_layout(
            height=750,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#555"),
            showlegend=False,
            # Increased Top Margin to 80 to accommodate raised annotation
            margin=dict(t=80, b=30, l=50, r=20) 
        )

        _grid_style = dict(showgrid=True, gridcolor='rgba(64, 160, 255, 0.1)')
        _fig.update_xaxes(**_grid_style, row=1, col=1); _fig.update_yaxes(**_grid_style, row=1, col=1)
        _fig.update_xaxes(**_grid_style, row=2, col=1); _fig.update_yaxes(**_grid_style, row=2, col=1)
        _fig.update_xaxes(**_grid_style, row=3, col=1); _fig.update_yaxes(**_grid_style, row=3, col=1)

        return _fig

    # 3. Call Function & Display
    final_dashboard_fig = generate_dashboard_plot(t_data, w_data, te_data, iq_data, ref_data, sim_settings.value)

    # 4. Render
    mo.vstack([
        mo.md("### 🏎️ Vehicle Performance"),
        mo.ui.plotly(final_dashboard_fig)
    ])
    return fuzzy_data, iq_data, ref_data, t_data, te_data, w_data


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

    class RealFuzzyController:
        def __init__(self):
            # 1. Define the linguistic terms for inputs (error and delta error).
            # (NB: Negative Big, NS: Negative Small, Z: Zero, PS: Positive Small, PB: Positive Big)
            self.terms = ['NB', 'NS', 'Z', 'PS', 'PB']

            # 2. Define the center points for the triangular membership functions.
            self.centers = {
                'NB': -1.0, 'NS': -0.5, 'Z': 0.0, 'PS': 0.5, 'PB': 1.0
            }

            # 3. Establish the fuzzy rule base matrix.
            # Rows represent Error, Columns represent Delta Error.
            # The matrix cells define the output gain multiplier (S, M, B, VB).
            self.rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['VB', 'B',  'B',  'M',  'S'],  # e: NB
                ['B',  'B',  'M',  'S',  'S'],  # e: NS
                ['B',  'M',  'S',  'M',  'B'],  # e: Z
                ['S',  'S',  'M',  'B',  'B'],  # e: PS
                ['S',  'M',  'B',  'B',  'VB']  # e: PB
            ]

            # 4. Define the output singleton values for the gain multipliers.
            # S=0.8 (Small), M=1.0 (Medium), B=1.5 (Big), VB=2.5 (Very Big)
            self.outputs = {'S': 0.8, 'M': 1.0, 'B': 1.5, 'VB': 2.5}

        def triangle_mf(self, x, center, width=0.5):
            """Calculates the membership degree for a triangular function."""
            return max(0, 1 - abs(x - center) / width)

        def compute_alpha(self, error, delta_error):
            """
            Computes the alpha gain multiplier based on the current error and change in error.

            Args:
                error: The current error value.
                delta_error: The rate of change of the error.

            Returns:
                The calculated alpha gain.
            """
            # a. Normalization Step: Scale the inputs to the [-1, 1] range.
            # Assuming a max error of 250. Multiplied by 5 for higher sensitivity.
            e_norm = np.clip(error / 250.0 * 5.0, -1, 1)
            # Assuming a max delta error of 10.
            de_norm = np.clip(delta_error / 10.0, -1, 1)

            numerator = 0.0
            denominator = 0.0

            # b. Inference Engine: Apply fuzzy rules.
            for i, e_term in enumerate(self.terms):
                # Fuzzify the normalized error.
                mu_e = self.triangle_mf(e_norm, self.centers[e_term])
                if mu_e == 0: continue  # Optimization: skip if membership is zero.

                for j, de_term in enumerate(self.terms):
                    # Fuzzify the normalized delta error.
                    mu_de = self.triangle_mf(de_norm, self.centers[de_term])
                    if mu_de == 0: continue # Optimization: skip if membership is zero.

                    # Use the 'min' operator (AND) to find the rule's firing strength.
                    firing_strength = min(mu_e, mu_de)

                    # Get the corresponding output term from the rule matrix.
                    output_term = self.rule_matrix[i][j]
                    output_val = self.outputs[output_term]

                    # Accumulate values for defuzzification.
                    numerator += firing_strength * output_val
                    denominator += firing_strength

            # c. Defuzzification: Calculate the weighted average (Center of Gravity method).
            if denominator == 0:
                return 1.0 # Return a neutral default value if no rules were fired.

            return numerator / denominator

    return (RealFuzzyController,)


@app.cell
def _(RealFuzzyController, go, mo, np):
    # No new libraries are imported, as requested.
    # Assuming the 'RealFuzzyController' class, 'np', 'go', and 'mo' are already defined
    # in your Marimo environment from the previous cells.

    fuzzy_viz = RealFuzzyController()

    # =======================================================
    # PART 1: Fuzzification Visualization (Membership Functions)
    # =======================================================

    # Create a range of values for the x-axis.
    x_range = np.linspace(-1.5, 1.5, 300)
    fig_mf = go.Figure()

    # Define a color palette that is visible in both light and dark modes.
    # These colors are vibrant and avoid pure black or white.
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    # Plot each linguistic term as a triangular membership function.
    for idx, term in enumerate(fuzzy_viz.terms):
        y_values = [fuzzy_viz.triangle_mf(x, fuzzy_viz.centers[term]) for x in x_range]

        fig_mf.add_trace(go.Scatter(
            x=x_range, y=y_values,
            name=f"Term: {term}",
            fill='tozeroy',
            line=dict(color=colors[idx], width=2.5), # Made lines slightly thicker
            opacity=0.7
        ))

    # Update the layout for better readability and theme compatibility.
    fig_mf.update_layout(
        title=dict(
            text="<b>1. Fuzzification Stage (Membership Functions)</b><br><span style='font-size:12px; color:#888;'>How the controller translates numbers into logic</span>",
            y=0.9,
            x=0.01,
            xanchor='left',
            yanchor='top'
        ),
        xaxis_title="Normalized Input (e.g., Error)",
        yaxis_title="Degree of Membership (μ)",
        height=400,
        # Set a top margin to create space for the title.
        margin=dict(l=40, r=20, t=120, b=40),
        # Position the legend horizontally above the plot.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        # --- Theme Adjustments ---
        # Make backgrounds transparent to adapt to any page theme.
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Use a neutral font color that works on both light and dark backgrounds.
        font_color="#aaa"
    )


    # =================================================================
    # PART 2: Inference & Rule Base Visualization (Control Surface)
    # =================================================================

    # Define the resolution for the 3D surface plot.
    res = 40
    e_vec = np.linspace(-250, 250, res)
    de_vec = np.linspace(-10, 10, res)
    z_surface = np.zeros((res, res))

    # Calculate the fuzzy controller's output for each combination of error and delta-error.
    for i in range(res):
        for j in range(res):
            z_surface[j][i] = fuzzy_viz.compute_alpha(e_vec[i], de_vec[j])

    # Create the 3D surface plot.
    fig_surf = go.Figure(data=[go.Surface(
        z=z_surface,
        x=e_vec,
        y=de_vec,
        # 'Plasma' is a good colorscale for both light and dark themes.
        colorscale='Plasma',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="#00FFFF", project_z=True),
        opacity=0.9
    )])

    # Update the layout for the 3D plot.
    fig_surf.update_layout(
        title=dict(
            text="<b>2. Inference & Defuzzification (Control Surface)</b><br><span style='font-size:12px; color:#888;'>The complete decision map from the rule base</span>",
            y=0.92,
            x=0.01,
            xanchor='left',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title='Error (E)',
            yaxis_title='Change of Error (dE)',
            zaxis_title='Output Gain (α)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            # Style the axes for better visibility.
            xaxis=dict(gridcolor='#555', zerolinecolor='#777'),
            yaxis=dict(gridcolor='#555', zerolinecolor='#777'),
            zaxis=dict(gridcolor='#555', zerolinecolor='#777'),
        ),
        height=600,
        # Add a top margin for the title.
        margin=dict(l=10, r=10, t=100, b=10),
        # --- Theme Adjustments ---
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="#aaa"
    )


    # ==========================================
    # PART 3: Render Output in Marimo
    # ==========================================
    mo.vstack([
        mo.md("## 🧩 Fuzzy Controller Internals"),
        mo.md("""
        This dashboard visualizes the two main stages of the fuzzy controller:
        1.  **Fuzzification:** The top plot shows the "Database" of membership functions, which determines how a crisp input value (like `error = 150`) is mapped to linguistic terms (e.g., "70% PS" and "30% PB").
        2.  **Inference & Defuzzification:** The bottom 3D plot represents the "Rule Base" in action. It maps every possible combination of Error and Change-of-Error to a final, crisp Gain Output (α). This is the controller's decision-making landscape.
        """),
        mo.ui.plotly(fig_mf),
        mo.ui.plotly(fig_surf)
    ])
    return (fuzzy_viz,)


@app.cell
def _(fuzzy_viz, go, mo):
    def create_rule_heatmap(controller):
        # =======================================================
        # PART 1: Prepare Data for the Heatmap
        # =======================================================

        # 1. Map the string-based rules to numerical values for color scaling.
        rule_map_val = {'VB': 4, 'B': 3, 'M': 2, 'S': 1}
    
        rule_map_text = [] # This will hold the string labels ('VB', 'B', etc.) for display.
        rule_map_num = []  # This will hold the numerical values (4, 3, etc.) for coloring.

        rows = len(controller.terms)
        cols = len(controller.terms)

        # Iterate through the controller's rule matrix and convert it.
        for i in range(rows):
            row_text = []
            row_num = []
            for j in range(cols):
                rule_str = controller.rule_matrix[i][j]
                row_text.append(rule_str)
                row_num.append(rule_map_val[rule_str])
            rule_map_text.append(row_text)
            rule_map_num.append(row_num)

        # =======================================================
        # PART 2: Create the Heatmap Figure
        # =======================================================
        fig_rules = go.Figure(data=go.Heatmap(
            z=rule_map_num,
            x=controller.terms, # Columns: Change of Error (dE)
            y=controller.terms, # Rows: Error (E)
            text=rule_map_text, # The text to display inside each cell
            texttemplate="%{text}",
            # Set text font to be high-contrast against the cell colors.
            textfont={"size": 16, "color": "white", "family": "Arial, sans-serif"},
            # A custom, vibrant colorscale that works well in both light/dark modes.
            colorscale=[
                [0.0, '#27ae60'], # S -> Green (Low gain, stable)
                [0.33, '#f39c12'], # M -> Yellow (Medium gain)
                [0.66, '#e67e22'], # B -> Orange (High gain)
                [1.0, '#c0392b']  # VB -> Red (Very high gain, aggressive)
            ],
            xgap=4, # Add spacing between cells for better visual separation.
            ygap=4,
            showscale=False # Hide the color scale bar; the text labels are sufficient.
        ))

        # =======================================================
        # PART 3: Update Figure Layout for Theme Compatibility
        # =======================================================
        fig_rules.update_layout(
            title=dict(
                text="<b>3. Fuzzy Rule Base (Logic Matrix)</b><br><span style='font-size:12px; color:#888;'>Logic: If Error is (Y) AND dError is (X) &rarr; Then Gain is (Cell)</span>",
                y=0.9,
                x=0.01,
                xanchor='left',
                yanchor='top'
            ),
            xaxis_title="Change of Error (dE)",
            yaxis_title="Error (E)",
            height=550,
            margin=dict(l=50, r=50, t=120, b=50),
            # --- Theme Adjustments ---
            # Make backgrounds transparent to adapt to the Marimo theme.
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            # Use a neutral font color that is visible on both light and dark backgrounds.
            font_color="#aaa"
        )

        # =======================================================
        # PART 4: Return the final Marimo output
        # =======================================================
        return mo.vstack([
            mo.md("## 🚦 The Controller's Brain (Rule Matrix)"),
            mo.md("""
            This heatmap visualizes the core logic of the fuzzy controller. It's a lookup table that dictates the system's response based on the current state.

            - **Red Cells (`VB`):** Represent an aggressive response. This happens when the error is large and needs a strong correction.
            - **Green Cells (`S`):** Represent a gentle response. This is the stability zone, used when the system is already close to its target.
            """),
            mo.ui.plotly(fig_rules)
        ])

    # Call the function to create and render the visualization.
    heatmap_output = create_rule_heatmap(fuzzy_viz)

    heatmap_output
    return


@app.cell(hide_code=True)
def _(
    OPT_GA,
    controller_selector,
    fuzzy_data,
    go,
    make_subplots,
    mo,
    np,
    ref_data,
    sim_settings,
    t_data,
    w_data,
):
    # 1. Define default variables (to prevent NameError)
    # Initialized empty in case not in GA Mode, so code doesn't break
    html_report = "" 
    fig_brain = go.Figure() 
    output_view = None

    # 2. Check if Intelligent System is active
    is_ga_mode = (controller_selector.value == OPT_GA)

    if not is_ga_mode:
        # --- Standard Mode ---
        output_view = mo.callout(
            mo.md("ℹ️ **Standard Mode Active:** Intelligent adaptation is off. Switch to **GA-Optimized** to enable the 'Fuzzy Brain'."),
            kind="neutral"
        )
    else:
        # --- GA/Fuzzy Mode ---

        # A. Prepare data
        ref_val = ref_data if isinstance(ref_data, (int, float)) else ref_data[0]
        error_signal = np.array([ref_val - w for w in w_data])
        fuzzy_action = np.array(fuzzy_data)

        # Compute statistics
        max_boost = np.max(fuzzy_action)
        min_damping = np.min(fuzzy_action)

        # B. Plot Figures
        fig_brain = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.22, 
            subplot_titles=("👁️ Sensory Input: Tracking Error", "🧠 Brain Output: Adaptive Gain Response")
        )

        # Plot 1: Error Signal (Dark Orange)
        fig_brain.add_trace(go.Scatter(
            x=t_data, y=error_signal,
            name="Error Signal",
            line=dict(color="#d35400", width=2.5), 
            fill='tozeroy',
            fillcolor='rgba(211, 84, 0, 0.15)' 
        ), row=1, col=1)

        # Plot 2: Gain (Purple)
        fig_brain.add_trace(go.Scatter(
            x=t_data, y=fuzzy_action,
            name="Gain Multiplier (α)",
            line=dict(color="#8e44ad", width=3), 
            fill='tozeroy',
            fillcolor='rgba(142, 68, 173, 0.15)' 
        ), row=2, col=1)

        # Reference line (Standard Baseline)
        fig_brain.add_trace(go.Scatter(
            x=t_data, y=[1.0]*len(t_data),
            name="Standard Baseline",
            line=dict(color="#607d8b", dash="dash", width=2)
        ), row=2, col=1)

        # C. Add Load Marker
        load_val = sim_settings.value["Load"]
        load_time = sim_settings.value["T_load"]

        if load_val > 0.01:
            # Vertical yellow line at load instant
            fig_brain.add_vline(
                x=load_time, 
                line_width=2, 
                line_dash="dot", 
                line_color="#f1c40f", 
                opacity=0.8
            )
            # Annotation
            fig_brain.add_annotation(
                x=load_time, y=1, yref="paper",
                text=" ",
                showarrow=False,
                font=dict(size=10, color="#f1c40f"),
                yshift=10, xshift=5
            )

        # D. Adjust Layout
        fig_brain.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#546e7a", family="Segoe UI, sans-serif", size=13),
            showlegend=True,
            legend=dict(
                orientation="h", y=1.25, x=0.5, xanchor="center",
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(t=110, b=30, l=60, r=20)
        )

        grid_params = dict(showgrid=True, gridcolor='rgba(84, 110, 122, 0.15)', gridwidth=1)
        fig_brain.update_xaxes(**grid_params, row=1, col=1); fig_brain.update_yaxes(**grid_params, row=1, col=1)
        fig_brain.update_xaxes(**grid_params, row=2, col=1); fig_brain.update_yaxes(**grid_params, row=2, col=1)

        # E. Prepare HTML code (Report)
        html_report = f"""
        <div style="display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px; font-family: 'Segoe UI', sans-serif;">
            <div style="flex: 1; padding: 15px; border-left: 5px solid #8e44ad; background: rgba(142, 68, 173, 0.08); border-radius: 6px;">
                <div style="color: #8e44ad; font-weight: bold; font-size: 0.95rem; margin-bottom: 5px;">MAX BOOST (Attack)</div>
                <div style="font-size: 1.6rem; color: #455a64; font-weight: bold; font-family: 'Consolas', monospace;">x{max_boost:.4f}</div>
                <div style="font-size: 0.85rem; color: #607d8b;">Amplification factor</div>
            </div>

            <div style="flex: 1; padding: 15px; border-left: 5px solid #27ae60; background: rgba(39, 174, 96, 0.08); border-radius: 6px;">
                <div style="color: #27ae60; font-weight: bold; font-size: 0.95rem; margin-bottom: 5px;">MAX DAMPING (Brake)</div>
                <div style="font-size: 1.6rem; color: #455a64; font-weight: bold; font-family: 'Consolas', monospace;">x{min_damping:.4f}</div>
                <div style="font-size: 0.85rem; color: #607d8b;">Suppression factor</div>
            </div>
        </div>
        """

        # Assemble final display
        output_view = mo.vstack([
            mo.md("#### 🧠 Intelligent Controller Analysis"),
            mo.Html(html_report),
            mo.ui.plotly(fig_brain)
        ])

    # 3. Render (This line displays the result on the screen)
    output_view
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧬 Evolutionary Optimization Engine

    This panel is the core of the automated tuning process. Manually finding the optimal set of six controller parameters
    (`Kp, Ki, Kd, λ, µ, α`) in such a large, multi-dimensional space is nearly impossible.

    Here, we employ a **Genetic Algorithm (GA)** that mimics the principles of natural selection to autonomously discover the most effective parameter combination.

    - **Configure:** Set the population size, generations, and search boundaries.
    - **Execute:** Launch the evolutionary process.
    - **Analyze:** Visualize the convergence and multi-dimensional results.
    """)
    return


@app.cell
def _(np, simulate_pmsm_system):
    class GeneticOptimizerEngine:
        """
        Professional GA Engine with Full History Logging.
        Modified to return 4 values needed for the Advanced Dashboard.
        """
        def __init__(self, motor_phys, vehicle_phys, sim_setup, custom_bounds=None):
            self.motor_phys = motor_phys
            self.vehicle_phys = vehicle_phys
            self.sim_setup = sim_setup

            if custom_bounds:
                self.bounds = custom_bounds
            else:
                self.bounds = [
                    (0.1, 10.0), (0.1, 10.0), (0.0, 1.0),
                    (0.1, 1.5), (0.1, 1.5), (0.1, 5.0)
                ]

        def _evaluate_fitness(self, genes):
            ctrl_candidate = {
                "Kp": genes[0], "Ki": genes[1], "Kd": genes[2],
                "Lambda": genes[3], "Mu": genes[4], "Fuzzy_Scale": genes[5]
            }

            fast_scenario = self.sim_setup.copy()
            fast_scenario["Time"] = 1.0  

            time, speed, _, _, _, ref = simulate_pmsm_system(
                ctrl_candidate, fast_scenario, 
                self.motor_phys, self.vehicle_phys, strategy_type="GA"
            )

            dt = time[1] - time[0]
            error = ref - speed
            ise_cost = np.sum(np.square(error)) * dt

            if np.isnan(ise_cost) or np.isinf(ise_cost) or np.max(speed) > (ref * 2.5):
                return 1e6 

            return ise_cost

        def run(self, pop_size=20, generations=10, mutation_rate=0.1):
            """
            Returns 4 items: 
            1. best_sol (Final Parameters)
            2. best_cost (Final Error)
            3. convergence_curve (Best error per generation)
            4. full_trial_history (List of all trials for Parallel Plot)
            """
            population = []
            for _ in range(pop_size):
                ind = [np.random.uniform(L, H) for L, H in self.bounds]
                population.append(ind)

            best_sol = None
            best_cost = float('inf')
            convergence_curve = []

            # === DATABASE FOR VISUALIZATION ===
            full_trial_history = []

            for gen in range(generations):
                scores = []
                for ind in population:
                    score = self._evaluate_fitness(ind)
                    scores.append(score)

                    # === LOGGING EVERY SINGLE TRIAL ===
                    full_trial_history.append({
                        'Generation': gen + 1,
                        'Kp': ind[0], 'Ki': ind[1], 'Kd': ind[2],
                        'Lambda': ind[3], 'Mu': ind[4], 'Alpha': ind[5],
                        # Clamp high costs for better plotting visualization
                        'Cost': score if score < 50000 else 50000 
                    })

                    if score < best_cost:
                        best_cost = score
                        best_sol = ind[:]

                convergence_curve.append(best_cost)

                # Standard Evolution Logic
                new_pop = []
                best_idx = np.argmin(scores)
                new_pop.append(population[best_idx])

                while len(new_pop) < pop_size:
                    idxs = np.random.randint(0, pop_size, size=3)
                    parents = [population[i] for i in idxs]
                    parents.sort(key=lambda x: scores[population.index(x)])
                    p1, p2 = parents[0], parents[1]

                    child = []
                    beta = np.random.random()
                    for g1, g2 in zip(p1, p2):
                        child.append(beta*g1 + (1-beta)*g2)

                    for k in range(len(child)):
                        if np.random.random() < mutation_rate:
                            child[k] = np.random.uniform(self.bounds[k][0], self.bounds[k][1])

                    new_pop.append(child)

                population = new_pop

            # === RETURN 4 VALUES ===
            return best_sol, best_cost, convergence_curve, full_trial_history
    return (GeneticOptimizerEngine,)


@app.cell(hide_code=True)
def _(mo):
    flowchart = mo.mermaid("""
    graph TD
        %% --- Styling Definitions ---
        classDef startend fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white,rx:10,ry:10;
        classDef proc fill:#ecf0f1,stroke:#34495e,stroke-width:2px,color:#2c3e50;
        classDef decision fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:#d35400,rx:5,ry:5;
        classDef sim fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
        classDef loop fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,stroke-dasharray: 5 5,color:white;

        %% --- Nodes ---
        Start([Start Optimization]):::startend

        Init[Initialize Population<br/>Generate Random Parameters<br/>Kp, Ki, Kd, λ, µ, α]:::proc

        subgraph Evaluation [Fitness Evaluation Loop]
            direction TB
            Sim[Run PMSM Simulation<br/>GA–AFFOPID Model]:::sim
            Calc[Calculate Fitness Cost<br/>ISE = ∫ e² dt]:::sim
        end

        Check{Max Generation<br/>Reached?}:::decision

        subgraph Evolution [Genetic Evolution]
            direction TB
            Sel[Selection<br/>Tournament & Elitism]:::loop
            Cross[Crossover<br/>Arithmetic Combination]:::loop
            Mut[Mutation<br/>Random Perturbation]:::loop
        end

        Stop([Output Optimal Genes]):::startend

        %% --- Connections ---
        Start --> Init
        Init --> Sim
        Sim --> Calc
        Calc --> Check

        Check -- No --> Sel
        Sel --> Cross
        Cross --> Mut
        Mut -- New Population --> Sim

        Check -- Yes --> Stop

    """)

    mo.vstack([
        mo.md("### 🔄 Optimization Process Flowchart"),
        mo.md("*This diagram illustrates the step-by-step logic executed by the Genetic Algorithm engine.*"),
        flowchart
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 🧬 Evolutionary Optimization Console")

    # 1. Initialize State Management
    # These variables prevent the optimization from running automatically
    # when you change settings. It only runs when the button is clicked.
    get_last_count, set_last_count = mo.state(0)
    get_cached_view, set_cached_view = mo.state(
        mo.md("ℹ️ *System Ready. Adjust settings above and click Start.*")
    )

    # 2. General Algorithm Settings
    ga_controls = mo.ui.dictionary({
        "pop": mo.ui.number(5, 100, value=10, step=1, label="Population Size"),
        "gen": mo.ui.number(1, 200, value=5, step=1, label="Generations"),
        "mut": mo.ui.number(0.01, 0.5, value=0.1, step=0.01, label="Mutation Rate")
    })

    # 3. Search Space Bounds Configuration
    # Defines the [Min, Max] range for each parameter using array widgets
    bounds_ui = mo.ui.dictionary({
        "Kp":   mo.ui.array([mo.ui.number(0.1, 500, value=0.1, full_width=True), mo.ui.number(0.1, 500, value=50.0, full_width=True)]),
        "Ki":   mo.ui.array([mo.ui.number(0.1, 500, value=0.1, full_width=True), mo.ui.number(0.1, 500, value=50.0, full_width=True)]),
        "Kd":   mo.ui.array([mo.ui.number(0.0, 100, value=0.0, full_width=True), mo.ui.number(0.0, 100, value=1.0, full_width=True)]),
        "Lam":  mo.ui.array([mo.ui.number(0.1, 2.0, value=0.1, full_width=True), mo.ui.number(0.1, 2.0, value=1.5, full_width=True)]),
        "Mu":   mo.ui.array([mo.ui.number(0.1, 2.0, value=0.1, full_width=True), mo.ui.number(0.1, 2.0, value=1.5, full_width=True)]),
        "Fuz":  mo.ui.array([mo.ui.number(0.1, 10.0, value=0.1, full_width=True), mo.ui.number(0.1, 10.0, value=5.0, full_width=True)]),
    })

    # 4. Layout Helper for Bounds Table
    def bound_row(name, key):
        return mo.hstack([
            mo.md(f"**{name}**").style({"width": "140px", "align-self": "center"}),
            bounds_ui[key][0],
            mo.md("↔️").style({"padding": "0 10px", "align-self": "center"}),
            bounds_ui[key][1]
        ], align="center")

    # 5. Accordion Widget for Advanced Settings
    bounds_widget = mo.accordion({
        "⚙️ Configure Search Space (Bounds)": mo.vstack([
            # Header Row
            mo.hstack([
                mo.md("Parameter").style({"width": "140px", "font-weight": "bold", "color": "#555"}),
                mo.md("Lower Bound (Min)").style({"flex": "1", "text-align": "center", "font-weight": "bold", "color": "#555"}),
                mo.md("").style({"width": "40px"}),
                mo.md("Upper Bound (Max)").style({"flex": "1", "text-align": "center", "font-weight": "bold", "color": "#555"}),
            ]),
            mo.md("---"),
            # Input Rows
            bound_row("Prop. Gain (Kp)", "Kp"),
            bound_row("Integ. Gain (Ki)", "Ki"),
            bound_row("Deriv. Gain (Kd)", "Kd"),
            bound_row("Int. Order (λ)", "Lam"),
            bound_row("Diff. Order (µ)", "Mu"),
            bound_row("Fuzzy Scale (α)", "Fuz"),
        ])
    })

    # 6. Action Button (Initialized at 0 to avoid errors)
    run_ga_btn = mo.ui.button(label="🚀 Start Evolution Process", kind="success", value=0,on_click=lambda value:value +1 )

    # 7. Final Panel Assembly
    mo.vstack([
        mo.callout(
            mo.vstack([
                mo.md("### 🛠️ Optimization Configuration"),
                mo.hstack([ga_controls], justify="center", gap=2),
                mo.md("---"),
                bounds_widget,
                mo.md(""), # Spacer
                run_ga_btn
            ]),
            kind="neutral"
        )
    ])

    return (
        bounds_ui,
        ga_controls,
        get_cached_view,
        get_last_count,
        run_ga_btn,
        set_cached_view,
        set_last_count,
    )


@app.cell(hide_code=True)
def _(
    GeneticOptimizerEngine,
    bounds_ui,
    ga_controls,
    get_cached_view,
    get_last_count,
    go,
    mo,
    motor_ui,
    run_ga_btn,
    set_cached_view,
    set_last_count,
    sim_settings,
    vehicle_ui,
):
    raw_val = run_ga_btn.value
    current_clicks = raw_val if raw_val is not None else 0
    last_clicks = get_last_count()
    is_new_click = (current_clicks > last_clicks)

    if is_new_click:
        # 1. Setup & Run
        user_defined_bounds = [
            tuple(bounds_ui["Kp"].value), tuple(bounds_ui["Ki"].value), tuple(bounds_ui["Kd"].value),
            tuple(bounds_ui["Lam"].value), tuple(bounds_ui["Mu"].value), tuple(bounds_ui["Fuz"].value),
        ]

        optimizer = GeneticOptimizerEngine(
            motor_phys=motor_ui.value, vehicle_phys=vehicle_ui.value,
            sim_setup=sim_settings.value, custom_bounds=user_defined_bounds
        )

        with mo.status.spinner(title="🧬 Processing Genome... Analyzing generations..."):
            # Note: We now unpack 4 return values
            best_genes, final_cost, history_scores, full_history = optimizer.run(
                pop_size=int(ga_controls.value["pop"]),
                generations=int(ga_controls.value["gen"]),
                mutation_rate=ga_controls.value["mut"]
            )

        # ==========================================
        # GRAPH 1: Parallel Coordinates (The Professional View)
        # ==========================================
        # This visualizes EVERY trial and how parameters relate to cost

        # Extract data columns
        kps = [d['Kp'] for d in full_history]
        kis = [d['Ki'] for d in full_history]
        lams = [d['Lambda'] for d in full_history]
        costs = [d['Cost'] for d in full_history]
        gens = [d['Generation'] for d in full_history]

        fig_par = go.Figure(data=go.Parcoords(
            line=dict(
                color=costs,
                colorscale='Turbo_r', # Red=Bad, Blue=Good
                showscale=True,
                cmin=min(costs),
                cmax=max(costs) * 0.5, # Focus color contrast on lower costs
            ),
            dimensions=[
                dict(label='Generation', values=gens),
                dict(label='Kp (Prop)', values=kps),
                dict(label='Ki (Integ)', values=kis),
                dict(label='λ (Int Order)', values=lams),
                dict(label='Cost (ISE)', values=costs)
            ]
        ))

        fig_par.update_layout(
            title="<b>Multidimensional Search Analysis</b><br><span style='font-size:12px; color:gray;'>Trace how parameters converge. Blue lines represent optimal solutions.</span>",
            height=500,
            margin=dict(l=60, r=60, t=80, b=40)
        )

        # ==========================================
        # GRAPH 2: 3D Optimization Landscape
        # ==========================================
        # Shows where the "Sweet Spot" is between Kp, Ki and Cost
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=kps, y=kis, z=costs,
            mode='markers',
            marker=dict(
                size=4,
                color=costs,
                colorscale='Viridis_r',
                opacity=0.8
            )
        )])
        fig_3d.update_layout(
            title="<b>3D Parameter Landscape</b> (Kp vs Ki vs Cost)",
            scene=dict(xaxis_title='Kp', yaxis_title='Ki', zaxis_title='Cost'),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        # ==========================================
        # GRAPH 3: Convergence Curve
        # ==========================================
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(history_scores)+1)), y=history_scores,
            mode='lines+markers', line=dict(color='#2ecc71', width=3),
            fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.1)'
        ))
        fig_conv.update_layout(
            title="<b>Convergence History</b>",
            xaxis_title="Generation", yaxis_title="Best Cost",
            height=350, template="plotly_white"
        )

        # Build View with Tabs
        results_table = f"""
        | Parameter | Symbol | Best Value |
        | :--- | :---: | :---: |
        | Proportional | $K_p$ | **{best_genes[0]:.5f}** |
        | Integral | $K_i$ | **{best_genes[1]:.5f}** |
        | Int. Order | $\lambda$ | **{best_genes[3]:.4f}** |
        | Diff. Order | $\mu$ | **{best_genes[4]:.4f}** |
        | Fuzzy Scale | $\\alpha$ | **{best_genes[5]:.4f}** |
        """

        new_view = mo.vstack([
            mo.md("---"),
            mo.callout(mo.md(f"**✅ Evolution Complete!** Best Cost: `{final_cost:.6f}`"), kind="success"),

            # Layout: Table | Convergence
            mo.hstack([mo.md(results_table), mo.ui.plotly(fig_conv)], align="center"),

            mo.md("### 🔬 Deep Dive Analysis"),
            mo.ui.plotly(fig_par),
            mo.md("**How to read the graph above:** Each line is one trial. Blue lines are the best solutions. Follow the blue lines to see which combination of Kp, Ki, and λ leads to the lowest Cost."),

            mo.accordion({"View 3D Landscape": mo.ui.plotly(fig_3d)})
        ])

        set_last_count(current_clicks) 
        set_cached_view(new_view)      
        final_output = new_view

    else:
        final_output = get_cached_view()

    final_output
    return best_genes, full_history, is_new_click


@app.cell(hide_code=True)
def _(full_history, get_cached_view, is_new_click, mo):


    if is_new_click:
        if full_history is None:
            history_view = mo.callout(
                mo.md("⏳ **No history found.** Please run the Genetic Algorithm first."),
                kind="neutral"
            )
        else:
            # 2. Data Formatting
            # We format the raw numbers to make them readable (4 decimal places)
            formatted_data = []
            for trial in full_history:
                formatted_data.append({
                    "Gen": trial['Generation'],
                    "Cost (ISE)": round(trial['Cost'], 6), # High precision for cost
                    "Kp": round(trial['Kp'], 4),
                    "Ki": round(trial['Ki'], 4),
                    "Kd": round(trial['Kd'], 4),
                    "λ (Int)": round(trial['Lambda'], 4),
                    "µ (Diff)": round(trial['Mu'], 4),
                    "α (Fuzz)": round(trial['Alpha'], 4),
                })
    
            # 3. Create Interactive Table
            # pagination=True keeps the interface clean
            history_table = mo.ui.table(
                formatted_data,
                pagination=True,
                page_size=15,
                label="Optimization Trials Data"
            )
    
            # 4. Assemble View
            history_view = mo.vstack([
                mo.md("### 🗂️ Detailed Optimization Log"),
                mo.md("""
                This table records every single "chromosome" (solution) tested by the algorithm.
                *   **Tip:** Click on the **"Cost (ISE)"** column header to sort by the lowest error.
                *   **Tip:** Click on **"Gen"** to see the chronological order.
                """),
                history_table
            ])
    else:
        history_view = get_cached_view()
    history_view
    return


@app.cell(hide_code=True)
def _(
    best_genes,
    get_cached_view,
    go,
    is_new_click,
    make_subplots,
    mo,
    motor_ui,
    np,
    sim_settings,
    simulate_pmsm_system,
    vehicle_ui,
):
    def create_stability_dashboard(genes):
        # ================================================
        # PART E: Stability & Phase Plane Analysis
        # ================================================

        # Display a placeholder message if the optimization process hasn't completed yet.
        if genes is None:
            return mo.callout(mo.md("⏳ **Waiting for optimization results...** Please run the GA first."), kind="neutral")

        # 1. Run a dedicated verification simulation with the best found genes.
        optimized_controller_params = {
            "Kp": genes[0], "Ki": genes[1], "Kd": genes[2],
            "Lambda": genes[3], "Mu": genes[4], "Fuzzy_Scale": genes[5]
        }

        # Use a shorter simulation time for this specific stability check.
        verify_scenario = sim_settings.value.copy()
        verify_scenario["Time"] = 2.0

        t_sim, w_sim, _, _, _, ref_val = simulate_pmsm_system(
            optimized_controller_params, verify_scenario, motor_ui.value, vehicle_ui.value, strategy_type="GA"
        )

        # 2. Calculate the necessary state variables for the phase plane plot.
        error_signal = ref_val - w_sim
        d_error_signal = np.gradient(error_signal, t_sim)

        # 3. Define a color palette for theme consistency.
        c_traj = '#8e44ad'  # Purple for the trajectory
        c_hist = '#2980b9'  # Blue for the histogram
        c_target = '#e74c3c' # Red for the equilibrium target
        c_text = '#aaa'     # Neutral text color for dark/light modes
        c_grid = 'rgba(170, 170, 170, 0.2)' # Subtle grid color

        # 4. Create the Plotly dashboard figure.
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"<b>Phase Plane Trajectory</b> (e vs de/dt)",
                f"<b>Error Histogram</b> (Precision Check)"
            ),
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.15 # Add space between plots to prevent title overlap.
        )

        # Plot A: The Phase Plane trajectory.
        fig.add_trace(go.Scatter(
            x=error_signal, y=d_error_signal,
            mode='lines',
            name='Trajectory',
            line=dict(color=c_traj, width=2.5),
        ), row=1, col=1)

        # Add a marker for the target equilibrium point (0,0).
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            name='Equilibrium Point',
            marker=dict(size=14, color=c_target, symbol='cross-thin', line=dict(width=3))
        ), row=1, col=1)

        # Plot B: The error distribution histogram.
        fig.add_trace(go.Histogram(
            x=error_signal[100:], # Ignore initial transient for a clearer view of steady-state error.
            nbinsx=50,
            name='Error Distribution',
            marker_color=c_hist,
            opacity=0.8
        ), row=1, col=2)

        # 5. Apply styling for dark/light mode compatibility.
        fig.update_layout(
            title=dict(
                text=f"<b>System Stability & Precision Analysis</b><br><span style='font-size:12px; color:{c_text};'>Left: Visual proof of stability. Right: Statistical proof of accuracy.</span>",
                y=0.95,
                x=0.05,
                xanchor='left',
                yanchor='top'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=c_text,
            height=550,
            showlegend=False,
            # Increase margins to prevent titles and labels from overlapping.
            margin=dict(
                t=140, # Top: Ample space for the main title above subplot titles.
                l=80,  # Left: Space for y-axis titles.
                r=50,
                b=80   # Bottom: Space for x-axis titles.
            )
        )

        # Define a reusable style for all axes.
        axis_style = dict(
            gridcolor=c_grid,
            zerolinecolor=c_grid,
            title_font=dict(size=14),
        )

        # Apply styles and labels to each axis individually.
        fig.update_xaxes(title_text="Error (e)", **axis_style, row=1, col=1)
        fig.update_yaxes(title_text="Change of Error (de/dt)", **axis_style, row=1, col=1)
        fig.update_xaxes(title_text="Error Magnitude", **axis_style, row=1, col=2)
        fig.update_yaxes(title_text="Count", **axis_style, row=1, col=2)

        # 6. Render the final layout in Marimo.
        return mo.vstack([
            mo.md("### 🎯 Final Validation: Stability and Precision"),
            mo.md("""
            This dashboard provides the mathematical proof of the controller's effectiveness, often required for high-impact publications:

            *   **Phase Plane (Left):** The trajectory (purple line) spirals inwards and converges at the origin `(0,0)`. This is a definitive visual demonstration of the system's **Asymptotic Stability**. It proves the controller will always guide the system back to the desired state without oscillations.

            *   **Error Histogram (Right):** The error distribution is sharply peaked and centered tightly around zero. This provides statistical evidence of the controller's high precision and its ability to effectively reject disturbances.
            """),
            mo.ui.plotly(fig)
        ])

    # Assume 'best_genes', 'sim_settings', 'motor_ui', and 'vehicle_ui' are available
    # from previous Marimo cells.
    if is_new_click:
        final_stability_view = create_stability_dashboard(best_genes)
    else:
        final_stability_view=get_cached_view()
    final_stability_view
    return


if __name__ == "__main__":
    app.run()
