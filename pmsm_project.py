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
            "Kd": mo.ui.number(0.0, 500.0, 0.000001, 0.01, label=r"Derivative ($K_d$)", full_width=True),
            "Lambda": mo.ui.number(0.01, 5, 0.000001, 0.9, label=r"Int. Order ($\lambda$)", full_width=True),
            "Mu": mo.ui.number(0.01, 5, 0.000001, 0.8, label=r"Diff. Order ($\mu$)", full_width=True),
        }),
        OPT_GA: mo.ui.dictionary({
            # These sliders now primarily define the GA's search space bounds, not direct simulation values.
            "Kp": mo.ui.number(0.0, 500.0, 0.000001, 5.0, label=r"GA Kp (Search Space)", full_width=True),
            "Ki": mo.ui.number(0.0, 500.0, 0.000001, 10.0, label=r"GA Ki (Search Space)", full_width=True),
            "Kd": mo.ui.number(0.0, 500.0, 0.000001, 0.01, label=r"GA Kd (Search Space)", full_width=True),
            "Lambda": mo.ui.number(0.01, 5, 0.000001, 0.978, label=r"GA Lambda (Search Space)", full_width=True),
            "Mu": mo.ui.number(0.01, 5, 0.000001, 0.862, label=r"GA Mu (Search Space)", full_width=True),
            "Fuzzy_Scale": mo.ui.number(0.1, 10.0, 0.000001, 1.0, label=r"GA Fuzzy Scale (Search Space)", full_width=True),
        })
    }

    # --- NEW ADDITION ---
    # 3. Define the parameters for the Switch logic and the steady-state FOPID controller.
    # These will be displayed in the sidebar ONLY when the OPT_GA mode is active.
    switch_and_fopid_params = mo.ui.dictionary({
        "Kp_fixed": mo.ui.number(0.00001, 500.0, 0.000001, 50.0, label=r"Fixed Kp (Steady-State)", full_width=True),
        "Ki_fixed": mo.ui.number(0.00001, 500.0, 0.000001, 100.0, label=r"Fixed Ki (Steady-State)", full_width=True),
        # The Kd can be shared, so we don't need a separate fixed Kd unless specified.
        "Threshold": mo.ui.number(0.1, 100.0, 0.000001, 0.5, label=r"Switching Threshold |E|", full_width=True)
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
        "Load": mo.ui.number(0, 500, 0.0001, 50, label="Load Torque $T_L$ (N.m)", full_width=True),
        "T_load": mo.ui.number(0, 5, 0.0001, 0.1, label="Step Load Time (sec)", full_width=True),
        "Time": mo.ui.number(0.1, 5.0, 0.0001, 0.2, label="Total Time (sec)", full_width=True)
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


@app.cell(hide_code=True)
def _(go, make_subplots, mo, np):
    # --- 1. Parameters & Setup ---
    # This cell isolates and validates the core mathematical component of our FOPID controller:
    # the fractional-order operator, s^α. We test a specific order here.
    test_order = 0.5
    filter_instance = OustaloupFilter(test_order, freq_low=0.01, freq_high=100.0)

    # --- 2. Frequency Domain Analysis (Bode Plot) ---
    # We calculate the theoretical frequency response of the Oustaloup approximation.
    freqs = np.logspace(-3, 3, 500)
    magnitudes = []
    phases = []

    for w in freqs:
        s = 1j * w
        h_s = filter_instance.gain
        # Re-calculate the continuous-time transfer function for plotting
        wb, wh, N = 0.01, 100.0, 3 
        for k in range(-N, N + 1):
            w_z = wb * (wh/wb)**((k + N + 0.5*(1 - test_order))/(2*N + 1))
            w_p = wb * (wh/wb)**((k + N + 0.5*(1 + test_order))/(2*N + 1))
            h_s *= (s + w_z) / (s + w_p)
        magnitudes.append(20 * np.log10(np.abs(h_s)))
        phases.append(np.angle(h_s, deg=True))

    # --- 3. Time Domain Analysis (Step Response) ---
    # We simulate the discrete-time filter's response to a sudden input.
    time_sim = OustaloupFilter(test_order)
    t_vec = np.linspace(0, 10, 1000)
    input_step = np.ones_like(t_vec)
    output_response = [time_sim.compute(val) for val in input_step]

    # --- 4. Plotting & Visualization ---
    # Define a theme-friendly color palette
    c_mag = '#00d2ff'   # Cyan
    c_phase = '#e74c3c' # Red
    c_resp = '#54a0ff'  # Blue
    c_text = '#aaa'     # Neutral Gray
    c_grid = 'rgba(170, 170, 170, 0.2)'

    # Create the figure with a secondary y-axis for the Bode plot
    fig_filter = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"<b style='color:{c_mag}'>Bode Plot (Frequency Domain)</b>",
            f"<b style='color:{c_resp}'>Step Response (Time Domain)</b>"
        ),
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {}]]
    )

    # Plot 1a: Magnitude
    fig_filter.add_trace(go.Scatter(x=freqs, y=magnitudes, name="Magnitude (dB)", line=dict(color=c_mag, width=2.5)), row=1, col=1, secondary_y=False)

    # Plot 1b: Phase
    fig_filter.add_trace(go.Scatter(x=freqs, y=phases, name="Phase (Deg)", line=dict(color=c_phase, width=2.5)), row=1, col=1, secondary_y=True)

    # Plot 2a: Step Response
    fig_filter.add_trace(go.Scatter(x=t_vec, y=output_response, name=f"Response of s<sup>{test_order}</sup>", line=dict(color=c_resp, width=2.5)), row=1, col=2)

    # Plot 2b: Input Step Reference
    fig_filter.add_trace(go.Scatter(x=t_vec, y=input_step, name="Input Step", line=dict(color=c_text, dash='dash', width=1.5), opacity=0.7), row=1, col=2)

    # --- 5. Professional Styling ---
    base_axis_style = dict(showgrid=True, gridcolor=c_grid, zerolinecolor=c_grid, tickfont=dict(color=c_text))
    fig_filter.update_layout(
        title=dict(
            text=f"<b>Analysis of the Fractional Operator s<sup>{test_order}</sup> (Oustaloup Method)</b>",
            y=0.92, x=0.5, xanchor='center', yanchor='top'
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=c_text),
        hovermode="x unified",
        margin=dict(t=100, l=80, r=80, b=50),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        # Axis styling
        xaxis=dict(type="log", title="Frequency (rad/s)", **base_axis_style),
        yaxis=dict(title="Magnitude (dB)", title_font=dict(color=c_mag), **base_axis_style),
        yaxis2=dict(title="Phase (Deg)", title_font=dict(color=c_phase), range=[0, 90], showgrid=False, tickfont=dict(color=c_phase)),
        xaxis2=dict(title="Time (s)", **base_axis_style),
        yaxis3=dict(title="Amplitude", **base_axis_style)
    )

    # Add a horizontal line for the theoretical target phase
    theoretical_phase = test_order * 90
    fig_filter.add_hline(
        y=theoretical_phase, line_dash="dot", line_color=c_phase,
        annotation_text=f"Theoretical Target: {theoretical_phase}°",
        annotation_font=dict(color=c_phase),
        annotation_position="bottom right",
        row=1, col=1, secondary_y=True
    )

    # --- 6. Final Output: Explanation and Plot ---
    # This is the most important part: explaining the "why"
    mo.vstack([
        mo.md(
            """
            ### 📉 Verifying the Core Component: The Fractional Operator

            **Why is this analysis here?** Our main controller is a *Fractional Order* PID (FOPID), which uses the non-standard mathematical operators `s^λ` and `s^μ`. Before we can trust our complex controller, we must first prove that our implementation of this core component—the `OustaloupFilter` class—is mathematically correct.

            This cell isolates a single fractional operator, **s<sup>{test_order}</sup>**, and validates its behavior:

            1.  **Bode Plot (Left):** The key property of a fractional operator is its **constant phase shift**. For an order of `{test_order}`, the theoretical phase shift is `{test_order} * 90° = {theoretical_phase}°`. As you can see, the red line holds steady at this target across a wide frequency range, **validating our implementation**.

            2.  **Step Response (Right):** This plot visualizes the operator's unique behavior in the time domain. It is neither a perfect integrator (which would be a straight ramp) nor a simple derivative, but something in between. This demonstrates the unique "memory" property of fractional calculus.
            """
        ),
        mo.ui.plotly(fig_filter)
    ])
    return


@app.cell
def _(FuzzyGainTuner, OPT_FOPID, OPT_GA, np):
    def simulate_pmsm_system(ctrl_params, scenario, motor_phys, vehicle_phys, switch_params, strategy_type):
        """
        Simulates the PMSM-based EV powertrain with the corrected parameter unpacking.
        """
        # --- START OF CORRECTION ---
        # 1. Unpack all parameters SAFELY by name (key) instead of by order.

        # Scenario parameters
        t_end = scenario["Time"]
        w_target = scenario["Ref"]
        t_load = scenario["T_load"]
        load_val = scenario["Load"]

        # Motor physical parameters
        Rs = motor_phys["Rs"]
        P = motor_phys["P"]
        Psi_m = motor_phys["Psi"]
        J = motor_phys["J"]
        B = motor_phys["B"]
        # Ld and Lq are not used in this simplified model, but it's good practice to acknowledge them.

        # Vehicle physical parameters
        Mass = vehicle_phys["Mass"]
        Rw = vehicle_phys["Rw"]
        Gear = vehicle_phys["Gear"]
        Cd = vehicle_phys["Cd"]
        Area = vehicle_phys["Area"]
        Rho = vehicle_phys["Rho"]
        # Cr (Rolling Resistance) is used in the vehicle dynamics calculation.

        # Advanced controller parameters
        kp_steady = switch_params["Kp_fixed"]
        ki_steady = switch_params["Ki_fixed"]
        error_threshold = switch_params["Threshold"]

        # Base gains for the transient AFFOPID mode
        kp_base = ctrl_params.get("Kp", 0.0)
        ki_base = ctrl_params.get("Ki", 0.0)

        # General controller parameters
        kd = ctrl_params.get("Kd", 0.0)
        lam = ctrl_params.get("Lambda", 1.0)
        mu = ctrl_params.get("Mu", 1.0)
        # --- END OF CORRECTION ---

        # 2. Initialize Controller Components
        is_advanced_mode = (strategy_type == OPT_GA)
        if is_advanced_mode:
            fuzzy_tuner = FuzzyGainTuner()

        use_fractional = (strategy_type == OPT_FOPID) or (strategy_type == OPT_GA)
        frac_integrator = OustaloupFilter(-lam) if use_fractional else None
        frac_differentiator = OustaloupFilter(mu) if use_fractional else None

        # 3. Initialize Simulation State & Data Loggers
        dt = 0.0001
        steps = int(t_end / dt)
        time = np.linspace(0, t_end, steps)

        speed_arr, torque_arr, iq_arr = np.zeros(steps), np.zeros(steps), np.zeros(steps)
        kp_log, ki_log = np.zeros(steps), np.zeros(steps) 
        u1_log, u2_log = np.zeros(steps), np.zeros(steps)

        w_mech, iq_current, error_sum, prev_error = 0.0, 0.0, 0.0, 0.0

        # 4. Main Simulation Loop
        for i in range(steps):
            t = time[i]
            ref = w_target if t > 0.005 else 0.0
            error = ref - w_mech
            delta_error = (error - prev_error) / dt if dt > 0 else 0.0

            # Block A: Gain Calculation with Switching Logic
            final_kp, final_ki, final_kd = 0.0, 0.0, 0.0
            u1, u2 = 1.0, 1.0

            if is_advanced_mode and abs(error) > error_threshold:
                # Transient State: Use Adaptive Fuzzy Control
                u1, u2 = fuzzy_tuner.compute_tuning_factors(error, delta_error)
                final_kp = kp_base * u1
                final_ki = ki_base * u2
                final_kd = kd
            else:
                # Steady-State: Use Fixed Gains
                if is_advanced_mode:
                    final_kp = kp_steady
                    final_ki = ki_steady
                    final_kd = 0
                else:
                    final_kp = ctrl_params["Kp"]
                    final_ki = ctrl_params["Ki"]
                    final_kd = ctrl_params["Kd"]

            # Log gains and fuzzy outputs
            kp_log[i], ki_log[i] = final_kp, final_ki
            u1_log[i], u2_log[i] = u1, u2

            # Block B: Core Controller Action
            p_term = final_kp * error
            if use_fractional:
                i_term = final_ki * frac_integrator.compute(error)
                d_term = final_kd * frac_differentiator.compute(error)
            else:
                error_sum += error * dt
                i_term = final_ki * error_sum
                d_term = final_kd * delta_error

            iq_ref = p_term + i_term + d_term

            # Saturation and Anti-windup
            if iq_ref > 200: iq_ref = 200; error_sum -= error * dt
            elif iq_ref < -200: iq_ref = -200; error_sum -= error * dt

            # Block C: Plant Model (Physics)
            iq_current += (iq_ref - iq_current) * (dt / 0.001)
            Te = 1.5 * P * Psi_m * iq_current
            T_ext = load_val if t >= t_load else 0.0
            v = (w_mech / Gear) * Rw
            F_drag = 0.5 * Rho * Cd * Area * v * abs(v)
            T_drag = (F_drag * Rw) / Gear
            dw_dt = (Te - T_ext - T_drag - B * w_mech) / J
            w_mech += dw_dt * dt
            if w_mech < 0: w_mech = 0

            # Log system state variables
            speed_arr[i], torque_arr[i], iq_arr[i] = w_mech, Te, iq_current
            prev_error = error

        return time, speed_arr, torque_arr, iq_arr, kp_log, ki_log, u1_log, u2_log, ref
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
    # 1. Run the simulation and unpack all the returned values.
    t_data, w_data, te_data, iq_data, kp_log, ki_log, u1_log, u2_log, ref_data = simulate_pmsm_system(
        active_params.value,
        sim_settings.value,
        motor_ui.value,
        vehicle_ui.value,
        switch_and_fopid_params.value, # <-- Pass the .value of the imported dictionary
        controller_selector.value
    )

    # 2. Define the plotting function.
    def generate_dashboard_plot(t_arr, w_arr, te_arr, iq_arr, ref_val, load_config):
        """
        Creates the main 3-panel performance plot.
        """
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("🚀 Speed Response", "⚙️ Electromagnetic Torque", "⚡ q-axis Current (Iq)")
        )

        # Speed Trace
        fig.add_trace(go.Scatter(x=t_arr, y=w_arr, name="Speed", line=dict(color="#00d2ff", width=3)), row=1, col=1)

        # Reference Line
        ref_value_corrected = ref_val if isinstance(ref_val, (int, float)) else ref_val
        ref_line = [ref_value_corrected] * len(t_arr)
        fig.add_trace(go.Scatter(x=t_arr, y=ref_line, name="Target", line=dict(dash="dash", color="#e74c3c", width=2)), row=1, col=1)

        # Torque Trace
        fig.add_trace(go.Scatter(x=t_arr, y=te_arr, name="Torque", line=dict(color="#2ecc71", width=2.5)), row=2, col=1)

        # Current Trace
        fig.add_trace(go.Scatter(x=t_arr, y=iq_arr, name="Current", line=dict(color="#f1c40f", width=2.5)), row=3, col=1)

        # Load Disturbance Marker
        if load_config["Load"] > 0.01:
            fig.add_vline(x=load_config["T_load"], line_width=2, line_dash="dot", line_color="#8e44ad", opacity=0.8)

        # General Styling
        fig.update_layout(
            height=750, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#aaa"), 
            showlegend=False, 
            margin=dict(t=80, b=30, l=60, r=20),
            hovermode='x unified'
        )
        grid_style = dict(showgrid=True, gridcolor='rgba(170, 170, 170, 0.2)')
        fig.update_xaxes(**grid_style)
        fig.update_yaxes(**grid_style)

        return fig

    # 3. Call the plotting function and render the final output.
    final_dashboard_fig = generate_dashboard_plot(t_data, w_data, te_data, iq_data, ref_data, sim_settings.value)

    mo.vstack([
        mo.md("### 🏎️ Vehicle Performance Simulation"),
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
    mo.md("""
    ## 🧩 Fuzzy Logic: Visualized

    Understanding the non-linear behavior of a fuzzy controller is best done visually.
    This panel breaks down the controller's decision-making process into its fundamental components, showing how it translates numerical data into intelligent, human-like actions.

    ---

    ### The Inputs: System State (`e` and `de/dt`)

    The fuzzy controller constantly monitors two critical pieces of information about the system's performance:

    1.  **Error (`e`)**: This is the difference between the target speed and the current speed (`e = ω* - ω`). It answers the question: *"How far are we from our goal right now?"* A large positive error means the motor is too slow; a large negative error means it's too fast.

    2.  **Change in Error (`de/dt`)**: This is the rate at which the error is changing. It answers the question: *"Are we getting better or worse, and how quickly?"* A positive `de/dt` means the error is growing (getting worse), while a negative `de/dt` means the error is shrinking (getting better).

    ### The Visualization Components:

    *   **Membership Functions:** This plot shows how the controller "fuzzifies" the crisp numerical values of `e` and `de/dt` into linguistic, understandable terms like "Negative Big" (NB), "Zero" (Z), or "Positive Small" (PS). It allows a single input value to belong to multiple categories with varying degrees of membership.

    *   **Rule Matrix (Heatmap):** This is the "brain" of the controller. It's a simple lookup table containing rules written by an expert, such as: **IF** the `Error` is "Positive Big" **AND** the `Change in Error` is "Zero," **THEN** the output action should be "Very Big" (VB). The heatmap visualizes this entire set of rules.

    *   **Control Surface (3D Plot):** This is the complete decision map. It shows the final output for every possible combination of `e` and `de/dt`. The smooth, curved surface highlights the controller's non-linear nature, allowing for nuanced responses that a traditional linear controller cannot achieve.

    ### The Output: Adaptive Gain Multiplier (`α`)

    After processing the inputs through the rules, the controller produces a single, crisp numerical output:

    *   **Adaptive Gain (`α`)**: This is a multiplier that dynamically scales the gains (`Kp`, `Ki`, `Kd`) of the main FOPID controller.
        *   If `α > 1.0`, the controller becomes more aggressive to quickly correct a large error.
        *   If `α < 1.0`, the controller becomes gentler to avoid overshoot when near the target.
        *   If `α = 1.0`, the controller behaves like a standard, non-adaptive FOPID.

    This adaptive gain is what allows the system to be both aggressive when needed and stable when it matters most.
    """)
    return


@app.cell
def _(np):
    class BaseFuzzyController:
        def __init__(self):
            self.terms = ['NB', 'NS', 'Z', 'PS', 'PB']
            self.centers = {'NB': -1.0, 'NS': -0.5, 'Z': 0.0, 'PS': 0.5, 'PB': 1.0}

        def triangle_mf(self, x, center, width=0.5):
            """Calculates the membership degree for a triangular function."""
            return max(0, 1 - abs(x - center) / width)

        def _compute_single_output(self, error, delta_error, rule_matrix, output_values):
            """A generic, reusable function to compute one fuzzy output."""
            # Normalize inputs to the range [-1, 1]
            e_norm = np.clip(error / 250.0, -1, 1)
            de_norm = np.clip(delta_error / 150.0, -1, 1)

            numerator = 0.0
            denominator = 0.0

            for i, e_term in enumerate(self.terms):
                mu_e = self.triangle_mf(e_norm, self.centers[e_term])
                if mu_e == 0: continue

                for j, de_term in enumerate(self.terms):
                    mu_de = self.triangle_mf(de_norm, self.centers[de_term])
                    if mu_de == 0: continue

                    firing_strength = min(mu_e, mu_de)
                    if firing_strength > 0:
                        output_term = rule_matrix[i][j]
                        output_val = output_values[output_term]
                        numerator += firing_strength * output_val
                        denominator += firing_strength

            if denominator == 0:
                # Return a neutral default value (e.g., Medium or Zero) if no rules fire
                return output_values.get('M', output_values.get('Z', 1.0))

            return numerator / denominator

    # --- THIS IS THE NEW, PAPER-ALIGNED FUZZY CONTROLLER ---
    # It generates TWO independent scaling factors, U1 and U2.
    class FuzzyGainTuner(BaseFuzzyController):
        def __init__(self):
            super().__init__()
            # Output values represent scaling factors (unitless, numbers around 1.0)
            self.u_outputs = {'S': 0.8, 'M': 1.0, 'B': 1.2, 'VB': 1.5}

            # Rule Base for U1 (for Kp scaling), designed for a fast transient response.
            # Aggressive when error is large.
            self.u1_rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['VB', 'B',  'B',  'M',  'S'],  # e: NB
                ['B',  'B',  'M',  'S',  'S'],  # e: NS
                ['M',  'M',  'S',  'M',  'M'],  # e: Z
                ['S',  'S',  'M',  'B',  'B'],  # e: PS
                ['S',  'M',  'B',  'B',  'VB']  # e: PB
            ]

            # Rule Base for U2 (for Ki scaling), designed to reduce overshoot.
            # Gentle (reduces integral action) when error is shrinking towards zero.
            self.u2_rule_matrix = [
                # de:  NB,   NS,   Z,   PS,   PB
                ['S',  'S',  'M',  'B',  'VB'], # e: NB
                ['S',  'S',  'M',  'B',  'B'],  # e: NS
                ['M',  'M',  'S',  'M',  'M'],  # e: Z
                ['B',  'B',  'M',  'S',  'S'],  # e: PS
                ['VB', 'B',  'M',  'S',  'S']   # e: PB
            ]

        def compute_tuning_factors(self, error, delta_error):
            """
            Computes and returns two independent tuning factors, U1 (for Kp) and U2 (for Ki).
            """
            u1 = self._compute_single_output(error, delta_error, self.u1_rule_matrix, self.u_outputs)
            u2 = self._compute_single_output(error, delta_error, self.u2_rule_matrix, self.u_outputs)
            return u1, u2
    return (FuzzyGainTuner,)


@app.cell
def _(FuzzyGainTuner, mo):
    OPT_U1 = "U1 Tuning Factor (for Kp)"
    OPT_U2 = "U2 Tuning Factor (for Ki)"

    # --- UI ELEMENT FOR SELECTION ---
    # Create the dropdown menu to select which output to visualize.
    fuzzy_viz_selector = mo.ui.dropdown(
        options=[OPT_U1, OPT_U2],
        value=OPT_U1,
        label="Select Control Surface to Visualize:"
    )

    # We only need one instance of our new, unified fuzzy controller.
    fuzzy_tuner_viz_instance = FuzzyGainTuner()

    # The final output of this cell is the user interface.
    # The selected value and the controller instance will be passed to the next cells.
    mo.vstack([
        mo.md("## 🧩 Fuzzy Controller Internals"),
        mo.md(
            """
            This dashboard visualizes the internal logic of the dual-output fuzzy controller.

            1.  **Fuzzification:** The top plot shows the shared membership functions for the `Error` and `Change in Error` inputs.
            2.  **Control Surface:** The bottom 3D plot shows the decision-making map. Use the dropdown to switch between visualizing the two independent outputs:
                - **`U1`:** The scaling factor for `Kp`, designed for a fast response.
                - **`U2`:** The scaling factor for `Ki`, designed to control overshoot.
            """
        ),
        fuzzy_viz_selector # Display the dropdown menu
    ])
    return OPT_U1, fuzzy_tuner_viz_instance, fuzzy_viz_selector


@app.cell
def _(OPT_U1, fuzzy_tuner_viz_instance, fuzzy_viz_selector, go, mo, np):
    selected_surface = fuzzy_viz_selector.value

    # =======================================================
    # PART 1: Fuzzification Plot (Membership Functions)
    # This plot is static and doesn't depend on the dropdown.
    # =======================================================
    x_range = np.linspace(-1.5, 1.5, 300)
    fig_mf = go.Figure()
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    # Use the single fuzzy_tuner_viz_instance
    for idx, term in enumerate(fuzzy_tuner_viz_instance.terms):
        y_values = [fuzzy_tuner_viz_instance.triangle_mf(x, fuzzy_tuner_viz_instance.centers[term]) for x in x_range]
        fig_mf.add_trace(go.Scatter(
            x=x_range, y=y_values, name=f"Term: {term}", fill='tozeroy',
            line=dict(color=colors[idx], width=2.5), opacity=0.7
        ))

    fig_mf.update_layout(
        title=dict(text="<b>1. Fuzzification Stage (Shared Membership Functions)</b>"),
        xaxis_title="Normalized Input (e.g., Error)", yaxis_title="Degree of Membership (μ)",
        height=400, margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#aaa"
    )

    # =================================================================
    # PART 2: Dynamic Control Surface Visualization
    # This part is reactive to the dropdown selection.
    # =================================================================
    res = 40
    e_vec = np.linspace(-250, 250, res)
    de_vec = np.linspace(-150, 150, res) # Expanded range for de/dt
    z_surface = np.zeros((res, res))

    # Get the correct rule matrix and output values based on the selection
    if selected_surface == OPT_U1:
        rule_matrix = fuzzy_tuner_viz_instance.u1_rule_matrix
        output_values = fuzzy_tuner_viz_instance.u_outputs
        z_axis_title = "U1 Tuning Factor (for Kp)"
    else: # OPT_U2
        rule_matrix = fuzzy_tuner_viz_instance.u2_rule_matrix
        output_values = fuzzy_tuner_viz_instance.u_outputs
        z_axis_title = "U2 Tuning Factor (for Ki)"

    # Calculate the z-surface for the plot
    for i in range(res):
        for j in range(res):
            # Use the generic, internal compute function from the BaseFuzzyController
            z_surface[j][i] = fuzzy_tuner_viz_instance._compute_single_output(
                e_vec[i], de_vec[j], rule_matrix, output_values
            )

    fig_surf = go.Figure(data=[go.Surface(
        z=z_surface, x=e_vec, y=de_vec, colorscale='Plasma',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="#00FFFF", project_z=True),
        opacity=0.9
    )])

    fig_surf.update_layout(
        title=dict(text=f"<b>2. Control Surface for: {z_axis_title}</b>"),
        scene=dict(
            xaxis_title='Error (e)', yaxis_title='Change of Error (de/dt)', zaxis_title=z_axis_title,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.4))
        ),
        height=600, margin=dict(l=10, r=10, t=80, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font_color="#aaa"
    )

    # ==========================================
    # PART 3: Render Final Plots in Marimo
    # ==========================================
    mo.vstack([
        mo.ui.plotly(fig_mf),
    ])
    return (fig_surf,)


@app.cell
def _(fig_surf, mo):
    mo.ui.plotly(fig_surf),
    return


@app.cell
def _(OPT_U1, fuzzy_tuner_viz_instance, fuzzy_viz_selector, go, mo):
    selected_heatmap = fuzzy_viz_selector.value

    # --- Step 1: Select the Correct Rule Matrix and Description ---
    # The logic is now much simpler.
    if selected_heatmap == OPT_U1:
        matrix_to_display = fuzzy_tuner_viz_instance.u1_rule_matrix
        title_text_1 = "Rule Matrix for U1 (Kp Scaling)"
        description_text_1 = """
        This heatmap visualizes the logic for the **U1** scaling factor, which adjusts `Kp`. 
        Notice the aggressive response (`VB`, `B`) when the error is large. This is designed to make the controller react quickly to large disturbances.
        """
    else: # OPT_U2
        matrix_to_display = fuzzy_tuner_viz_instance.u2_rule_matrix
        title_text_1 = "Rule Matrix for U2 (Ki Scaling)"
        description_text_1 = """
        This heatmap visualizes the logic for the **U2** scaling factor, which adjusts `Ki`. 
        This logic is often designed to be more conservative to prevent overshoot, reducing the integral action (`S`) when the system is approaching the target.
        """

    # =======================================================
    # PART 2: Prepare Data for the Heatmap
    # =======================================================
    # Define the mapping from linguistic terms to numerical values for coloring.
    rule_map_to_num = {'VB': 4, 'B': 3, 'M': 2, 'S': 1}

    # Convert the selected text matrix to a numerical matrix for the z-axis.
    numerical_matrix = [
        [rule_map_to_num.get(rule, 0) for rule in row] 
        for row in matrix_to_display
    ]

    # =======================================================
    # PART 3: Create the Heatmap Figure
    # =======================================================
    fig_rules = go.Figure(data=go.Heatmap(
        z=numerical_matrix,
        x=fuzzy_tuner_viz_instance.terms,
        y=fuzzy_tuner_viz_instance.terms,
        text=matrix_to_display, # Display the original text labels ('VB', 'B', etc.)
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[
            [0.25*i, color] for i, color in enumerate(['#27ae60', '#f39c12', '#e67e22', '#c0392b'])
        ],
        xgap=4, ygap=4, showscale=False
    ))

    # =======================================================
    # PART 4: Update Figure Layout
    # =======================================================
    fig_rules.update_layout(
        xaxis_title="Change of Error (de/dt)",
        yaxis_title="Error (e)",
        height=550,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#aaa"
    )

    # =======================================================
    # PART 5: Render the final Marimo output with dynamic content
    # =======================================================
    mo.vstack([
        mo.md(f"## 🚦 The Controller's Brain: {title_text_1}"),
        mo.md(description_text_1),
        mo.ui.plotly(fig_rules)
    ])
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
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):

    flowchart = mo.mermaid("""
    graph TD
        %% --- Styling Definitions ---
        classDef startend fill:#27ae60,stroke:#2c3e50,stroke-width:2px,color:white,font-weight:bold,rx:8,ry:8;
        classDef process fill:#ecf0f1,stroke:#34495e,stroke-width:2px,color:#2c3e50;
        classDef decision fill:#f39c12,stroke:#d35400,stroke-width:2px,color:#2c3e50,font-weight:bold;
        classDef io fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
        classDef loop fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:white,stroke-dasharray: 5 5;

        %% --- Diagram Flow ---
        A(Start GA Process):::startend
        B[Initialize Population\nCreate random sets of controller parameters - Chromosomes]:::process
        C{Loop for Each Generation}:::decision

        subgraph "Evolutionary Cycle"
            direction TB
            D[Evaluate Fitness\nRun simulation for each individual\nCalculate ISE cost score]:::io
            E[Selection\nChoose the best individuals - Parents\nusing Elitism & Tournament]:::loop
            F[Crossover\nCombine genes from two parents\nto create a new child]:::loop
            G[Mutation\nIntroduce small, random changes\nto the child's genes]:::loop
        end

        H{Generations Complete?}:::decision
        I(End: Output Best Solution\nThe single best chromosome found):::startend

        %% --- Connections ---
        A --> B
        B --> C
        C -- "Start New Generation" --> D
        D --> E
        E --> F
        F --> G
        G -- "New Population" --> H
        H -- "No, continue..." --> C
        H -- "Yes" --> I

    """)
    main = f"""
    ## 🧬 Evolutionary Optimization Engine
    # The Optimization Engine: Genetic Algorithm (GA)

    The core of this project's intelligence lies in its ability to automatically find the optimal tuning parameters for the controller. Manually tuning six interdependent parameters (`Kp`, `Ki`, `Kd`, `λ`, `μ`, `α`) is a nearly impossible task for a human. It's like trying to solve a six-dimensional puzzle.

    This is where the **Genetic Algorithm (GA)** comes in.

    {flowchart}

    ## The Core Idea: Digital Evolution

    Instead of manual trial-and-error, we use a process that mimics Charles Darwin's theory of evolution and "survival of the fittest." We create a large population of potential controllers and let them compete against each other over many generations. Only the best ones survive and "reproduce," leading to progressively better solutions.

    Here’s how this "digital evolution" works, step-by-step:

    ### 1. Initialization: The First Generation

    First, we create an initial **Population**. In our case, a "population" is a group of candidate controllers (e.g., 50 of them).
    *   Each **Individual** in the population is a complete set of the six controller parameters. This is often called a **Chromosome**.
    *   Each parameter (`Kp`, `Ki`, etc.) within that set is a **Gene**.

    This first generation is created completely randomly within the search boundaries you define in the "Optimization Console."

    ### 2. Fitness Evaluation: The "Test Drive"

    This is the most critical step. To determine which controllers are "good," we need to test them.
    *   For every single individual (chromosome) in the population, we run a quick, standardized simulation.
    *   We then calculate its performance using a **Cost Function**. In this project, we use the **Integral of Squared Error (ISE)**.
        *   `ISE = ∫ e² dt`
    *   A lower ISE score means the controller tracked the target speed more accurately. This score is the individual's **Fitness**. A lower score means higher fitness (better performance).

    ### 3. Selection: Choosing the "Parents"

    Now that every controller has a fitness score, we decide who gets to "reproduce." We use two main strategies:
    *   **Elitism:** The single best individual from the current generation is automatically guaranteed to pass on to the next generation, untouched. This ensures we never lose our best solution.
    *   **Tournament Selection:** We randomly pick a few individuals (e.g., 3) from the population and make them "compete." The one with the best fitness score (lowest error) wins and is selected as a "parent" for the next generation. We repeat this process until we have enough parents.

    ### 4. Crossover: Creating "Children"

    This step mimics biological reproduction. We take two "parent" controllers that were chosen during Selection and combine their "genes" to create a new "child" controller.

    For example, a new `Kp` value for the child might be a blend of the `Kp` values from its two parents. This allows the new generation to inherit the successful traits of the previous one.

    ### 5. Mutation: Introducing New Ideas

    To avoid getting stuck with similar solutions, we introduce random changes. With a small probability (the **Mutation Rate**), we might randomly tweak one of the "genes" of a new child. For instance, we might slightly increase its `Kd` value.

    This step is crucial for exploring new, potentially better solutions that couldn't be created through crossover alone.

    ### 6. Repeat!

    The new generation of "children" (created from Crossover and Mutation) now becomes the current population. We then repeat the entire process—**Fitness Evaluation, Selection, Crossover, Mutation**—for the number of generations you specified.

    By the end of this process, the algorithm returns the "super-survivor": the single best set of parameters found across all generations. This is the solution that consistently produced the lowest error and represents the optimized controller.

    """
    main=mo.md(main)
    main

    # The cell ends here
    return


@app.cell
def _(OPT_GA, np, simulate_pmsm_system):
    class GeneticOptimizerEngine:
        """
        A Genetic Algorithm (GA) engine designed to find the optimal controller parameters
        for the transient response of the AFFOPID system.
        """
        def __init__(self, motor_phys, vehicle_phys, sim_setup, switch_params, custom_bounds=None):
            self.motor_phys = motor_phys
            self.vehicle_phys = vehicle_phys
            self.sim_setup = sim_setup
            self.switch_params = switch_params
            self.bounds = custom_bounds or [
                (0.1, 150.0), (0.1, 300.0), (0.0, 1.0),
                (0.1, 1.5), (0.1, 1.5), (0.1, 5.0)
            ]

        def _evaluate_fitness(self, individual_genes):
            """
            Calculates the fitness of a single set of controller parameters.
            """
            # The GA optimizes the BASE gains for the transient controller
            controller_candidate = {
                "Kp_base": individual_genes[0], 
                "Ki_base": individual_genes[1], 
                "Kd": individual_genes[2],
                "Lambda": individual_genes[3], 
                "Mu": individual_genes[4]
                # The 6th gene is not used here as it's part of the fuzzy logic itself
            }

            # Use a short simulation time for fast evaluation
            fast_scenario = self.sim_setup.copy()
            fast_scenario["Time"] = 0.5

            # --- THIS IS THE CORRECTED LINE ---
            # Unpack all 9 return values, using placeholders (_) for ones we don't need for fitness calculation.
            time, speed, _, _, _, _, _, _, ref = simulate_pmsm_system(
                controller_candidate, fast_scenario, self.motor_phys, self.vehicle_phys, 
                self.switch_params, strategy_type=OPT_GA
            )
            # --- END OF CORRECTION ---

            dt = time[1] - time[0]
            ref_val = ref if isinstance(ref, (int, float)) else ref
            error = ref_val - speed
            ise_cost = np.sum(np.square(error)) * dt

            is_unstable = np.isnan(ise_cost) or np.isinf(ise_cost) or np.max(speed) > (ref_val * 2.5)
            if is_unstable:
                return 1e6 

            return ise_cost

        def run(self, pop_size=20, generations=10, mutation_rate=0.1, crossover_rate=0.7):
            """
            Executes the main Genetic Algorithm loop.
            """
            population = [[np.random.uniform(L, H) for L, H in self.bounds] for _ in range(pop_size)]
            best_solution = None
            best_fitness = float('inf')

            convergence_history = []
            full_trial_history = []

            for gen in range(generations):
                fitness_scores = [self._evaluate_fitness(individual) for individual in population]

                for i, individual in enumerate(population):
                    full_trial_history.append({
                        'Generation': gen + 1, 'Kp_base': individual[0], 'Ki_base': individual[1], 'Kd': individual[2],
                        'Lambda': individual[3], 'Mu': individual[4],
                        'Cost': fitness_scores[i]
                    })

                min_fitness_in_gen = min(fitness_scores)
                if min_fitness_in_gen < best_fitness:
                    best_fitness = min_fitness_in_gen
                    best_solution = population[fitness_scores.index(min_fitness_in_gen)]

                convergence_history.append(best_fitness)

                # Evolution: Create the next generation
                new_population = [best_solution] # Elitism
                while len(new_population) < pop_size:
                    # Parent Selection (Tournament)
                    p1_idx, p2_idx = np.random.choice(range(pop_size), 2, replace=False)
                    parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p2_idx]
                    p3_idx, p4_idx = np.random.choice(range(pop_size), 2, replace=False)
                    parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p4_idx]

                    # Crossover & Mutation
                    child = parent1[:]
                    if np.random.random() < crossover_rate:
                        alpha = np.random.random()
                        child = [p1 * alpha + p2 * (1 - alpha) for p1, p2 in zip(parent1, parent2)]
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
        "Kp":   {"label": "Prop. Gain (Kp)",   "min": 0.1, "max": 500.0, "start_min": 0.1,  "start_max": 300.0},
        "Ki":   {"label": "Integ. Gain (Ki)",   "min": 0.1, "max": 500.0, "start_min": 0.1,  "start_max": 300.0},
        "Kd":   {"label": "Deriv. Gain (Kd)",   "min": 0.0, "max": 300.0, "start_min": 0.0,  "start_max": 2.0},
        "Lam":  {"label": "Int. Order (λ)",     "min": 0.1, "max": 2.0,   "start_min": 0.8,  "start_max": 1.2},
        "Mu":   {"label": "Diff. Order (µ)",    "min": 0.1, "max": 2.0,   "start_min": 0.8,  "start_max": 1.2},
        "Fuz":  {"label": "Fuzzy Scale (α)",    "min": 0.1, "max": 10.0,  "start_min": 0.5,  "start_max": 5.0}
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
def _(get_ga_full_history, mo):
    # Cell 1: Data Preparation and Filter UI

    # Check if the GA has been run yet.
    if 'get_ga_full_history' not in globals() or not get_ga_full_history():
        # Display a placeholder if no data is available.
        ga_data_status = mo.md("⏳ *Run the Genetic Algorithm to generate results for analysis.*")
        # Define placeholder variables to prevent errors in downstream cells.
        history_df = None
        ga_result_filter_slider = None
    else:
        # If data exists, prepare it and create the UI.
        import pandas as pd

        # Create the main DataFrame from the full history and sort it by performance.
        history_df = pd.DataFrame(get_ga_full_history()).sort_values(by='Cost').reset_index(drop=True)

        # Create the interactive slider for filtering results.
        ga_result_filter_slider = mo.ui.range_slider(
            0, 100, value=[0, 80], step=1,full_width=True,
            label="Filter Top Performers (%):"
        )

        # Display the slider and a title.
        ga_data_status = mo.vstack([
            mo.md("### 🔬 Interactive Analysis of GA Trials"),
            mo.md("Use the slider below to focus on a specific percentile of the best-performing solutions found by the algorithm. The plots below will update reactively."),
            ga_result_filter_slider
        ])

    # This cell's output is the UI. The dataframes are passed to other cells.
    ga_data_status
    return ga_result_filter_slider, history_df


@app.cell
def _(ga_result_filter_slider, history_df):
    # Cell 2: Filtering Logic (No Visual Output)
    # This cell simply takes the full dataframe and the slider value,
    # and returns the filtered dataframe for other cells to use.

    if history_df is not None and ga_result_filter_slider is not None:
        min_percent, max_percent = ga_result_filter_slider.value
        start_idx = int(len(history_df) * (min_percent / 100.0))
        # Ensure end_idx is at least start_idx + 1 to avoid empty slices
        end_idx = max(start_idx + 1, int(len(history_df) * (max_percent / 100.0)))

        filtered_df = history_df.iloc[start_idx:end_idx]
    else:
        filtered_df = None
    return (filtered_df,)


@app.cell(hide_code=True)
def _(filtered_df, go, mo):
    # Cell 4: Parallel Coordinates Plot (Reactive)

    if filtered_df is not None and not filtered_df.empty:
        lll=fig_par = go.Figure(data=go.Parcoords(
            line=dict(color=filtered_df['Cost'], colorscale='Turbo_r', showscale=True),
            dimensions=[{'label': col, 'values': filtered_df[col]} for col in filtered_df.columns]
        ))
        fig_par.update_layout(title="<b>Parallel Coordinates of Filtered Solutions</b>", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        mo.ui.plotly(fig_par)
    else:
        lll=mo.md("ℹ️ *No data in the selected filter range. Adjust the slider above.*")
    lll
    return (lll,)


@app.cell
def _(get_ga_best_genes, get_ga_convergence, go, mo):
    # Cell 3: Static Summary - Best Solution & Convergence Curve

    # Define the output variable at the start to ensure it always exists.
    summary_output = None

    # Check if the necessary data from the GA run is available.
    if 'get_ga_best_genes' in globals() and get_ga_best_genes():
        best_genes_data = get_ga_best_genes()
        history_scores_data = get_ga_convergence()

        # Markdown table for the best parameters found (no changes needed here).
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

        # --- ENHANCEMENTS APPLIED HERE ---

        # 1. Create the Plotly figure for the convergence history.
        fig_conv = go.Figure(data=go.Scatter(
            x=list(range(1, len(history_scores_data) + 1)), 
            y=history_scores_data,
            mode='lines+markers',
            line=dict(color='#00d2ff', width=3), # A vibrant cyan color for the line
            marker=dict(color='#00d2ff', size=6),
            fill='tozeroy', # Add a fill under the line for better visuals
            fillcolor='rgba(0, 210, 255, 0.1)' # A subtle, semi-transparent fill
        ))

        # 2. Update the layout for theme compatibility and better aesthetics.
        fig_conv.update_layout(
            title="<b>Overall Convergence History</b>",
            height=350,
            # --- Theme Adjustments ---
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font=dict(color="#aaa"),        # Neutral gray font color for text
            xaxis_title="Generation",
            yaxis_title="Best Cost (ISE)",
            # Style the grid lines to be subtle and theme-friendly
            xaxis=dict(gridcolor='rgba(170, 170, 170, 0.2)'),
            yaxis=dict(gridcolor='rgba(170, 170, 170, 0.2)')
        )

        # 3. Assemble the final view.
        summary_output = mo.vstack([
            mo.vstack([mo.ui.plotly(fig_conv)], align="center"),
        ])

    # This line will display the final output object.
    summary_output
    return results_table, summary_output


@app.cell
def _(lll, mo, results_table, summary_output):
    just_table_to_show=lll
    if summary_output!=None:
        just_table_to_show=mo.vstack([mo.md("### 🏆 Best Solution & Convergence"),mo.md(results_table)])
    just_table_to_show
    return


@app.cell
def _(filtered_df):
    filtered_df
    return


@app.cell
def _(filtered_df, go, mo):
    # Cell 5: 3D Landscape Plot (Reactive and Enhanced Visualization)

    # Define the output variable at the start to ensure it always exists.
    plot_3d_output = None

    # Check if the filtered data from the upstream cell is available and not empty.
    if filtered_df is not None and not filtered_df.empty:

        # --- Step 1: Identify the "Winner" ---
        # Find the index of the solution with the lowest cost in the filtered data.
        winner_idx = filtered_df['Cost'].idxmin()
        winner_solution = filtered_df.loc[winner_idx]

        # --- Step 2: Create a Figure with Two Layers ---
        # We will use two separate 'traces' to style the winner differently.
        fig_3d = go.Figure()

        # Trace 1: All the other data points (the "trials")
        fig_3d.add_trace(go.Scatter3d(
            x=filtered_df['Kp_base'], 
            y=filtered_df['Ki_base'], 
            z=filtered_df['Cost'], 
            mode='markers',
            marker=dict(
                size=5,  # Slightly larger points
                color=filtered_df['Cost'], 
                colorscale='Viridis_r', # "_r" reverses the scale (purple = good)
                showscale=True,
                colorbar=dict(title='Cost (ISE)'), # Add a title to the color bar
                opacity=0.7 # Use opacity to see dense clusters
            ),
            name='GA Trials' # Label for the legend
        ))

        # Trace 2: The single "Winner" point
        fig_3d.add_trace(go.Scatter3d(
            x=[winner_solution['Kp_base']], # Must be in a list
            y=[winner_solution['Ki_base']],
            z=[winner_solution['Cost']],
            mode='markers',
            marker=dict(
                size=12,             # Much larger to stand out
                color='#FFD777',     # A distinct, bright gold color
                symbol='diamond',    # A unique symbol for the winner
                line=dict(width=1.5, color='black') # A black border for contrast
            ),
            name='Best Solution' # Label for the legend
        ))

        # --- Step 3: Apply Professional Layout and Styling ---
        fig_3d.update_layout(
            title="<b>3D Optimization Landscape with Best Solution Highlighted</b>", 
            height=600, # A bit taller for a better 3D perspective
            width=None, # Let the plot expand to the full width of the cell
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', # Makes the plot background transparent
            font=dict(family="Segoe UI, system-ui, sans-serif", color="#aaa"), # Clean, neutral font
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1), # Position the legend neatly
            margin=dict(l=0, r=0, b=0, t=40), # Adjust margins for a tight fit
            scene=dict(
                xaxis_title='Proportional Gain (Kp)', # More descriptive titles
                yaxis_title='Integral Gain (Ki)',
                zaxis_title='Cost (ISE)',
                # Style the 3D axes for better visibility in dark/light modes
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
            )
        )

        # The final output for this cell is the Plotly figure.
        plot_3d_output = mo.ui.plotly(fig_3d)

    # This final line will display either the plot or None (if no data),
    # preventing any errors.
    plot_3d_output
    return


@app.cell
def _(filtered_df, mo):
    # Cell 6: Detailed Log Table (Reactive and Directly Visible)

    # Define the output variable at the start to ensure it always exists.
    table_output = None

    # Check if the filtered data from the upstream cell is available and not empty.
    if filtered_df is not None and not filtered_df.empty:

        # 1. Create a clear title for the table using Marimo's markdown.
        table_title = mo.md("### Detailed Log of Filtered Solutions")

        # 2. Create the interactive table object directly.
        # The 'label' parameter provides helpful context for the table.
        detailed_table = mo.ui.table(
            data=filtered_df.round(4).to_dict('records'), 
            pagination=True, 
            page_size=14,
            label="A log of all trials within the selected performance percentile."
        )

        # 3. Combine the title and the table into a single vertical stack for a clean layout.
        table_output = mo.vstack([table_title, detailed_table])

    # This final line displays the output.
    # It will show the title and table if data exists, or nothing if there's no data.
    table_output
    return


if __name__ == "__main__":
    app.run()
