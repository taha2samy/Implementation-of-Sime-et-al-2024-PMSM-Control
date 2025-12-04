import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
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
    # القائمة بتختار منها، ولما تغيرها، القيم اللي فوق (في Cell 1) مش هتتأثر
    controller_selector = mo.ui.dropdown(
        options=[OPT_PID, OPT_FOPID, OPT_GA],
        value=OPT_PID,
        label="Architecture",
        full_width=True
    )

    mo.vstack([
        mo.md("### 🎛️ Control Strategy Selection"),
        mo.callout(
            mo.vstack([
                mo.md("Select the controller type below."),
                controller_selector
            ]),
            kind="neutral"
        )
    ])
    return (controller_selector,)


@app.cell
def _(np):
    class RealFuzzyController:
        def __init__(self):
            # 1. تعريف المصطلحات (NB: Negative Big ... PB: Positive Big)
            self.terms = ['NB', 'NS', 'Z', 'PS', 'PB']

            # 2. مراكز المثلثات (Membership Functions Centers)
            self.centers = {
                'NB': -1.0, 'NS': -0.5, 'Z': 0.0, 'PS': 0.5, 'PB': 1.0
            }

            # 3. جدول القوانين (Rule Base Matrix)
            # الصفوف: Error | الأعمدة: Delta Error
            # النتيجة: Gain Multiplier (S, M, B, VB)
            self.rule_matrix = [
                ['VB', 'B',  'B',  'M',  'S'], # NB
                ['B',  'B',  'M',  'S',  'S'], # NS
                ['B',  'M',  'S',  'M',  'B'], # Z
                ['S',  'S',  'M',  'B',  'B'], # PS
                ['S',  'M',  'B',  'B',  'VB']  # PB
            ]

            # 4. قيم الخرج (Output Singletons)
            # S=0.8 (هدّي اللعب), M=1.0 (عادي), B=1.5 (زود), VB=2.5 (زود جامد)
            self.outputs = {'S': 0.8, 'M': 1.0, 'B': 1.5, 'VB': 2.5}

        def triangle_mf(self, x, center, width=0.5):
            """حساب درجة العضوية للمثلث"""
            return max(0, 1 - abs(x - center) / width)

        def compute_alpha(self, error, delta_error):
            """
            Input: Error, Change of Error
            Output: Alpha Gain (التكبير/التصغير)
            """
            # أ. توحيد المقاييس (Normalization)
            # بنفترض أقصى خطأ 250، وبنضرب في 5 لحساسية أعلى
            e_norm = np.clip(error / 250.0 * 5.0, -1, 1)
            # بنفترض أقصى تغير 10
            de_norm = np.clip(delta_error / 10.0, -1, 1)

            numerator = 0.0
            denominator = 0.0

            # ب. الاستنتاج (Inference Engine)
            for i, e_term in enumerate(self.terms):
                mu_e = self.triangle_mf(e_norm, self.centers[e_term])
                if mu_e == 0: continue 

                for j, de_term in enumerate(self.terms):
                    mu_de = self.triangle_mf(de_norm, self.centers[de_term])
                    if mu_de == 0: continue

                    # (Min Implication) ناخد القيمة الأقل
                    firing_strength = min(mu_e, mu_de)

                    # نجيب النتيجة من الجدول
                    output_term = self.rule_matrix[i][j]
                    output_val = self.outputs[output_term]

                    # تجميع للـ Defuzzification
                    numerator += firing_strength * output_val
                    denominator += firing_strength

            # ج. حساب المتوسط الموزون (Center of Gravity)
            if denominator == 0:
                return 1.0

            return numerator / denominator


    return (RealFuzzyController,)


@app.cell
def _(RealFuzzyController, go, mo, np):
    fuzzy_viz = RealFuzzyController()

    # ==========================================
    # PART 1: Fuzzification Visualization (Membership Functions)
    # ==========================================
    x_range = np.linspace(-1.5, 1.5, 200)
    fig_mf = go.Figure()

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db'] 

    for idx, term in enumerate(fuzzy_viz.terms):
        y_values = [fuzzy_viz.triangle_mf(x, fuzzy_viz.centers[term]) for x in x_range]

        fig_mf.add_trace(go.Scatter(
            x=x_range, y=y_values,
            name=f"Term: {term}",
            fill='tozeroy',
            line=dict(color=colors[idx], width=2),
            opacity=0.6
        ))

    fig_mf.update_layout(
        title=dict(
            text="<b>1. Fuzzification Stage (Membership Functions)</b><br><span style='font-size:12px; color:gray;'>How the controller translates Error into Logic</span>",
            y=0.9,  # نزلنا العنوان شوية عشان ميبقاش في سقف الصفحة
            x=0.01,
            xanchor='left',
            yanchor='top'
        ),
        xaxis_title="Normalized Input (Error)",
        yaxis_title="Degree of Membership (µ)",
        height=400, # زودنا الطول شوية عشان العناصر تاخد راحتها
        # === التعديل هنا: زودنا الهوامش العلوية (t=140) عشان نفصل العنوان عن الرسمة ===
        margin=dict(l=20, r=20, t=140, b=20),
        # === التعديل هنا: رفعنا الـ Legend لفوق خالص (y=1.3) ===
        legend=dict(
            orientation="h", 
            y=1.3, 
            x=0.5, 
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)" # خلفية شفافة عشان لو جه فوق خط ميبوظش الشكل
        ),
        template="plotly_white"
    )

    # ==========================================
    # PART 2: Inference & Rule Base Visualization (Control Surface)
    # ==========================================
    res = 40
    e_vec = np.linspace(-250, 250, res)
    de_vec = np.linspace(-10, 10, res)
    z_surface = np.zeros((res, res))

    for i in range(res):
        for j in range(res):
            z_surface[j][i] = fuzzy_viz.compute_alpha(e_vec[i], de_vec[j])

    fig_surf = go.Figure(data=[go.Surface(
        z=z_surface, 
        x=e_vec, 
        y=de_vec,
        colorscale='Viridis',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
        opacity=0.9
    )])

    fig_surf.update_layout(
        title=dict(
            text="<b>2. Inference & Defuzzification (Control Surface)</b><br><span style='font-size:12px; color:gray;'>The complete decision map (Rule Base result)</span>",
            y=0.92,
            x=0.01
        ),
        scene=dict(
            xaxis_title='Error (E)',
            yaxis_title='Change of Error (dE)',
            zaxis_title='Output Gain (α)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        height=600,
        # === التعديل هنا: زودنا الهامش العلوي (t=100) ===
        margin=dict(l=10, r=10, t=100, b=10),
        template="plotly_white"
    )

    # ==========================================
    # PART 3: Render Output
    # ==========================================
    mo.vstack([
        mo.md("## 🧩 Fuzzy Controller Internals"),
        mo.md("""
        This dashboard visualizes the two main stages of the fuzzy controller shown in your diagram:
        1.  **Fuzzification:** The top plot shows the "Database," determining which linguistic term (like "Negative Big") the current error belongs to.
        2.  **Inference Engine:** The bottom 3D plot shows the result of the "Rule Base." It maps every combination of Error and Change-of-Error to a specific Gain Output.
        """),
        mo.ui.plotly(fig_mf),
        mo.ui.plotly(fig_surf)
    ])
    return (fuzzy_viz,)


@app.cell
def _(fuzzy_viz, go, mo):
    def create_rule_heatmap(controller):
        # ==========================================
        # PART 3: Rule Base Matrix Visualization (Heatmap)
        # ==========================================

        # 1. Map text rules to numbers for coloring logic
        rule_map_val = {'VB': 4, 'B': 3, 'M': 2, 'S': 1}

        rule_map_text = []
        rule_map_num = []

        rows = len(controller.terms) # 5
        cols = len(controller.terms) # 5

        # Convert the rule matrix from text to numbers
        # 'i' and 'j' are now local variables inside this function
        for i in range(rows):
            row_text = []
            row_num = []
            for j in range(cols):
                rule_str = controller.rule_matrix[i][j]
                row_text.append(rule_str)
                row_num.append(rule_map_val[rule_str])
            rule_map_text.append(row_text)
            rule_map_num.append(row_num)

        # 2. Create Heatmap
        fig_rules = go.Figure(data=go.Heatmap(
            z=rule_map_num,
            x=controller.terms, # dE terms (Columns)
            y=controller.terms, # Error terms (Rows)
            text=rule_map_text, # Show 'VB', 'S' inside blocks
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white", "family": "Arial"},
            colorscale=[
                [0.0, '#2ecc71'], # S (Green - Small Gain)
                [0.33, '#f1c40f'], # M (Yellow - Medium)
                [0.66, '#e67e22'], # B (Orange - Big)
                [1.0, '#c0392b']  # VB (Red - Very Big)
            ],
            xgap=4, # Spacing between blocks
            ygap=4,
            showscale=False # Hide the color bar since text explains it
        ))

        fig_rules.update_layout(
            title=dict(
                text="<b>3. Fuzzy Rule Base (Logic Matrix)</b><br><span style='font-size:12px; color:gray;'>Logic: If Error is (Y) AND dError is (X) &rarr; Then Gain is (Cell Value)</span>",
                y=0.9,
                x=0.01
            ),
            xaxis_title="Change of Error (dE)",
            yaxis_title="Error (E)",
            height=550,
            margin=dict(l=50, r=50, t=100, b=50),
            template="plotly_white"
        )

        # Render
        return mo.vstack([
            mo.md("## 🚦 The Controller's Brain (Rule Matrix)"),
            mo.md("""
            This heatmap visualizes the lookup table used by the controller to make decisions.
            *   **Red Cells (VB):** High aggression zone. Used when the error is large.
            *   **Green Cells (S):** Stability zone. Used when the system is close to the target.
            """),
            mo.ui.plotly(fig_rules)
        ])

    # Call the function and store the result
    heatmap_output = create_rule_heatmap(fuzzy_viz)

    heatmap_output
    return


@app.class_definition
class OustaloupFilter:
    """
    Implementation of Fractional Order calculus using Oustaloup's Recursive Approximation.
    It approximates s^alpha as a cascade of 1st-order high/low pass filters.
    """
    def __init__(self, alpha, num_poles=3, freq_low=0.01, freq_high=1000.0, dt=0.0001):
        """
        alpha: The fractional order (e.g., 0.9 for integration, -0.9 for differentiation)
               Note: usually inputs are positive for I/D terms, we handle sign internally.
        """
        self.alpha = alpha
        self.dt = dt
        self.active = (abs(alpha) > 1e-6) # لو الأس صفر، مبيعملش حاجة
        
        # مخازن لحفظ القيم السابقة (States) لكل فلتر فرعي
        # إحنا هنكسر الفلتر الكبير لفلاتر صغيرة متتابعة (Cascade) عشان الاستقرار
        self.filters_state_x = [] # مدخلات سابقة
        self.filters_state_y = [] # مخرجات سابقة
        self.coeffs = []          # معاملات الفلتر (b0, b1, a1)
        
        if self.active:
            self._compute_coefficients(alpha, num_poles, freq_low, freq_high)
            # تهيئة المخازن بالأصفار
            self.filters_state_x = [0.0] * len(self.coeffs)
            self.filters_state_y = [0.0] * len(self.coeffs)

    def _compute_coefficients(self, alpha, N, wb, wh):
        # 1. Oustaloup Math (حساب الأقطاب والأصفار في التردد Continuous)
        mu = wh / wb
        
        # معادلات أوستالوب لحساب ترددات الأصفار (zk) والأقطاب (pk)
        zeros = []
        poles = []
        
        for k in range(-N, N + 1):
            # حساب الـ omega للصفر والقطب
            w_z = wb * (wh/wb)**((k + N + 0.5*(1 - alpha))/(2*N + 1))
            w_p = wb * (wh/wb)**((k + N + 0.5*(1 + alpha))/(2*N + 1))
            zeros.append(w_z)
            poles.append(w_p)
        
        # حساب الـ Gain الكلي
        self.gain = (wh)**alpha
        
        # 2. Discretization (Tustin Method) لكل زوج (s+z)/(s+p)
        # بنحول كل جزء صغير لمعادلة: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        # المعادلة في s-domain: H(s) = (s + w_z) / (s + w_p)
        # بنعوض عن s بـ (2/dt * (1-z^-1)/(1+z^-1))
        
        for w_z, w_p in zip(zeros, poles):
            # معاملات Tustin
            k = (2.0 / self.dt)
            
            # Coefficients for (s + w_z) / (s + w_p)
            # البسط (Numerator): k(1-z) + w_z(1+z) = (k+w_z) + (w_z-k)z^-1
            # المقام (Denominator): k(1-z) + w_p(1+z) = (k+w_p) + (w_p-k)z^-1
            
            a0 = k + w_p
            b0 = k + w_z
            b1 = w_z - k
            a1 = w_p - k
            
            # Normalizing by a0 to make equation: y[n] = ...
            self.coeffs.append({
                'b0': b0 / a0,
                'b1': b1 / a0,
                'a1': a1 / a0
            })

    def compute(self, input_val):
        """
        Processing function: Takes raw signal, returns fractionally filtered signal.
        """
        if not self.active:
            return input_val
        
        # نبدأ بالإشارة مضروبة في الـ Gain الأساسي
        current_signal = input_val * self.gain
        
        # نمرر الإشارة على سلسلة الفلاتر (Cascade)
        for i, coeff in enumerate(self.coeffs):
            # استرجاع القيم السابقة
            x_prev = self.filters_state_x[i]
            y_prev = self.filters_state_y[i]
            
            # معادلة الفلتر (Difference Equation)
            # y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
            output = (coeff['b0'] * current_signal) + (coeff['b1'] * x_prev) - (coeff['a1'] * y_prev)
            
            # تحديث الذاكرة
            self.filters_state_x[i] = current_signal
            self.filters_state_y[i] = output
            
            # الإشارة اللي خرجت من هنا، تدخل في الفلتر اللي بعده
            current_signal = output
            
        return current_signal


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


@app.cell
def _(RealFuzzyController, np):
    def simulate_pmsm_system(ctrl_params, scenario, motor_phys, vehicle_phys, strategy_type):
        """
        Returns: time, speed, torque, current, FUZZY_GAIN, ref
        """
        # 1. Inputs & Parameters
        t_end = scenario["Time"]
        w_target = scenario["Ref"]
        t_load = scenario["T_load"]
        load_val = scenario["Load"]

        Rs, P, Psi_m, J, B = motor_phys["Rs"], motor_phys["P"], motor_phys["Psi"], motor_phys["J"], motor_phys["B"]
        Mass, Rw, Gear, Cd, Area, Rho = vehicle_phys["Mass"], vehicle_phys["Rw"], vehicle_phys["Gear"], vehicle_phys["Cd"], vehicle_phys["Area"], vehicle_phys["Rho"]

        kp, ki, kd = ctrl_params["Kp"], ctrl_params["Ki"], ctrl_params["Kd"]
        lam = ctrl_params.get("Lambda", 1.0)
        mu = ctrl_params.get("Mu", 1.0)
        alpha_scale = ctrl_params.get("Fuzzy_Scale", 1.0)

        # === 1. تجهيز العقل (Fuzzy) ===
        fuzzy_brain = RealFuzzyController()

        # === 2. تجهيز القلب (Fractional Filters) ===
        use_fractional = ("Fractional" in strategy_type) or ("GA" in strategy_type)
    
        # فلتر التكامل (بياخد سالب lambda)
        frac_integrator = OustaloupFilter(-lam) if use_fractional else None
    
        # فلتر التفاضل (بياخد mu موجبة)
        frac_differentiator = OustaloupFilter(mu) if use_fractional else None

        # 3. Initialization
        dt = 0.0001
        steps = int(t_end / dt)
        time = np.linspace(0, t_end, steps)

        speed_arr = np.zeros(steps)
        torque_arr = np.zeros(steps)
        iq_arr = np.zeros(steps)
        gain_log = np.zeros(steps) 

        w_mech = 0.0
        iq_current = 0.0
        error_sum = 0.0
        prev_error = 0.0

        # 4. Main Simulation Loop
        for i in range(steps):
            t = time[i]
            ref = w_target if t > 0.005 else 0.0
            error = ref - w_mech
            delta_error = error - prev_error

            # --- A. Fuzzy Logic Block ---
            gain_factor = 1.0
            if "GA" in strategy_type or "Fuzzy" in strategy_type:
                fuzzy_output = fuzzy_brain.compute_alpha(error, delta_error)
                gain_factor = fuzzy_output * alpha_scale

            gain_log[i] = gain_factor

            # --- B. Controller Block (PID vs FOPID) ---
        
            # 1. Proportional
            p_term = (kp * gain_factor) * error

            # 2. Integral & Derivative
            if use_fractional:
                # === Real Fractional Calculus ===
                i_signal = frac_integrator.compute(error)
                d_signal = frac_differentiator.compute(error)
            
                i_term = (ki * gain_factor) * i_signal
                d_term = (kd * gain_factor) * d_signal
            else:
                # === Standard Integer Calculus ===
                error_sum += error * dt
                i_term = (ki * gain_factor) * error_sum
                d_term = (kd * gain_factor) * (delta_error / dt)

            # تجميع الإشارة
            iq_ref = p_term + i_term + d_term

            # Saturation (Limiter) - FIXED HERE
            if iq_ref > 200: 
                iq_ref = 200
                if not use_fractional: 
                    error_sum -= error * dt
            elif iq_ref < -200: 
                iq_ref = -200
                if not use_fractional: 
                    error_sum -= error * dt

            # --- C. Plant Model (Physics) ---
            iq_current += (iq_ref - iq_current) * (dt/0.001)
            Te = 1.5 * P * Psi_m * iq_current

            T_ext = load_val if t >= t_load else 0.0
            v = (w_mech/Gear)*Rw
            F_drag = 0.5 * Rho * Cd * Area * v * abs(v)
            T_drag = (F_drag * Rw)/Gear

            dw_dt = (Te - T_ext - T_drag - B*w_mech) / J
            w_mech += dw_dt * dt
            if w_mech < 0: w_mech = 0

            speed_arr[i] = w_mech
            torque_arr[i] = Te
            iq_arr[i] = iq_current
            prev_error = error

        return time, speed_arr, torque_arr, iq_arr, gain_log, ref

    return (simulate_pmsm_system,)


@app.cell
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
        mo.md("## 🏎️ Vehicle Performance"),
        mo.ui.plotly(final_dashboard_fig)
    ])
    return fuzzy_data, ref_data, t_data, w_data


@app.cell
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
            mo.md("### 🧠 Intelligent Controller Analysis"),
            mo.Html(html_report),
            mo.ui.plotly(fig_brain)
        ])

    # 3. Render (This line displays the result on the screen)
    output_view
    return


@app.cell
def _(go, make_subplots, mo, np):
    test_order = 0.5 
    filter_viz = OustaloupFilter(test_order, freq_low=0.01, freq_high=100.0)

    # ------------------------------------------
    # Part A: Frequency Response (Bode Plot)
    # ------------------------------------------
    freqs = np.logspace(-3, 3, 500)
    magnitudes = []
    phases = []

    for w in freqs:
        s = 1j * w
        h_s = filter_viz.gain
        wb = 0.01; wh = 100.0; N = 3
        alpha = test_order
        for k in range(-N, N + 1):
            w_z = wb * (wh/wb)**((k + N + 0.5*(1 - alpha))/(2*N + 1))
            w_p = wb * (wh/wb)**((k + N + 0.5*(1 + alpha))/(2*N + 1))
            h_s *= (s + w_z) / (s + w_p)

        magnitudes.append(20 * np.log10(np.abs(h_s)))
        phases.append(np.angle(h_s, deg=True))

    # ------------------------------------------
    # Part B: Time Domain Response
    # ------------------------------------------
    time_sim = OustaloupFilter(test_order)
    t_vec = np.linspace(0, 10, 1000)
    input_step = np.ones_like(t_vec)
    output_response = []
    for val in input_step:
        output_response.append(time_sim.compute(val))

    # ------------------------------------------
    # Plotting Configuration
    # ------------------------------------------
    c_mag = '#00d2ff'   # Cyan
    c_phase = '#e74c3c' # RED (عشان يبان واضح زي ما طلبت)
    c_resp = '#54a0ff'  # Blue
    c_text = '#8899a6'  # Grey
    c_grid = 'rgba(128, 128, 128, 0.2)' 

    # === التعديل الجوهري هنا ===
    # لازم نقوله specs=[[{"secondary_y": True}, {}]]
    # عشان يفهم إن الرسمة الأولى فيها محورين Y
    fig_filter = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b style='color:#00d2ff'>Bode Plot (Frequency)</b>", 
            "<b style='color:#54a0ff'>Step Response (Time)</b>"
        ),
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}, {}]] # <--- الحل السحري
    )

    # Plot 1: Magnitude (على المحور الأساسي)
    fig_filter.add_trace(go.Scatter(
        x=freqs, y=magnitudes,
        name="Magnitude (dB)",
        line=dict(color=c_mag, width=2.5),
        legendgroup="group1"
    ), row=1, col=1, secondary_y=False)

    # Plot 1: Phase (على المحور الثانوي - الخط الأحمر)
    fig_filter.add_trace(go.Scatter(
        x=freqs, y=phases,
        name="Phase (Deg)",
        line=dict(color=c_phase, width=2.5),
        legendgroup="group1"
    ), row=1, col=1, secondary_y=True) # <--- ظهرناه هنا

    # Plot 2: Time Response
    fig_filter.add_trace(go.Scatter(
        x=t_vec, y=output_response,
        name=f"s<sup>{test_order}</sup> Response",
        line=dict(color=c_resp, width=2.5)
    ), row=1, col=2)

    # Input Step
    fig_filter.add_trace(go.Scatter(
        x=t_vec, y=input_step,
        name="Input Step",
        line=dict(color=c_text, dash='dash', width=1.5),
        opacity=0.5
    ), row=1, col=2)

    # ------------------------------------------
    # Styling
    # ------------------------------------------
    base_axis_style = dict(
        showgrid=True, gridcolor=c_grid, zerolinecolor=c_grid, tickfont=dict(color=c_text)
    )

    fig_filter.update_layout(
        title=dict(
            text=f"<b>Fractional Filter Analysis (Oustaloup Method)</b><br><span style='font-size:12px; color:{c_text};'>Verifying that s<sup>{test_order}</sup> behaves like a half-derivative</span>",
            y=0.92, x=0.05
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=c_text),
        hovermode="x unified",
        margin=dict(t=100, l=80, r=80, b=50),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", bgcolor='rgba(0,0,0,0)'),
    
        # Axis 1 (Magnitude)
        xaxis=dict(type="log", title="Frequency (rad/s)", title_font=dict(color=c_text), **base_axis_style),
        yaxis=dict(title="Magnitude (dB)", title_font=dict(color=c_mag), **base_axis_style),
    
        # Axis 2 (Phase - The Red Line)
        yaxis2=dict(
            title="Phase (Deg)",
            title_font=dict(color=c_phase),
            range=[0, 90],
            showgrid=False,
            tickfont=dict(color=c_phase)
        ),
    
        # Axis 3 & 4 (Time Plot)
        xaxis2=dict(title="Time (s)", title_font=dict(color=c_text), **base_axis_style),
        yaxis3=dict(title="Amplitude", title_font=dict(color=c_text), **base_axis_style)
    )

    # Target Line
    theoretical_phase = test_order * 90
    fig_filter.add_hline(
        y=theoretical_phase, line_dash="dot", line_color=c_phase, 
        annotation_text=f"Target: {theoretical_phase}°", annotation_font=dict(color=c_phase),
        row=1, col=1, secondary_y=True
    )

    mo.vstack([
        mo.md("## 📉 The Math Behind FOPID (Oustaloup Check)"),
        mo.md(f"""
        Checking the mathematical validity of the fractional operator **s<sup>{test_order}</sup>**:
        """),
        mo.ui.plotly(fig_filter)
    ])

    return


@app.cell
def _(np, simulate_pmsm_system):
    class GeneticOptimizerEngine:
        """
        Genetic Engine based STRICTLY on the Research Paper Methodology.
        Target: Minimize Equation (36) -> ISE (Integral of Squared Error).
        """
        def __init__(self, motor_phys, vehicle_phys, sim_setup, custom_bounds=None):
            self.motor_phys = motor_phys
            self.vehicle_phys = vehicle_phys
            self.sim_setup = sim_setup
        
            if custom_bounds:
                self.bounds = custom_bounds
            else:
                # Default Bounds tailored around Table 2 values in the paper
                # Paper Results: Kp=0.53, Ki=1.83, Kd=0.0005, Mu=0.86, Lam=0.97
                self.bounds = [
                    (0.1, 5.0),    # Kp
                    (0.1, 5.0),    # Ki
                    (0.0, 0.1),    # Kd (Usually very small)
                    (0.1, 1.1),    # Lambda (Around 1.0)
                    (0.1, 1.1),    # Mu (Around 1.0)
                    (0.1, 5.0)     # FuzzyScale
                ]

        def _evaluate_fitness(self, genes):
            # 1. Map Genes to Controller Parameters
            ctrl_candidate = {
                "Kp": genes[0], "Ki": genes[1], "Kd": genes[2],
                "Lambda": genes[3], "Mu": genes[4], "Fuzzy_Scale": genes[5]
            }
        
            # 2. Run Simulation
            # Using 1.0s is sufficient to capture rise time and overshoot
            # Paper mentions "Step Response", so we simulate a step input.
            fast_scenario = self.sim_setup.copy()
            fast_scenario["Time"] = 1.0  
        
            # Run simulation with "GA" strategy
            time, speed, _, _, _, ref = simulate_pmsm_system(
                ctrl_candidate, 
                fast_scenario, 
                self.motor_phys, 
                self.vehicle_phys, 
                strategy_type="GA"
            )
        
            # 3. Calculate Fitness (Equation 36 in Paper)
            # ISE = Integral( e(t)^2 ) dt
        
            error = ref - speed
        
            # Discrete Integration (Summation * dt)
            dt = time[1] - time[0] # Time step
            ise = np.sum(np.square(error)) * dt
        
            # Stability Check:
            # If the system is unstable (Speed goes to infinity or NaN), punish heavily.
            # This is implicit in any control paper.
            if np.isnan(ise) or np.isinf(ise) or np.max(speed) > (ref * 2):
                return 1e6 # High penalty for instability
            
            return ise

        def run(self, pop_size=20, generations=10, mutation_rate=0.1):
            """
            Standard Genetic Algorithm Loop.
            """
            # A. Initialize Population
            population = []
            for _ in range(pop_size):
                ind = [np.random.uniform(L, H) for L, H in self.bounds]
                population.append(ind)
        
            best_sol = None
            best_cost = float('inf')
            history_log = []

            for gen in range(generations):
                scores = []
                # 1. Evaluate Fitness for all individuals
                for ind in population:
                    score = self._evaluate_fitness(ind)
                    scores.append(score)
                
                    if score < best_cost:
                        best_cost = score
                        best_sol = ind[:]
            
                history_log.append(f"Gen {gen+1}: ISE Cost = {best_cost:.5f}")

                # 2. Create Next Generation
                new_pop = []
            
                # Elitism: Carry over the single best individual (common practice)
                best_idx = np.argmin(scores)
                new_pop.append(population[best_idx])
            
                while len(new_pop) < pop_size:
                    # Tournament Selection
                    # Pick 2 random parents and select the better one
                    candidates = np.random.randint(0, pop_size, size=2)
                    p1_idx = candidates[0] if scores[candidates[0]] < scores[candidates[1]] else candidates[1]
                
                    candidates = np.random.randint(0, pop_size, size=2)
                    p2_idx = candidates[0] if scores[candidates[0]] < scores[candidates[1]] else candidates[1]
                
                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]
                
                    # Crossover (Arithmetic)
                    child = []
                    beta = np.random.random()
                    for g1, g2 in zip(parent1, parent2):
                        child.append(beta*g1 + (1-beta)*g2)
                
                    # Mutation
                    for k in range(len(child)):
                        if np.random.random() < mutation_rate:
                            # Random reset within bounds (Standard Mutation)
                            child[k] = np.random.uniform(self.bounds[k][0], self.bounds[k][1])
                
                    new_pop.append(child)
            
                population = new_pop

            return best_sol, best_cost, history_log

    return (GeneticOptimizerEngine,)


@app.cell
def _(mo):
    mo.md("### 🧬 Evolutionary Optimization Console")

    # 1. General Settings
    ga_controls = mo.ui.dictionary({
        "pop": mo.ui.number(5, 100, value=10, step=1, label="Population Size"),
        "gen": mo.ui.number(1, 200, value=5, step=1, label="Generations"),
        "mut": mo.ui.number(0.01, 0.5, value=0.1, step=0.01, label="Mutation Rate")
    })

    # 2. Search Space Bounds (The Fix is Here)
    # We use mo.ui.array([...]) instead of list [...] to fix the ValueError
    bounds_ui = mo.ui.dictionary({
        "Kp":   mo.ui.array([mo.ui.number(0.1, 500, value=0.1, full_width=True), mo.ui.number(0.1, 500, value=50.0, full_width=True)]),
        "Ki":   mo.ui.array([mo.ui.number(0.1, 500, value=0.1, full_width=True), mo.ui.number(0.1, 500, value=50.0, full_width=True)]),
        "Kd":   mo.ui.array([mo.ui.number(0.0, 100, value=0.0, full_width=True), mo.ui.number(0.0, 100, value=1.0, full_width=True)]),
        "Lam":  mo.ui.array([mo.ui.number(0.1, 2.0, value=0.1, full_width=True), mo.ui.number(0.1, 2.0, value=1.5, full_width=True)]),
        "Mu":   mo.ui.array([mo.ui.number(0.1, 2.0, value=0.1, full_width=True), mo.ui.number(0.1, 2.0, value=1.5, full_width=True)]),
        "Fuz":  mo.ui.array([mo.ui.number(0.1, 10.0, value=0.1, full_width=True), mo.ui.number(0.1, 10.0, value=5.0, full_width=True)]),
    })

    # 3. Custom Grid Layout for Bounds (Professional Look)
    # Helper function to create a row
    def bound_row(name, key):
        return mo.hstack([
            mo.md(f"**{name}**").style({"width": "120px", "align-self": "center"}), # Label
            bounds_ui[key][0], # Min Input
            mo.md("↔️").style({"padding": "0 10px", "align-self": "center"}),
            bounds_ui[key][1]  # Max Input
        ], align="center")

    bounds_widget = mo.accordion({
        "⚙️ Configure Search Bounds (حدود البحث)": mo.vstack([
            # Header Row
            mo.hstack([
                mo.md("Param").style({"width": "120px", "font-weight": "bold", "color": "#666"}),
                mo.md("Minimum Value").style({"flex": "1", "text-align": "center", "font-weight": "bold", "color": "#666"}),
                mo.md("").style({"width": "30px"}),
                mo.md("Maximum Value").style({"flex": "1", "text-align": "center", "font-weight": "bold", "color": "#666"}),
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

    # 4. Action Button
    run_ga_btn = mo.ui.button(label="🚀 Start Evolution Process", kind="success",value=1)
    run_ga_btn.on_click = lambda value: True
    # 5. Final Render
    mo.vstack([
        mo.callout(
            mo.vstack([
                mo.md("### 🛠️ Optimization Settings"),
                mo.hstack([ga_controls], justify="center"), # Settings in one line
                mo.md("---"),
                bounds_widget,
                mo.md(""), # Spacer
                run_ga_btn
            ]),
            kind="neutral"
        )
    ])

    return bounds_ui, ga_controls, run_ga_btn


app._unparsable_cell(
    r"""
    class GeneticOptimizerEngine:
        \"\"\"
        Genetic Engine based STRICTLY on the Research Paper Methodology.
        Target: Minimize Equation (36) -> ISE (Integral of Squared Error).
        \"\"\"
        def __init__(self, motor_phys, vehicle_phys, sim_setup, custom_bounds=None):
            self.motor_phys = motor_phys
            self.vehicle_phys = vehicle_phys
            self.sim_setup = sim_setup
        
            if custom_bounds:
                self.bounds = custom_bounds
            else:
                # Default Bounds tailored around Table 2 values in the paper
                # Paper Results: Kp=0.53, Ki=1.83, Kd=0.0005, Mu=0.86, Lam=0.97
                self.bounds = [
                    (0.1, 5.0),    # Kp
                    (0.1, 5.0),    # Ki
                    (0.0, 0.1),    # Kd (Usually very small)
                    (0.1, 1.1),    # Lambda (Around 1.0)
                    (0.1, 1.1),    # Mu (Around 1.0)
                    (0.1, 5.0)     # FuzzyScale
                ]

        def _evaluate_fitness(self, genes):
            # 1. Map Genes to Controller Parameters
            ctrl_candidate = {
                \"Kp\": genes[0], \"Ki\": genes[1], \"Kd\": genes[2],
                \"Lambda\": genes[3], \"Mu\": genes[4], \"Fuzzy_Scale\": genes[5]
            }
        
            # 2. Run Simulation
            # Using 1.0s is sufficient to capture rise time and overshoot
            # Paper mentions \"Step Response\", so we simulate a step input.
            fast_scenario = self.sim_setup.copy()
            fast_scenario[\"Time\"] = 1.0  
        
            # Run simulation with \"GA\" strategy
            time, speed, _, _, _, ref = simulate_pmsm_system(
                ctrl_candidate, 
                fast_scenario, 
                self.motor_phys, 
                self.vehicle_phys, 
                strategy_type=\"GA\"
            )
        
            # 3. Calculate Fitness (Equation 36 in Paper)
            # ISE = Integral( e(t)^2 ) dt
        
            error = ref - speed
        
            # Discrete Integration (Summation * dt)
            dt = time[1] - time[0] # Time step
            ise = np.sum(np.square(error)) * dt
        
            # Stability Check:
            # If the system is unstable (Speed goes to infinity or NaN), punish heavily.
            # This is implicit in any control paper.
            if np.isnan(ise) or np.isinf(ise) or np.max(speed) > (ref * 2):
            

        def run(self, pop_size=20, generations=10, mutation_rate=0.1):
            \"\"\"
            Standard Genetic Algorithm Loop.
            \"\"\"
            # A. Initialize Population
            population = []
            for _ in range(pop_size):
                ind = [np.random.uniform(L, H) for L, H in self.bounds]
                population.append(ind)
        
            best_sol = None
            best_cost = float('inf')
            history_log = []

            for gen in range(generations):
                scores = []
                # 1. Evaluate Fitness for all individuals
                for ind in population:
                    score = self._evaluate_fitness(ind)
                    scores.append(score)
                
                    if score < best_cost:
                        best_cost = score
                        best_sol = ind[:]
            
                history_log.append(f\"Gen {gen+1}: ISE Cost = {best_cost:.5f}\")

                # 2. Create Next Generation
                new_pop = []
            
                # Elitism: Carry over the single best individual (common practice)
                best_idx = np.argmin(scores)
                new_pop.append(population[best_idx])
            
                while len(new_pop) < pop_size:
                    # Tournament Selection
                    # Pick 2 random parents and select the better one
                    candidates = np.random.randint(0, pop_size, size=2)
                    p1_idx = candidates[0] if scores[candidates[0]] < scores[candidates[1]] else candidates[1]
                
                    candidates = np.random.randint(0, pop_size, size=2)
                    p2_idx = candidates[0] if scores[candidates[0]] < scores[candidates[1]] else candidates[1]
                
                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]
                
                    # Crossover (Arithmetic)
                    child = []
                    beta = np.random.random()
                    for g1, g2 in zip(parent1, parent2):
                        child.append(beta*g1 + (1-beta)*g2)
                
                    # Mutation
                    for k in range(len(child)):
                        if np.random.random() < mutation_rate:
                            # Random reset within bounds (Standard Mutation)
                            child[k] = np.random.uniform(self.bounds[k][0], self.bounds[k][1])
                
                    new_pop.append(child)
            
                population = new_pop
    """,
    name="_"
)


@app.cell
def _(
    GeneticOptimizerEngine,
    bounds_ui,
    ga_controls,
    mo,
    motor_ui,
    run_ga_btn,
    sim_settings,
    vehicle_ui,
):
    result_view = mo.md("ℹ️ *System Ready. Adjust settings above and click Start.*")

    if run_ga_btn.value:
    
        # A. Prepare the Bounds List from UI
        # CORRECT WAY to read from mo.ui.array: use .value to get the list [min, max]
        user_defined_bounds = [
            tuple(bounds_ui["Kp"].value),
            tuple(bounds_ui["Ki"].value),
            tuple(bounds_ui["Kd"].value),
            tuple(bounds_ui["Lam"].value),
            tuple(bounds_ui["Mu"].value),
            tuple(bounds_ui["Fuz"].value),
        ]

        # B. Initialize Engine with Custom Bounds
        optimizer = GeneticOptimizerEngine(
            motor_phys=motor_ui.value,
            vehicle_phys=vehicle_ui.value,
            sim_setup=sim_settings.value,
            custom_bounds=user_defined_bounds
        )

        # C. Run Optimization
        with mo.status.spinner(title="🧬 Evolving... Please wait"):
            best_genes, final_cost, history = optimizer.run(
                pop_size=int(ga_controls.value["pop"]),
                generations=int(ga_controls.value["gen"]),
                mutation_rate=ga_controls.value["mut"]
            )

        # D. Format Results
        results_table = f"""
        | **Parameter** | **Symbol** | **Optimized Value** | **Role** |
        | :--- | :---: | :---: | :--- |
        | Proportional Gain | $K_p$ | **{best_genes[0]:.5f}** | Error Correction Strength |
        | Integral Gain | $K_i$ | **{best_genes[1]:.5f}** | Steady-State Error Elimination |
        | Derivative Gain | $K_d$ | **{best_genes[2]:.5f}** | Damping & Prediction |
        | **Integral Order** | $\lambda$ | **{best_genes[3]:.4f}** | Fractional Memory (History) |
        | **Derivative Order** | $\mu$ | **{best_genes[4]:.4f}** | Fractional Smoothness |
        | Fuzzy Scaling | $\\alpha$ | **{best_genes[5]:.4f}** | Adaptive Logic Gain |
        """

        log_text = "\n".join(history)

        result_view = mo.vstack([
            mo.md("---"),
            mo.callout(
                mo.md(f"**✅ Optimization Successful!**<br>Minimum Cost (Error): `{final_cost:.6f}`"),
                kind="success"
            ),
            mo.md("### 🏆 Optimal Parameters Found"),
            mo.md(results_table),
            mo.accordion({
                "📜 View Convergence History (Log)": mo.md(f"```text\n{log_text}\n```")
            })
        ])

    result_view

    return


if __name__ == "__main__":
    app.run()
