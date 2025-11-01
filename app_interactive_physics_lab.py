
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Physics Lab", page_icon="🧪", layout="wide")

# ===================== Helpers =====================
def rk4(f, y0, t):
    
    # Generic 4th-order Runge–Kutta ODE solver.
    #f: function defining dy/dt = f(y, t)
    #y0: initial condition array
    #t: array of time points 
    
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + 0.5*dt*k1, t[i] + 0.5*dt)
        k3 = f(y[i] + 0.5*dt*k2, t[i] + 0.5*dt)
        k4 = f(y[i] + dt*k3, t[i] + dt)
        y[i+1] = y[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# ===================== Projectile =====================
def simulate_projectile(v0, angle_deg, g, dt, t_max, drag_coeff=0.0, mass=1.0):
     
    #Simulates projectile motion with optional linear drag.
    #Returns arrays of time, position, velocity, and energies.
    
    theta = math.radians(angle_deg)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)
    # Time array
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    # Initialize arrays
    x = np.zeros(n); y = np.zeros(n)
    vx = np.zeros(n); vy = np.zeros(n)
     # Initial conditions
    x[0] = 0.0; y[0] = 0.0; vx[0] = vx0; vy[0] = vy0
    # Numerical integration
    for i in range(n-1):
        ax = -(drag_coeff/mass) * vx[i]
        ay = -g -(drag_coeff/mass) * vy[i]
        vx[i+1] = vx[i] + ax*dt
        vy[i+1] = vy[i] + ay*dt
        x[i+1] = x[i] + vx[i]*dt
        y[i+1] = y[i] + vy[i]*dt
        # Stop when projectile hits ground
        if y[i+1] < 0:
            # Linear interpolation for smoother landing
            if y[i] > 0:
                alpha = y[i] / (y[i] - y[i+1])
                x[i+1] = x[i] + alpha * (x[i+1] - x[i])
                y[i+1] = 0.0
                vx[i+1] = vx[i] + alpha * (vx[i+1]-vx[i])
                vy[i+1] = 0.0
            x = x[:i+2]; y = y[:i+2]; t = t[:i+2]; vx = vx[:i+2]; vy = vy[:i+2]
            break
    # Energy calculations
    KE = 0.5*mass*(vx**2 + vy**2)
    PE = mass*g*np.maximum(y, 0)
    return t, x, y, vx, vy, KE, PE

def projectile_figure(t, x, y):
    # Builds an animated Plotly figure showing projectile motion.
    N = len(t)
    frames = []
    trail = max(5, N//15)
    # Animation frames
    for k in range(N):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x[:k+1], y=y[:k+1], mode="lines", name="trajectory"),
                go.Scatter(x=[x[k]], y=[y[k]], mode='text', text=['🚀'], textfont=dict(size=20), marker=dict(size=12), name="projectile"),
            ],
            name=str(k), traces=[0,1]
        ))
     # Base figure
    fig = go.Figure(
        data=[
            go.Scatter(x=[x[0]], y=[y[0]], mode="lines", name="trajectory"),
            go.Scatter(x=[x[0]], y=[y[0]], mode='text', text=['🚀'], textfont=dict(size=20), marker=dict(size=12), name="projectile"),
        ],
        frames=frames
    )
    # Layout + play/pause buttons + slider
    fig.update_layout(
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1, rangemode="tozero"),
        title="Projectile Motion (animated)",
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=1.0, xanchor="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"fromcurrent": True, "frame": {"duration": 30, "redraw": True},
                                                "transition": {"duration": 0}}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": False},
                                                  "transition": {"duration": 0}}])
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[str(k)],
                                        {"mode": "immediate",
                                         "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}],
                                  label=f"{t[k]:.2f}s") for k in range(N)],
                      x=0.1, y=-0.1, len=0.8)]
    )
    return fig

def energy_figure(t, KE, PE, title):
    #Returns animated energy vs. time chart (Kinetic, Potential, Total)
    N = len(t)
    frames = []
    for k in range(N):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=t[:k+1], y=KE[:k+1], mode="lines", name="Kinetic"),
                go.Scatter(x=t[:k+1], y=PE[:k+1], mode="lines", name="Potential"),
                go.Scatter(x=t[:k+1], y=(KE[:k+1]+PE[:k+1]), mode="lines", name="Total"),
            ], name=str(k), traces=[0,1,2]
        ))
    fig = go.Figure(
        data=[
            go.Scatter(x=[t[0]], y=[KE[0]], mode="lines", name="Kinetic"),
            go.Scatter(x=[t[0]], y=[PE[0]], mode="lines", name="Potential"),
            go.Scatter(x=[t[0]], y=[KE[0]+PE[0]], mode="lines", name="Total"),
        ],
        frames=frames
    )
    fig.update_layout(
        xaxis_title="time (s)", yaxis_title="Energy (J)", title=title,
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=1.0, xanchor="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"fromcurrent": True, "frame": {"duration": 30, "redraw": True},
                                                "transition": {"duration": 0}}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": False},
                                                  "transition": {"duration": 0}}])
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[str(k)],
                                        {"mode": "immediate",
                                         "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}],
                                  label=f"{t[k]:.2f}s") for k in range(N)],
                      x=0.1, y=-0.1, len=0.8)]
    )
    return fig

# ===================== Pendulum =====================
def simulate_pendulum(theta0_deg, omega0, L, g, dt, t_max, damping=0.0):
    #Simulates damped pendulum motion using RK4.
    #Returns time, angle, angular velocity, and energy arrays.
    theta0 = math.radians(theta0_deg)
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    def f(y, _t):
        theta, omega = y
        dtheta = omega
        domega = -(g/L)*math.sin(theta) - damping*omega
        return np.array([dtheta, domega], dtype=float)
    y = rk4(f, np.array([theta0, omega0], dtype=float), t)
    theta = y[:,0]; omega = y[:,1]
    # Convert to Cartesian coordinates
    x = L * np.sin(theta)
    y_pos = -L * np.cos(theta) + L
    # Energies (mass normalized)
    v2 = (L*omega)**2
    KE = 0.5 * v2
    PE = (g/L) * (1 - np.cos(theta))
    return t, theta, omega, x, y_pos, KE, PE

def pendulum_figure(t, x, y):
    #Animated Plotly figure of pendulum bob + rod motion.
    N = len(t)
    frames = []
    for k in range(N):
        frames.append(go.Frame(
            data=[go.Scatter(x=[0, x[k]], y=[0, y[k]], mode="lines+markers", marker=dict(size=[0,14]), name="rod+bob")],
            name=str(k), traces=[0]
        ))
    fig = go.Figure(
        data=[go.Scatter(x=[0, x[0]], y=[0, y[0]], mode="lines+markers", marker=dict(size=[0,14]), name="rod+bob")],
        frames=frames
    )
    L_plot = max(1.2*max(abs(x.max()), abs(x.min()), y.max(), abs(y.min())), 1.0)
    fig.update_layout(
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(range=[-L_plot, L_plot], zeroline=False),
        yaxis=dict(range=[-0.1, L_plot], scaleanchor="x", scaleratio=1, zeroline=False),
        title="Simple Pendulum (animated)",
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=1.0, xanchor="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"fromcurrent": True, "frame": {"duration": 30, "redraw": True},
                                                "transition": {"duration": 0}}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": False},
                                                  "transition": {"duration": 0}}])
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[str(k)],
                                        {"mode": "immediate",
                                         "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}],
                                  label=f"{t[k]:.2f}s") for k in range(N)],
                      x=0.1, y=-0.1, len=0.8)]
    )
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', text=['🚀'], textfont=dict(size=20), marker=dict(size=10), name="pivot"))
    return fig

# ===================== Mass-Spring =====================
def simulate_mass_spring(m, k, c, x0, v0, dt, t_max):
    #Simulate a damped mass-spring oscillator using simple Euler integration.
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    x = np.zeros(n); v = np.zeros(n)
    x[0] = x0; v[0] = v0
    for i in range(1, n):
        a = -(c/m)*v[i-1] - (k/m)*x[i-1]
        v[i] = v[i-1] + a*dt
        x[i] = x[i-1] + v[i]*dt
    KE = 0.5 * m * (v**2)
    PE = 0.5 * k * (x**2)
    return t, x, v, KE, PE

def spring_polyline(x_left, x_right, y_mid=0.0, amp=0.15, turns=8):
    #Generates the (x, y) coordinates of a coiled spring for plotting.
    xs = [x_left]; ys = [y_mid]
    L = max(x_right - x_left, 1e-6)
    body = np.linspace(x_left + 0.1*L, x_right - 0.1*L, 2*turns+1)
    up = True
    for xb in body:
        ys.append(y_mid + (amp if up else -amp)); xs.append(xb); up = not up
    xs.append(x_right); ys.append(y_mid)
    return np.array(xs), np.array(ys)

def mass_spring_figure(t, x, m=1.0, k=1.0):
    #Animated spring–mass system visualization.
    N = len(t)
    wall_x = -0.5
    wall = go.Scatter(x=[wall_x-0.1, wall_x-0.1, wall_x, wall_x, wall_x-0.1],
                      y=[-0.6, 0.6, 0.6, -0.6, -0.6],
                      mode="lines", fill="toself", name="wall")
    block_w = 0.4; block_h = 0.4
    frames = []
    for kf in range(N):
        left_face = 0.0 + x[kf]
        xs, ys = spring_polyline(wall_x, left_face, y_mid=0.0, amp=0.15, turns=8)
        bx = [left_face, left_face, left_face+block_w, left_face+block_w, left_face]
        by = [-block_h/2, block_h/2, block_h/2, -block_h/2, -block_h/2]
        frames.append(go.Frame(
            data=[
                wall,
                go.Scatter(x=xs, y=ys, mode="lines", name="spring"),
                go.Scatter(x=bx, y=by, mode="lines", fill="toself", name="mass"),
            ],
            name=str(kf), traces=[0,1,2]
        ))
    # Initial state
    left_face0 = 0.0 + x[0]
    xs0, ys0 = spring_polyline(wall_x, left_face0, y_mid=0.0, amp=0.15, turns=8)
    bx0 = [left_face0, left_face0, left_face0+block_w, left_face0+block_w, left_face0]
    by0 = [-block_h/2, block_h/2, block_h/2, -block_h/2, -block_h/2]
    fig = go.Figure(data=[wall,
                          go.Scatter(x=xs0, y=ys0, mode="lines", name="spring"),
                          go.Scatter(x=bx0, y=by0, mode="lines", fill="toself", name="mass")],
                    frames=frames)
    fig.update_layout(
        title="Mass-Spring Sketch (animated)", xaxis_title="x (m)", yaxis_title="",
        xaxis=dict(range=[-1.0, 2.0], zeroline=False),
        yaxis=dict(range=[-1.0, 1.0], scaleanchor="x", scaleratio=1, showticklabels=False, zeroline=False),
        showlegend=False,
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=1.0, xanchor="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"fromcurrent": True, "frame": {"duration": 30, "redraw": True},
                                                "transition": {"duration": 0}}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": False},
                                                  "transition": {"duration": 0}}])
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[str(kf)],
                                        {"mode": "immediate",
                                         "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}],
                                  label=f"{t[kf]:.2f}s") for kf in range(N)],
                      x=0.1, y=-0.1, len=0.8)]
    )
    return fig

# ===================== UI =====================
st.title("🧪 Interactive Physics Lab")
st.caption("Projectile • Pendulum • Mass–Spring — all with animated visuals and energy plots.")

# Create three tabs for the systems
tab1, tab2, tab3 = st.tabs(["🎯 Projectile Motion", "🕰️ Pendulum", "🪤 Mass–Spring"])

with tab1:
    # Input columns
    c1, c2, c3 = st.columns(3)
    with c1:
        v0 = st.number_input("Initial speed v₀ (m/s)", 1.0, 200.0, 40.0, 1.0, key="proj_v0")
        angle = st.slider("Launch angle θ (deg)", 0.0, 90.0, 40.0, 1.0, key="proj_angle")
    with c2:
        g = st.number_input("Gravity g (m/s²)", 0.5, 30.0, 9.81, 0.01, key="proj_g")
        drag = st.slider("Linear drag coeff k (kg/s)", 0.0, 2.0, 0.0, 0.01, key="proj_drag")
    with c3:
        mass = st.number_input("Mass m (kg)", 0.1, 100.0, 1.0, 0.1, key="proj_mass")
        tmax = st.number_input("Max time (s)", 1.0, 60.0, 8.0, 0.5, key="proj_tmax")
        dt = st.number_input("Time step Δt (s)", 0.005, 0.5, 0.03, 0.005, key="proj_dt")
        
    # Run simulation
    t, x, y, vx, vy, KEp, PEp = simulate_projectile(v0, angle, g, dt, tmax, drag_coeff=drag, mass=mass)

    # Display results
    colA, colB = st.columns([2, 1])
    with colA:
        st.plotly_chart(projectile_figure(t, x, y), use_container_width=True)
    with colB:
        st.subheader("Quick stats")
        st.metric("Flight time", f"{t[-1]:.2f} s")
        st.metric("Range", f"{x[-1]:.2f} m")
        st.metric("Max height", f"{y.max():.2f} m")
        st.markdown("---")
        st.plotly_chart(energy_figure(t, KEp, PEp, "Projectile Energy (animated)"), use_container_width=True)
    # 💾 Export CSV for Projectile
    if st.button("💾 Exporter les données du projectile en CSV", key="export_projectile_tab1"):
        projectile_data = pd.DataFrame({
            "Temps (s)": t,
            "Position X (m)": x,
            "Position Y (m)": y,
            "Vitesse X (m/s)": vx,
            "Vitesse Y (m/s)": vy,
            "Énergie cinétique (J)": KEp,
            "Énergie potentielle (J)": PEp
        })
        projectile_data.to_csv("projectile_data.csv", index=False)
        st.success("✅ Données du projectile enregistrées dans 'projectile_data.csv'")


with tab2:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theta0 = st.slider("Initial angle θ₀ (deg)", -89.0, 89.0, 20.0, 1.0, key="pend_theta0")
        omega0 = st.slider("Initial angular speed ω₀ (rad/s)", -5.0, 5.0, 0.0, 0.1, key="pend_omega0")
    with c2:
        L = st.number_input("Rod length L (m)", 0.1, 10.0, 1.0, 0.1, key="pend_L")
        g2 = st.number_input("Gravity g (m/s²)", 0.5, 30.0, 9.81, 0.01, key="pend_g")
    with c3:
        damping = st.slider("Damping (s⁻¹)", 0.0, 1.0, 0.02, 0.01, key="pend_damp")
        tmax2 = st.number_input("Max time (s)", 1.0, 120.0, 15.0, 0.5, key="pend_tmax")
    with c4:
        dt2 = st.number_input("Time step Δt (s)", 0.002, 0.2, 0.02, 0.002, key="pend_dt")
    t2, theta, omega, x2, y2, KE2, PE2 = simulate_pendulum(theta0, omega0, L, g2, dt2, tmax2, damping=damping)
    colA2, colB2 = st.columns([2, 1])
    with colA2:
        st.plotly_chart(pendulum_figure(t2, x2, y2), use_container_width=True)
    with colB2:
        st.subheader("Quick stats")
        period_est = 2*math.pi*math.sqrt(L/g2)
        st.metric("Small-angle period (est.)", f"{period_est:.2f} s")
        st.metric("Max |ω|", f"{abs(omega).max():.2f} rad/s")
        st.metric("Max |θ|", f"{abs(theta).max():.1f} rad")
        st.markdown("---")
        st.plotly_chart(energy_figure(t2, KE2, PE2, "Pendulum Energy (animated)"), use_container_width=True)
    # 💾 Export CSV for Pendulum
    if st.button("💾 Exporter les données du pendule en CSV", key="export_pendulum_tab2"):
        pendulum_data = pd.DataFrame({
            "Temps (s)": t2,
            "Angle θ (rad)": theta,
            "Vitesse angulaire ω (rad/s)": omega,
            "Position X (m)": x2,
            "Position Y (m)": y2,
            "Énergie cinétique (J)": KE2,
            "Énergie potentielle (J)": PE2
        })
        pendulum_data.to_csv("pendulum_data.csv", index=False)
        st.success("✅ Données du pendule enregistrées dans 'pendulum_data.csv'")


with tab3:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        m = st.number_input("Mass m (kg)", 0.1, 100.0, 1.0, 0.1, key="spring_tab3_mass")
        k = st.number_input("Spring constant k (N/m)", 0.1, 500.0, 20.0, 0.1, key="spring_tab3_k")
    with c2:
        c = st.number_input("Damping c (kg/s)", 0.0, 50.0, 0.5, 0.1, key="spring_tab3_c")
        tmax3 = st.number_input("Max time (s)", 1.0, 120.0, 15.0, 0.5, key="spring_tab3_tmax")
    with c3:
        x0 = st.slider("Initial displacement x₀ (m)", -2.0, 2.0, 0.6, 0.01, key="spring_tab3_x0")
        v0 = st.slider("Initial velocity v₀ (m/s)", -5.0, 5.0, 0.0, 0.1, key="spring_tab3_v0")
    with c4:
        dt3 = st.number_input("Time step Δt (s)", 0.001, 0.2, 0.01, 0.001, key="spring_tab3_dt")
    t3, x3, v3, KE3, PE3 = simulate_mass_spring(m, k, c, x0, v0, dt3, tmax3)
    colA3, colB3 = st.columns([2, 1])
    with colA3:
        st.plotly_chart(mass_spring_figure(t3, x3, m=m, k=k), use_container_width=True)
    with colB3:
        st.subheader("Quick stats")
        omega_n = math.sqrt(k / m)
        zeta = c / (2 * math.sqrt(k * m))
        period = (2 * math.pi / omega_n) if zeta < 1 else float("nan")
        st.metric("Natural frequency ωₙ", f"{omega_n:.2f} rad/s")
        st.metric("Damping ratio ζ", f"{zeta:.2f}")
        st.metric("Undamped period Tₙ", f"{period:.2f} s" if not math.isnan(period) else "—")
        st.markdown("---")
        st.plotly_chart(energy_figure(t3, KE3, PE3, "Mass–Spring Energy (animated)"), use_container_width=True)
    # 💾 Export CSV for Mass–Spring
    if st.button("💾 Exporter les données du masse–ressort en CSV", key="export_spring_tab3"):
        spring_data = pd.DataFrame({
            "Temps (s)": t3,
            "Position (m)": x3,
            "Vitesse (m/s)": v3,
            "Énergie cinétique (J)": KE3,
            "Énergie potentielle (J)": PE3
        })
        spring_data.to_csv("mass_spring_data.csv", index=False)
        st.success("✅ Données du masse–ressort enregistrées dans 'mass_spring_data.csv'")

