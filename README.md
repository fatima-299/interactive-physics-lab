# 🧪 Interactive Physics Lab

## 🎯 Project Objective
The main goal of **Interactive Physics Lab** is to make **learning physics more interactive, visual, and intuitive**.  
This application, developed in **Python** using **Streamlit** and **Plotly**, allows users to **simulate and visualize in real time** three fundamental physical systems:  
the **projectile motion**, the **simple pendulum**, and the **mass–spring system**.  

It is designed for **educational and experimental purposes** — enabling users to modify physical parameters (such as velocity, mass, length, angle, damping, etc.) and instantly observe how these changes affect the system’s behavior.

---

## ⚙️ Main Features

- **Accurate Numerical Simulation**  
  Each physical system is modeled from its fundamental differential equations using numerical integration methods such as **Euler** and **Runge–Kutta 4 (RK4)**.  

- **Dynamic and Interactive Visualization**  
  With **Plotly**, the app generates real-time animated graphs showing trajectories, oscillations, and energy variations (kinetic and potential).  

- **Intuitive User Interface**  
  The app is divided into **three tabs**:
  - 🎯 *Projectile Motion*
  - 🕰️ *Pendulum*
  - 🪤 *Mass–Spring System*

  Each tab contains:
  - **sliders** and **numeric inputs** to adjust physical parameters;  
  - a **Run Simulation** button;  
  - and an **Export CSV** button to save simulation results.

- **Data Export and Analysis**  
  Simulation results (positions, velocities, energies, etc.) can be **exported as CSV files** for further analysis in tools such as Excel, Python, or MATLAB.

- **Smooth and Responsive Experience**  
  Parameter changes are reflected instantly in the plots, offering **real-time interactivity** and fluid visualization.

---

## 🧮 Core Equations

### 🚀 Projectile:
\[
\begin{cases}
v_x' = -\frac{k}{m}v_x \\
v_y' = -g - \frac{k}{m}v_y \\
x' = v_x \\
y' = v_y
\end{cases}
\]

### 🕰️ Pendulum:
\[
\theta'' + \frac{g}{L}\sin(\theta) + c\theta' = 0
\]

### 🪤 Mass–Spring System:
\[
m x'' + c x' + k x = 0
\]

---

## 🧩 Technologies Used

- **Python 3.10+**
- **Streamlit** — Interactive web interface
- **NumPy** — Scientific computing
- **Pandas** — Data management
- **Plotly** — Interactive graphs and animations
- **Math** — Trigonometric functions and constants

---

## 🚀 How to Run the Project
```bash
python -m streamlit run app_interactive_physics_lab.py
```

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/fatima-299/interactive-physics-lab.git
cd interactive-physics-lab
