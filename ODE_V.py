import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL AND NUMERICAL PARAMETERS
# =============================================================================
# Cable Properties
L = 100.0           # Unstretched cable length (m)
N = 50              # Number of segments
EA = 1.5e6          # Axial stiffness (N)
mu = 0.177          # Mass per unit length in air (kg/m)
d = 0.038           # Cable diameter (m)

# Environment Properties
rho_water = 1025    # Water density (kg/m^3)
g = 9.81            # Gravity (m/s^2)
Cd_cable = 1.2      # Drag coefficient for cable

# Towfish Properties
M_towfish = 200.0   # Mass of towfish (kg)
V_towfish = 0.0657  # Volume of towfish (m^3)
Cd_towfish = 0.8    # Drag coefficient for towfish
A_towfish = 0.1     # Projected area of towfish (m^2)

# Derived Parameters
num_nodes = N + 1
ds = L / N          # Unstretched length of each segment
mass_node = mu * ds # Physical mass of each cable node
A_cable = np.pi * (d/2)**2 # Cross-sectional area of cable

# CORRECT DEFINITION for towfish buoyancy
F_buoyancy_towfish = rho_water * g * V_towfish

# --- Added Mass Calculation ---
# Added mass for a cylinder accelerating perpendicular to its axis
mass_added_cable = rho_water * A_cable * ds
# Total inertial mass for a cable node
mass_inertial_node = mass_node + mass_added_cable
# Estimated inertial mass for the towfish
mass_inertial_towfish = M_towfish + (rho_water * V_towfish * 0.5)

# Simulation Control
# --- STABILITY FIX: Calculate dt based on CFL condition ---
v_wave = np.sqrt(EA / mu)
dt_cfl = ds / v_wave
dt = dt_cfl * 0.8  # Use 80% of the CFL limit for safety
total_time = 100  # Total simulation time (s)
steps = int(total_time / dt)
print(f"Wave speed in cable: {v_wave:.1f} m/s")
print(f"CFL Time Step Limit: {dt_cfl:.6f} s")
print(f"Using Time Step: {dt:.6f} s")

# Ship Motion (Top Boundary Condition)
def ship_velocity(t):
    if t < 10.0:
        return 0.2 * t
    else:
        return 2.0

def ship_position_x(t):
    if t < 10.0:
        return 0.1 * t**2
    else:
        return 10.0 + 2.0 * (t - 10)

# =============================================================================
# 2. INITIALIZATION OF STATE ARRAYS
# =============================================================================
r = np.zeros((num_nodes, 2))
r[:, 1] = np.linspace(0, -L, num_nodes)
v = np.zeros((num_nodes, 2))
a = np.zeros((num_nodes, 2))

# Storage for plotting results
plot_interval = int(0.5 / dt)
history = []

# =============================================================================
# 3. MAIN SIMULATION LOOP (using Velocity Verlet)
# =============================================================================
for step in range(steps):
    if step % plot_interval == 0:
        history.append(r.copy())

    # --- Velocity Verlet: First half-step for velocity ---
    v += 0.5 * a * dt

    # --- Velocity Verlet: Full-step for position ---
    r += v * dt

    # --- Calculate Forces at New Positions ---
    F_net = np.zeros((num_nodes, 2))

    # Loop through ALL nodes that are not fixed by the ship (from 1 to N)
    for i in range(1, num_nodes):
        # --- Common Forces ---
        # Drag Force
        v_mag = np.linalg.norm(v[i])
        if v_mag > 1e-6:
            if i < num_nodes - 1: # Cable Node Drag
                F_drag = -0.5 * Cd_cable * rho_water * d * ds * v_mag * v[i]
            else: # Towfish Drag
                F_drag = -0.5 * Cd_towfish * rho_water * A_towfish * v_mag * v[i]
        else:
            F_drag = np.zeros(2)

        # Tension Force from the segment to the left (connecting to node i-1)
        vec_left = r[i] - r[i-1]
        len_left = np.linalg.norm(vec_left)
        unit_vec_left = vec_left / len_left
        strain_left = (len_left - ds) / ds
        T_left_mag = EA * strain_left if strain_left > 0 else 0
        F_tension_left = -T_left_mag * unit_vec_left

        # --- Forces specific to node type ---
        if i < num_nodes - 1: # --- This is an Internal Cable Node ---
            # Effective Weight (Gravity - Buoyancy)
            F_buoyancy_segment = rho_water * A_cable * ds * g
            F_eff_weight = np.array([0, -(mass_node * g - F_buoyancy_segment)])

            # Tension from the segment to the right
            vec_right = r[i+1] - r[i]
            len_right = np.linalg.norm(vec_right)
            unit_vec_right = vec_right / len_right
            strain_right = (len_right - ds) / ds
            T_right_mag = EA * strain_right if strain_right > 0 else 0
            F_tension_right = T_right_mag * unit_vec_right

            F_net[i] = F_eff_weight + F_drag + F_tension_left + F_tension_right
        else: # --- This is the Towfish Node ---
            F_eff_weight_towfish = np.array([0, -(M_towfish * g - F_buoyancy_towfish)])
            # The towfish only has tension from the left
            F_net[i] = F_eff_weight_towfish + F_drag + F_tension_left

    # --- Calculate new accelerations ---
    a[1:-1] = F_net[1:-1] / mass_inertial_node
    a[-1] = F_net[-1] / mass_inertial_towfish

    # --- Velocity Verlet: Second half-step for velocity ---
    v += 0.5 * a * dt

    # --- Apply Top Boundary Condition ---
    t_current = (step + 1) * dt
    r[0, 0] = ship_position_x(t_current)
    r[0, 1] = 0
    v[0, 0] = ship_velocity(t_current)
    v[0, 1] = 0

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 12))
history.append(r.copy()) # Add final state
num_plots = min(10, len(history))
plot_indices = np.linspace(0, len(history) - 1, num_plots, dtype=int)

for i, idx in enumerate(plot_indices):
    r_plot = history[idx]
    time_plot = idx * plot_interval * dt if idx < len(history)-1 else total_time
    plt.plot(r_plot[:, 0], r_plot[:, 1], 'o-', markersize=3, label=f't = {time_plot:.1f} s')

plt.title("Dynamic Simulation of Towed Cable (Numerically Stable and Corrected)")
plt.xlabel("Horizontal Position (m)")
plt.ylabel("Depth (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

