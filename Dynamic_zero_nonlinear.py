import numpy as np
import matplotlib.pyplot as plt

# -------------------- PARAMETERS --------------------
L = 100           # Total length of the cable (m)
M = 200           # Mass of the towed object (kg)
N = 50            # Number of segments
l = L / N         # Length of each segment (m)
g = 9.81          # Gravity (m/s^2)
Vy_0 = 0          # Vertical velocity (m/s)
Ay = 0            # Vertical acceleration (m/s^2)
d = 38 / 1000     # Diameter of cable (m)
rho = 1000        # Fluid density (water) (kg/m^3)
m = 0.072         # Mass per unit length of the cable (kg/m)
w = 1.47          # Weight by buoyancy ratio (unitless)

# Acceleration decay parameters
A0 = 0.51          # Initial horizontal acceleration (m/s^2)
k_decay = 0.5     # Decay rate constant (1/s)
Vx_0 = 0      # Initial horizontal velocity (m/s)

# Buoyancy force per segment
F_B = np.ones(N + 1) * (m * L / w)
F_B[-1] = rho * g * 0.0657  # Adjust for towed object volume (tune if needed)


def Ax_func(t, A0=A0, k=k_decay):
    # Starts from zero acceleration, rises and decays
    return A0 * np.exp(-k * t)  # 0 at t=0, peaks near 1/k, then decays

# def Vx_func(t, V0=Vx_0, A0=A0, k=k_decay):
#     # Velocity is integral of Ax_func
#     # Vx(0) = 0 now
#     return V0 * (1 - np.exp(-k * t))  # Starts from zero, rises to V0



def solve_cable_shape(Vx, Vy, Ax, Ay, theta_init=None, tolerance=1e-5, max_iters=100):
    theta = np.full(N + 1, np.pi/2) if theta_init is None else theta_init.copy()  # Vertical cable at start
    theta_dot = np.zeros(N + 1)
    theta_ddot = np.zeros(N + 1)
    T = np.zeros(N + 1)
    ax = np.zeros(N + 1)
    ay = np.zeros(N + 1)
    vx = np.zeros(N + 1)
    vy = np.zeros(N + 1)
    k_arr = np.ones(N + 1)

    theta_prev = np.ones(N + 1) * 10  # Large initial difference

    for iteration in range(max_iters):
        for i in range(1, N + 1):
            vx[i] = Vx + l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1])
            vy[i] = Vy + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1])
            ax[i] = Ax + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.sin(theta[1:i + 1]) * theta_ddot[1:i + 1])
            ay[i] = Ay - l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.cos(theta[1:i + 1]) * theta_ddot[1:i + 1])

        k_arr[-1] = 0.5 * rho * 0.0324 * 0.004 * abs(np.sqrt(vx[-1]**2 + vy[-1]**2))  # Drag coefficient on last segment
        A_guess = M * g - F_B[-1]
        B_guess = k_arr[-1] * vx[-1]
        T[-1] = np.sqrt(A_guess ** 2 + B_guess ** 2)
        theta[-1] = np.arctan2(A_guess, B_guess)

        for i in range(N - 1, -1, -1):
            Cd = 1.2
            v_i = np.sqrt(vx[i] ** 2 + vy[i] ** 2)
            A_proj = l * d * abs(np.sin(theta[i + 1]))
            k_arr[i] = 0.5 * rho * A_proj * v_i * Cd

            A = T[i + 1] * np.sin(theta[i + 1]) - k_arr[i] * vy[i] - F_B[i] + m * g * l - m * ay[i]
            B = T[i + 1] * np.cos(theta[i + 1]) + k_arr[i] * vx[i] + m * ax[i]

            T[i] = np.sqrt(A ** 2 + B ** 2)
            theta[i] = np.arctan2(A, B)

        if np.linalg.norm(theta - theta_prev) < tolerance:
            break
        theta_prev[:] = theta

    x = [0]
    y = [0]
    for i in range(1, N + 1):
        x.append(x[-1] - l * np.cos(theta[i]))
        y.append(y[-1] + l * np.sin(theta[i]))

    return np.array(x), np.array(y), theta, T


times = [2]

# plt.figure(figsize=(12, 7))
# theta_init = None

# for t in times:
#     Ax_t = Ax_func(t)
#     Vx_t = Vx_func(t)
#     Vy_t = Vy_0 + Ay * t

#     x, y, theta, T = solve_cable_shape(Vx_t, Vy_t, Ax_t, Ay, theta_init)
#     theta_init = theta

#     plt.plot(x, y, marker='o', label=f't = {t} s')

# plt.title('Cable Configurations at Different Times with Decaying Acceleration')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.gca().invert_yaxis()
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# plt.tight_layout()
# plt.show()


from scipy.integrate import cumulative_trapezoid

# Given acceleration function Ax_func(t), create a velocity function by integrating it numerically:

# Define a fine time grid for integration
times_for_integral = np.linspace(0, max(times), 1000)

# Compute acceleration values on this grid
Ax_vals = Ax_func(times_for_integral)

# Integrate acceleration to get velocity (initial velocity = 0)
Vx_vals = cumulative_trapezoid(Ax_vals, times_for_integral, initial=0)

# Integrate velocity to get position (initial position = 0)
X_hel_vals = cumulative_trapezoid(Vx_vals, times_for_integral, initial=0)

# Create interpolation functions for velocity and position at arbitrary t
from scipy.interpolate import interp1d
Vx_interp = interp1d(times_for_integral, Vx_vals, kind='linear', fill_value='extrapolate')
X_hel_interp = interp1d(times_for_integral, X_hel_vals, kind='linear', fill_value='extrapolate')

# Now, in your main loop, use Vx_interp(t) for Vx and X_hel_interp(t) for the cable base position shift:

plt.figure(figsize=(12, 7))
theta_init = None

for t in times:
    Ax_t = Ax_func(t)
    Vx_t = Vx_interp(t)   # velocity from integrated acceleration
    Vy_t = Vy_0 + Ay * t

    x, y, theta, T = solve_cable_shape(Vx_t, Vy_t, Ax_t, Ay, theta_init)
    theta_init = theta

    X_hel = X_hel_interp(t)  # helicopter horizontal position at time t
    x_shifted = x     # shift cable by helicopter position

    plt.plot(x_shifted, y, marker='o', label=f't = {t} s')

plt.title('Cable Configurations Moving with Helicopter Horizontal Displacement')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

# import csv 

# filename = "Dynamic_plot.csv"

# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['x', 'y'])  # Header
#     for i in range(N + 1):
#         writer.writerow([x[i], y[i]])

# print(f"Coordinates saved to {filename}")
