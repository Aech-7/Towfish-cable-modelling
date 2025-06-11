import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


#--------------Rope1---------------------
L = 460           # Total length of the cable (m)
M = 10025           # Mass of the towed object (kg)
N = 30            # Number of segments
l = L / N         # Length of each segment (m)
g = 9.81          # Gravity (m/s^2)
Vy_0 = 0          # Vertical velocity (m/s)
Ay = 0            # Vertical acceleration (m/s^2)

L_B =325
d1 = 0.04     # Diameter of cable (m)
rho = 1000        # Fluid density (water) (kg/m^3)
# w = 1.47          # Weight by buoyancy ratio (unitless)
rho_c = 3917.39  # Density of cable (kg/m^3)
m1 = rho_c * np.pi * (d1 / 2) ** 2  # Mass per unit length of the cable (kg/m)


#--------------Rope2---------------------
L_F = 135
d2  = 0.05 
rho_c2 = 1900.78
CD2 = 0.15
m2 = rho_c2 * np.pi * (d2 / 2) ** 2
# Acceleration decay parameters        
k = 2.5723/49.5        
Vf=2.5723
# Buoyancy force per segment
Cd = np.ones(N + 1)

d = np.ones(N + 1)   # Diameter of cable for each segment
d[:22] = d1
d[22:] = d2

m = np.ones(N + 1) 
m[:22] = m1 * l
m[22:] = m2 * l

F_B = np.ones(N + 1)  # Buoyant force on each segment
F_B[:22] = rho*g*(np.pi * (d1 / 2) ** 2 * l)
F_B[22:] = rho*g*(np.pi * (d2 / 2) ** 2 * l)
F_B[-1] = rho * g * 6.24  

# Define a function for horizontal acceleration
def Ax_func(t, k=k):
    if t<49.5:
        return k 
    else:
        return 0

#Integrate and define the velocity function or integrate in the code itself
def Vx_func(t, Vf, k=k): 
    if t< 49.5: 
        return k*t 
    else:
        return Vf 

def solve_cable_shape(Vx, Vy, Ax, Ay, theta_init=None, tolerance=1e-5, max_iters=100):
    theta = np.full(N + 1, np.pi/2) if theta_init is None else theta_init.copy() 
    theta_dot = np.zeros(N + 1)
    theta_ddot = np.zeros(N + 1)
    T = np.zeros(N + 1)
    ax = np.zeros(N + 1)
    ay = np.zeros(N + 1)
    vx = np.zeros(N + 1)
    vy = np.zeros(N + 1)
    k_arr = np.ones(N + 1)

    theta_prev = np.ones(N + 1) * 10 

    for iteration in range(max_iters):
        # Non linear terms in the expression of velocity and acceleration and acceleration are approximated to zero
        for i in range(1, N + 1):
            vx[i] = Vx + l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1])
            vy[i] = Vy + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1])
            ax[i] = Ax + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.sin(theta[1:i + 1]) * theta_ddot[1:i + 1])
            ay[i] = Ay - l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.cos(theta[1:i + 1]) * theta_ddot[1:i + 1])

        k_arr[-1] = 0.5 * rho * 0.768 * 2.13 * np.sqrt(vx[-1]**2 + vy[-1]**2)  # Remove abs(), add velocity

        A_N = M*g - M*ay[-1] - k_arr[-1]*vy[-1] - F_B[-1]
        B_N = M*ax[-1] + k_arr[-1]*vx[-1]
        T_N = np.sqrt(A_N ** 2 + B_N ** 2)
        theta[-1] = np.arctan2(A_N, B_N)
        T[-1] = T_N

        for i in range(N - 1, -1, -1):
            Cd[i] = 2 if i < 22 else 0  # Set drag coefficient based on segment
            v_i = np.sqrt(vx[i] ** 2 + vy[i] ** 2)
            A_proj = l * d[i] * abs(np.sin(theta[i + 1]))
            # A_proj = l * d[i] * max(abs(np.sin(theta[i + 1])), 1e-6)

            k_arr[i] = 0.5 * rho * A_proj * v_i * Cd[i]

            A = T[i + 1] * np.sin(theta[i + 1]) - k_arr[i] * vy[i] - F_B[i] + m[i] * g  - m[i] * ay[i]
            B = T[i + 1] * np.cos(theta[i + 1]) + k_arr[i] * vx[i] + m[i] * ax[i]

            T[i] = np.sqrt(A ** 2 + B ** 2)
            theta[i] = np.arctan2(A, B)

        if np.linalg.norm(theta - theta_prev) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
        theta_prev[:] = theta

    x = [0]
    y = [0]
    for i in range(1, N + 1):
        x.append(x[-1] - l * np.cos(theta[i]))
        y.append(y[-1] + l * np.sin(theta[i]))

    return np.array(x), np.array(y), theta, T


times = [400]  # Time points to evaluate the cable shape

# Integrate the acceleration to get velocity and position
# times_for_integral = np.linspace(0, max(times), 1000)
# Ax_vals = Ax_func(times_for_integral)
# Vx_vals = cumulative_trapezoid(Ax_vals, times_for_integral, initial=0)
# X_vals = cumulative_trapezoid(Vx_vals, times_for_integral, initial=0)
# Vx_interp = interp1d(times_for_integral, Vx_vals, kind='linear', fill_value='extrapolate')
# X_interp = interp1d(times_for_integral, X_vals, kind='linear', fill_value='extrapolate')

def X_interp(t):
    t_accel = 49.5
    if t < t_accel:
        return 0.5 * k * t**2
    else:
        return 0.5 * k * t_accel**2 + Vf * (t - t_accel)
    
plt.figure(figsize=(12, 7))
theta_init = None

for t in times:
    Ax_t = Ax_func(t)
    Vx_t = Vx_func(t,Vf,k)   # velocity from integrated acceleration
    Vy_t = Vy_0 + Ay * t

    x, y, theta, T = solve_cable_shape(Vx_t, Vy_t, Ax_t, Ay, theta_init)
    theta_init = theta

    X_t = X_interp(t)  # horizontal position at time t
    x_shifted = x + X_t   

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

# filename = "Dynamic_COMPARISON.csv"

# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['x_shifted', 'y'])  # Header
#     for i in range(N + 1):
#         writer.writerow([x_shifted[i], y[i]])

# print(f"Coordinates saved to {filename}")
