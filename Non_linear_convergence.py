import numpy as np
import matplotlib.pyplot as plt
from Dynamic_zero_nonlinear import solve_cable_shape
from scipy.interpolate import CubicSpline

L = 100           # Total length of the cable (m)
M = 200           # Mass of the towed object (kg)   \
N = 50            # Number of segments
l = L / N         # Length of each segment (m)
g = 9.81          # Gravity (m/s^2)
d = 38 / 1000     # Diameter of cable (m)
rho = 1000        # Fluid density (water) (kg/m^3)  
m = 0.072         # Mass per unit length of the cable (kg/m)
w = 1.47        # Weight by buoyancy ratio

def Ax_func(t):
    return 0.51 *np.exp(-0.5 * t)

def Vx_func(t):
    return 1.02 * (1 - np.exp(-0.5 * t)) 

Vy = 0
Ay =0 

# Buoyancy force per segment
k = np.ones(N+1) * 1   # Drag coefficient on each segment
V = np.pi*((d/2)**2)*l  # Volume of each segment
# F_B = np.ones(N+1)*rho*V*g

#Buoyant force on towfish
F_B = np.ones(N+1)*(m*L/w)
F_B[-1] = rho*g*0.0657

#Initialize arrays
t = 1  # Total simulation time in seconds
theta = np.full(N + 1, np.pi/2)
dt = 0.01
time = 0
tolerance = 1e-5
max_iters = 100 
theta_prev = theta.copy()
theta_dot_prev = np.zeros(N + 1)  # Initialize theta_dot_prev to zero
T = np.zeros(N + 1)
theta_dot = np.zeros(N + 1)
theta_ddot = np.zeros(N + 1)
Omega=[0]
alpha=[0]
VEL = [0]
theta_history = []

while time < t:  # Simulate for 10 seconds
    # theta_assumed = solve_cable_shape(Vx_func(time),Vy, Ax_func(time), Ay, theta_init=None, tolerance=1e-5, max_iters=100)  # Assume vertical cable at start
    _, _, theta_assumed, _ = solve_cable_shape(Vx_func(time), Vy, Ax_func(time), Ay, theta_init=None, tolerance=1e-5, max_iters=100)

    iterations = 0
    while iterations < max_iters:
        Vx = Vx_func(time)
        Ax = Ax_func(time)
        
        if len(theta_history) >= 5:
            times = np.linspace(time - dt*(len(theta_history)-1), time, len(theta_history))
            theta_array = np.array(theta_history).T  # shape: (N+1, time_steps)


            for i in range(N+1):
                cs = CubicSpline(times, theta_array[i])
                theta_dot[i] = cs.derivative(1)(time)
                theta_ddot[i] = cs.derivative(2)(time)
        else:
            theta_dot = (theta - theta_prev) / dt
            print(theta_dot)
            theta_ddot = (theta_dot - theta_dot_prev) / dt
        Vx = Vx_func(time)
        Ax = Ax_func(time)
        vx = np.zeros(N + 1)
        vy = np.zeros(N + 1)
        ax = np.zeros(N + 1)
        ay = np.zeros(N + 1)
        for i in range(1, N + 1):
            vx[i] = Vx + l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1])
            vy[i] = Vy + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1])
            ax[i] = Ax + l * np.sum(np.cos(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.sin(theta[1:i + 1]) * theta_ddot[1:i + 1])
            ay[i] = Ay - l * np.sum(np.sin(theta[1:i + 1]) * theta_dot[1:i + 1] ** 2) + \
                    l * np.sum(np.cos(theta[1:i + 1]) * theta_ddot[1:i + 1])

        #Solve for Nth node
        k[-1] = 0.5*rho*0.0324*0.004 * abs(np.sqrt(vx[-1]**2 + vy[-1]**2))
        A_N = M*g - M*ay[-1] - k[-1]*vy[-1] - F_B[-1]
        B_N = M*ax[-1] + k[-1]*vx[-1]
        T_N = np.sqrt(A_N ** 2 + B_N ** 2)
        theta[-1] = np.arctan2(A_N, B_N)
        T[-1] = T_N
        
        #Iterate backwards to solve for other nodes
        
        for i in range(N - 1, -1, -1):
            Cd = 1.2
            v_i = np.sqrt(vx[i]**2 + vy[i]**2)
            A_proj = l * d * abs(np.sin(theta_assumed[i + 1]))
            k[i] = 0.5 * rho * A_proj * v_i * Cd

            A = T[i + 1] * np.sin(theta_assumed[i + 1]) - k[i] * vy[i] - F_B[i] + m * g * l - m * l * ay[i]
            B = T[i + 1] * np.cos(theta_assumed[i + 1]) + k[i] * vx[i] + m * l * ax[i]
            
            T[i] = np.sqrt(A ** 2 + B ** 2)
            theta[i] = np.arctan2(A, B)
            
        # Check convergence
        if np.linalg.norm(theta - theta_assumed) < tolerance:
            print(f"Converged in {iterations} iterations at time {time:.2f} seconds")
            break
        
        # Instead of: theta_assumed = theta.copy()
        relaxation_factor = 0.5  # Start with 0.5, adjust as needed
        theta_assumed = relaxation_factor * theta + (1 - relaxation_factor) * theta_assumed

        iterations += 1
    theta_history.append(theta.copy())
    VEL.append(time)
    Omega.append(theta_dot[-1])
    alpha.append(theta_ddot[-1])
    theta_dot_prev = theta_dot.copy()
    theta_prev = theta.copy()
    time += dt
    
    
# Calculate x and y positions
x = np.zeros(N + 1)
y = np.zeros(N + 1)
for i in range(1, N + 1):
    x[i] = x[i - 1] - l * np.cos(theta[i])
    y[i] = y[i - 1] + l * np.sin(theta[i])
    
# Adjust positions based on initial offsets
Xm = 0.0  # Initial x offset        
Ym = 0.0  # Initial y offset
x += Xm
y += Ym

# Plot the cable shape
plt.figure(figsize=(12, 7))
plt.plot(x, y, label='Cable Shape', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Cable Shape at t = {:.2f} seconds'.format(t))
plt.grid(True)  
plt.gca().invert_yaxis()  # Invert y-axis to match cable orientation
plt.axis('equal')  # Equal scaling for x and y axes
plt.legend()
plt.show()

