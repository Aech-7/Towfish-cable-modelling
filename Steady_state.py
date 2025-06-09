import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100          # Length of the cable
M = 200         # Mass of the towed object
N = 50          # Number of segments
l = L / N       # Length of each segment
g = 9.81        # Gravity
Vx = 1.2      # Helicopter X velocity
Vy = 0          # Helicopter Y velocity  
Ax = 0          # Helicopter X acceleration
Ay = 0          # Helicopter Y acceleration
Xm = 0          # Helicopter X position
Ym = 0          # Helicopter Y position
d = 38/1000    # diameter of cable in m
rho = 1000     # Density of the medium
m = 0.072       # Mass per unit length of the cable
w = 1.47        # Weight by buoyancy ratio

k = np.ones(N+1) * 1  
V = np.pi*((d/2)**2)*l
# F_B = np.ones(N+1)*rho*g*V
F_B = np.ones(N+1)*m*L/w
F_B[-1] = rho*g*0.0657
print(F_B)

theta = np.zeros(N + 1)
theta_dot = np.zeros(N + 1)
theta_ddot = np.zeros(N + 1)
T = np.zeros(N + 1)
ax = np.zeros(N + 1)
ay = np.zeros(N + 1)

vx = np.zeros(N + 1)
vy = np.zeros(N + 1)

for i in range(1, N + 1):
    vx[i] = Vx + l * np.sum(np.sin(theta[1:i+1]) * theta_dot[1:i+1])
    vy[i] = Vy + l * np.sum(np.cos(theta[1:i+1]) * theta_dot[1:i+1])
    ax[i] = Ax + l * np.sum(np.cos(theta[1:i+1]) * theta_dot[1:i+1]**2) + \
                  l * np.sum(np.sin(theta[1:i+1]) * theta_ddot[1:i+1])
    ay[i] = Ay - l * np.sum(np.sin(theta[1:i+1]) * theta_dot[1:i+1]**2) + \
                  l * np.sum(np.cos(theta[1:i+1]) * theta_ddot[1:i+1])

vx_N = vx[N]
vy_N = vy[N]
ax_N = ax[N]
ay_N = ay[N]
k[-1] = 0.5*rho*0.0324*0.004*abs(np.sqrt(vx[-1]**2 + vy[-1]**2))
kN = k[-1]
F_BN = F_B[-1]

TN = np.sqrt((kN * vx_N)**2 + (M * g - kN * vy_N - F_BN)**2)
theta[N] = np.arctan2((M * g - kN * vy_N), (kN * vx_N))
T[N] = TN

for i in range(N - 1, -1, -1):
    Cd = 1.2 
    v_i = np.sqrt(vx[i]**2 + vy[i]**2)
    A_proj = l * d * abs(np.sin(theta[i+1]))  
    k[i] = 0.5*rho*A_proj*v_i
    
    A = T[i + 1] * np.sin(theta[i + 1]) - k[i]*vy[i] - F_B[i] + m*l*g # Vertical component
    B = T[i + 1] * np.cos(theta[i + 1]) + k[i] * vx[i]  # Horizontal + drag
    T[i] = np.sqrt(A**2 + B**2)
    theta[i] = np.arctan2(A, B)

x = np.zeros(N + 1)
y = np.zeros(N + 1)

for i in range(1, N + 1):
    x[i] = x[i - 1] - l * np.cos(theta[i])
    y[i] = y[i - 1] + l * np.sin(theta[i])

x += Xm
y += Ym

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo-', label='Cable') 
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(' Cable Configuration')
plt.grid(True)
plt.legend()
plt.axis('equal') 
plt.gca().invert_yaxis()  


print("Final Tensions:", T)
print("Final Angles (deg):", np.degrees(theta))
print("X positions:", x)
print("Y positions:", y)

# from mpl_toolkits.mplot3d import Axes3D

# # 3D Plot: X, Y vs Tension (T)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(x, y, T, 'bo-', label='Cable with Tension')
# ax.set_xlabel('X Position (m)')
# ax.set_ylabel('Y Position (m)')
# ax.set_zlabel('Tension (N)')
# ax.set_title('3D Plot of Node Positions vs Tension')
# ax.view_init(elev=30, azim=-60)  # Adjust viewing angle if needed
# ax.legend()
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(N + 1), T, 'r-o')
plt.xlabel('Node Index (0 = Helicopter, N = Payload)')
plt.ylabel('Tension (N)')
plt.title('Tension Variation Along the Cable')
plt.grid(True)
plt.show()

