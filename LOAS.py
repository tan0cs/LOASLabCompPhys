import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the system of ODEs
def derivatives(t, y, m, gamma, k):
    x, v = y
    dxdt = v
    dvdt = - (gamma / m) * v - (k / m) * x
    return np.array([dxdt, dvdt])

# Runge-Kutta 4th Order Method
def runge_kutta_4(f, y0, t0, tf, dt, m, gamma, k):
    t_values = np.arange(t0, tf + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        k1 = f(t, y, m, gamma, k)
        k2 = f(t + dt/2, y + dt/2 * k1, m, gamma, k)
        k3 = f(t + dt/2, y + dt/2 * k2, m, gamma, k)
        k4 = f(t + dt, y + dt * k3, m, gamma, k)
        y_values[i] = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t_values, y_values

# Parameters (you can change these)
m = 0.24902       # mass
gammaair = 0.045  # damping coefficient
gammawater = 0.6
k = 10.503169100419       # spring constant

# Initial conditions
x0air = -0.1251      # initial position
x0water = -0.1
v0 = 0.0      # initial velocity

# Time setup
t0 = 0.0
tf = 27.0
tfwater = 8.0
dt = 0.01

# Solve the ODE
t_vals, y_valsair = runge_kutta_4(derivatives, [x0air, v0], t0, tf, dt, m, gammaair, k)
x_vals = y_valsair[:, 0]
t_valswater, y_valswater = runge_kutta_4(derivatives, [x0water, v0], t0, tfwater, dt, m, gammawater, k)
x_valswater = y_valswater[:, 0]

data = pd.read_csv('DATA.csv')


data.plot.scatter(x='Time1',y="Air",color='red', label='Air Data')
plt.title('Scatter plot of Time vs Position in Air')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.show()

data.plot.scatter(x='Time2',y="Water",color='green', label='Water Data')
plt.title('Scatter plot of Time vs Position in Water')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.show()

data.plot.scatter(x='Time1',y="Air",color='red', label='Air Data')
plt.plot(t_vals+3.7, x_vals, label='x(t)', color='blue')
plt.title('Scatter plot of Time vs Position in Air')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.show()

data.plot.scatter(x='Time2',y="Water",color='green', label='Water Data')
plt.plot(t_valswater+2.6, x_valswater, label='x(t)', color='blue')
plt.title('Scatter plot of Time vs Position in Water')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.show()


# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(t_vals, x_vals, label='x(t)', color='blue')
plt.title('Damped Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Position (x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_valswater, x_valswater, label='x(t)', color='green')
plt.title('Damped Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Position (x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
