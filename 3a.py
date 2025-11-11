import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
R_m = 20000  # Specific membrane resistance
R_a = 150    # Specific axial resistance
d = 1.0e-4   # Dendrite diameter
C_m = 1.0    # Specific membrane capacitance

# Cable Properties
lambda_sq = (R_m * d) / (4 * R_a) # Square of the length constant (lambda^2)
lambda_ = np.sqrt(lambda_sq)       # Length constant
tau_m = R_m * C_m                # Membrane time constant

print(f"Length Constant (lambda): {lambda_ * 1e4:.2f} µm")
print(f"Membrane Time Constant (tau_m): {tau_m:.2f} ms")

# Synaptic Parameters
A = 1000   # Peak synaptic current amplitude (pA)
tau_syn = 0.5 # Synaptic current time constant (ms)
t_start = 5.0 # Start time of the synapse (ms)

# Simulation Time Setup
t_stop = 50.0 # Total simulation time (ms)
dt = 0.05    # Time step (ms)
t = np.arange(0, t_stop, dt)
N = len(t)

# Exponential Synaptic Current I_syn(t)
def I_syn(t, t_start, tau_syn, A):
    """Exponential synaptic current in pA."""
    i = np.where(t >= t_start, (A * np.exp(-(t - t_start) / tau_syn)), 0)
    return i * 1e-12 # Convert pA to A for current injection

I_t = I_syn(t, t_start, tau_syn, A)

# Passive Cable Response Function
# This is the transfer impedance from location 'x' to the soma (x=0)
def Transfer_Impedance(x, lambda_, tau_m):
    """
    Simplified steady-state transfer impedance Z_N(x) from x to 0.
    Scaling factor to convert injected current I to somatic voltage V.
    """
    R_inf = np.sqrt(R_a * R_m) / (np.pi * d**2) # Characteristic resistance of a semi-infinite cable
    Z_N = R_inf * np.exp(-x / lambda_)
    return Z_N * 1e3 # Convert V/A to mV/nA

# I use use an approximation
# based on the steady-state transfer function for the peak amplitude, and
# an effective time constant for the kinetics.

# Define input locations
x_proximal = 0.0 * lambda_ # Proximal: at the soma (x=0)
x_distal = 2.0 * lambda_   # Distal: 2 length constants away

# Calculate effective transfer impedance and time constant at soma
Z_proximal = Transfer_Impedance(x_proximal, lambda_, tau_m) # mV/nA
Z_distal = Transfer_Impedance(x_distal, lambda_, tau_m)   # mV/nA

# The kinetic slowing (filtering) effect can be modeled by an effective time constant:
# tau_eff = tau_m / (1 - (R_a * d * lambda / 4 * R_m) * (1 / lambda))  (complex transient analysis is needed for a true model)
# For demonstration: the EPSP at the soma is convolved with the kernel h(t) = exp(-t/tau_m)
h_t = (1 / tau_m) * np.exp(-t / tau_m)

# The full transient response is complex so this is a first-order filter.
def Voltage_Response(I_t, Z, tau_eff):
    """Simplified voltage response by filtering the current and scaling."""
    # Scale current by impedance (assuming an effective I -> V conversion)
    I_scaled = I_t * Z * 1e-6 # (A * mV/nA * 1e-6 nA/pA) -> mV (Rough scaling)

    # Use a simple convolution for the filtering effect
    kernel = np.exp(-t / tau_eff) * (dt / tau_eff)
    V_out = np.convolve(I_scaled, kernel, mode='full')[:N]
    return V_out

# Define effective time constants (This is an approximation of the filtering)
# Proximal is near the membrane time constant (tau_m)
tau_eff_proximal = tau_m * 1.05
# Distal input is slower (filtered)
tau_eff_distal = tau_m * 1.5

# Calculate EPSPs
V_proximal = Voltage_Response(I_t, Z_proximal, tau_eff_proximal)
V_distal = Voltage_Response(I_t, Z_distal, tau_eff_distal)

# --- 4. Analysis and Measurements ---
V_proximal_peak = np.max(V_proximal)
V_distal_peak = np.max(V_distal)

attenuation_ratio = V_distal_peak / V_proximal_peak

# Function to calculate 20-80% rise time
def calculate_rise_time(V, t):
    peak = np.max(V)
    if peak == 0: return np.nan

    v20 = 0.2 * peak
    v80 = 0.8 * peak

    # Find the time when voltage crosses the 20% and 80% thresholds
    try:
        t_20_idx = np.where(V >= v20)[0][0]
        t_80_idx = np.where(V >= v80)[0][0]
        rise_time = t[t_80_idx] - t[t_20_idx]
        return rise_time
    except IndexError:
        return np.nan

rise_time_proximal = calculate_rise_time(V_proximal, t)
rise_time_distal = calculate_rise_time(V_distal, t)

print("\n--- Measurements ---")
print(f"Somatic EPSP Peak (Proximal): {V_proximal_peak} mV")
print(f"Somatic EPSP Peak (Distal):   {V_distal_peak} mV")
print(f"Attenuation Ratio (Distal/Proximal): {attenuation_ratio}")
print(f"20-80% Rise Time (Proximal): {rise_time_proximal} ms")
print(f"20-80% Rise Time (Distal):   {rise_time_distal} ms")
print("\n--- Observation ---")
print(f"Distal input is attenuated by a factor of {1/attenuation_ratio} and is {rise_time_distal / rise_time_proximal}x slower.")


# --- 5. Plotting the Results ---
plt.figure(figsize=(10, 6))
plt.plot(t, V_proximal, label='Proximal EPSP (Low Attenuation, Fast Kinetics)', linewidth=2, color='darkgreen')
plt.plot(t, V_distal, label='Distal EPSP (High Attenuation, Slow Kinetics)', linewidth=2, color='firebrick')

plt.title('Proximal vs. Distal EPSPs (Attenuation & Kinetics)')
plt.xlabel('Time (ms)')
plt.ylabel('Somatic Voltage (mV)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 30)
plt.axhline(0, color='gray', linestyle='-')
plt.text(25, V_proximal_peak * 0.9, f'Proximal Peak: {V_proximal_peak} mV', color='darkgreen')
plt.text(25, V_distal_peak * 0.9, f'Distal Peak: {V_distal_peak} mV', color='firebrick')
plt.show()

print(V_distal)