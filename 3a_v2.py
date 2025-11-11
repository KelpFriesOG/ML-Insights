import numpy as np
import matplotlib.pyplot as plt

# --- 0. EPSP METRICS FUNCTION (Adapted from message.txt) ---
def epsp_metrics(t, V, t0_ms, win_ms=30.0, spike_times=None, v0=None):
    """
    Compute subthreshold EPSP peak and 20–80% rise time within a window after t0_ms.
    If a spike occurs before the window ends, truncate at the spike.
    """
    dt = t[1] - t[0]
    idx_t0 = np.argmin(np.abs(t - t0_ms))
    
    # 1. Estimate baseline (v0)
    if v0 is None:
        # Use median in a short pre-stimulus window (t0_ms-5ms to t0_ms)
        idx_pre_start = np.argmin(np.abs(t - (t0_ms - 5.0)))
        v0 = np.median(V[idx_pre_start:idx_t0]) 

    # 2. Define measurement window
    idx_end = np.argmin(np.abs(t - (t0_ms + win_ms)))
    
    # 3. Handle spikes: truncate the analysis window at the first spike
    if spike_times is not None and len(spike_times) > 0:
        first_spike_time = spike_times[0]
        idx_spike = np.argmin(np.abs(t - first_spike_time))
        idx_end = min(idx_end, idx_spike) 

    V_window = V[idx_t0:idx_end]
    t_window = t[idx_t0:idx_end]
    
    if len(V_window) == 0:
        return np.nan, np.nan, np.nan

    # Baseline-subtracted EPSP
    Vw = V_window - v0
    
    # 4. Find peak amplitude and its time
    peak_val = np.max(Vw)
    if peak_val <= 0:
        # No depolarizing EPSP
        return np.nan, np.nan, np.nan
    
    # Peak time is relative to t=0
    idx_peak_in_window = np.argmax(Vw)
    peak_time_abs = t_window[idx_peak_in_window] 

    # 5. Compute 20–80% rise time
    level_20 = 0.2 * peak_val
    level_80 = 0.8 * peak_val
    
    try:
        # Find first crossing indices 
        idx_t20_in_window = np.where(Vw >= level_20)[0][0]
        idx_t80_in_window = np.where(Vw >= level_80)[0][0]
        
        # Calculate times
        t20 = t_window[idx_t20_in_window]
        t80 = t_window[idx_t80_in_window]
        
        rise_time = t80 - t20
    except IndexError:
        # No crossing to 20%/80% found
        rise_time = np.nan
        
    # Return peak (mV), peak time (ms), rise time (ms)
    return peak_val, peak_time_abs, rise_time

# Simulation Parameters
R_m = 20000  # Specific membrane resistance
R_a = 150    # Specific axial resistance
d = 1.0e-4   # Dendrite diameter
C_m = 1.0    # Specific membrane capacitance

# Cable Properties
lambda_sq = (R_m * d) / (4 * R_a)
lambda_ = np.sqrt(lambda_sq)       
tau_m = R_m * C_m                

# Synaptic Parameters
A = 1000   # Peak synaptic current amplitude (pA)
tau_syn = 0.5 # Synaptic current time constant (ms)
t_start = 5.0 # Start time of the synapse (ms)

# Simulation Time Setup
t_stop = 50.0 
dt = 0.05    
t = np.arange(0, t_stop, dt)
N = len(t)

# Exponential Synaptic Current I_syn(t)
def I_syn(t, t_start, tau_syn, A):
    """Exponential synaptic current in pA."""
    i = np.where(t >= t_start, (A * np.exp(-(t - t_start) / tau_syn)), 0)
    return i * 1e-12 # Convert pA to A for current injection

I_t = I_syn(t, t_start, tau_syn, A)

# Passive Cable Response Function
def Transfer_Impedance(x, lambda_, tau_m):
    """Simplified steady-state transfer impedance Z_N(x) from x to 0."""
    R_inf = np.sqrt(R_a * R_m) / (np.pi * d**2) 
    Z_N = R_inf * np.exp(-x / lambda_)
    return Z_N * 1e3 # Convert V/A to mV/nA

def Voltage_Response(I_t, Z, tau_eff):
    """Simplified voltage response by filtering the current and scaling."""
    I_scaled = I_t * Z * 1e-6 
    kernel = np.exp(-t / tau_eff) * (dt / tau_eff)
    V_out = np.convolve(I_scaled, kernel, mode='full')[:N]
    return V_out

# Define input locations and effective time constants
x_proximal = 0.0 * lambda_ 
x_distal = 2.0 * lambda_   

Z_proximal = Transfer_Impedance(x_proximal, lambda_, tau_m) 
Z_distal = Transfer_Impedance(x_distal, lambda_, tau_m)   

tau_eff_proximal = tau_m * 1.05
tau_eff_distal = tau_m * 1.5

# Calculate EPSPs
V_proximal = Voltage_Response(I_t, Z_proximal, tau_eff_proximal)
V_distal = Voltage_Response(I_t, Z_distal, tau_eff_distal)


# Analysis and measurementss

# Use 25ms window, no spikes in this passive model
p_prox, tp_prox, rt_prox = epsp_metrics(t, V_proximal, t0_ms=t_start, win_ms=25.0, spike_times=[])
p_dist, tp_dist, rt_dist = epsp_metrics(t, V_distal, t0_ms=t_start, win_ms=25.0, spike_times=[])

V_proximal_peak = p_prox
V_distal_peak = p_dist

attenuation_ratio = V_distal_peak / V_proximal_peak

print("\n--- Model Properties ---")
print(f"Length Constant (lambda): {lambda_ * 1e4} µm")
print(f"Membrane Time Constant (tau_m): {tau_m} ms")

print("\n--- Somatic EPSP Measurements ---")
print(f"Somatic EPSP Peak (Proximal): {V_proximal_peak} mV")
print(f"20-80% Rise Time (Proximal): {rt_prox} ms")
print("-" * 35)
print(f"Somatic EPSP Peak (Distal):   {V_distal_peak} mV")
print(f"20-80% Rise Time (Distal):   {rt_dist} ms")
print("\n--- Attenuation and Slowing ---")
print(f"Attenuation Ratio (Distal/Proximal): {attenuation_ratio}")
print(f"Kinetic Slowing Factor (Distal/Proximal): {rt_dist / rt_prox}x")


# Final Plot
plt.figure(figsize=(10, 6))
plt.plot(t, V_proximal, label=f'Proximal EPSP (Peak: {V_proximal_peak} mV)', linewidth=2, color='darkgreen')
plt.plot(t, V_distal, label=f'Distal EPSP (Peak: {V_distal_peak} mV)', linewidth=2, color='firebrick')

plt.title('Proximal vs. Distal EPSPs (Attenuation & Kinetics)')
plt.xlabel('Time (ms)')
plt.ylabel('Somatic Voltage (mV)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 30)
plt.axhline(0, color='gray', linestyle='-')
plt.show()