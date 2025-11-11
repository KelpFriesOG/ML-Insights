import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the steady-state activation function
def x_infinity(V, V_half, k):
    return 1 / (1 + np.exp(-(V - V_half) / k))

# Define the membrane potential range
V = np.linspace(-100, 100, 400)

# Streamlit app elements
st.title('Steady-State Activation Function $x_\\infty(V, [Ca^{2+}])$')

# Create sliders for V_half and k
V_half = st.slider('V_half [mV]', min_value=-50, max_value=10, value=-20, step=1)
k = st.slider('Slope factor k:', min_value=1, max_value=30, value=10, step=1)

# Calculate the activation function
x_inf = x_infinity(V, V_half, k)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(V, x_inf, label=f'V_half={V_half}, k={k}')
plt.title('Steady-State Activation Function $x_\\infty(V, [Ca^{2+}])$')
plt.xlabel('Membrane Potential (V) [mV]')
plt.ylabel('$x_\\infty(V, [Ca^{2+}])$')
plt.axhline(0.5, color='gray', linestyle='--')  # Reference line at 0.5
plt.axvline(0, color='black', linestyle='--')   # Reference line at V=0
plt.xlim(-100, 100)
plt.ylim(-0.1, 1.1)
plt.grid()
plt.legend()

# Render the plot in the Streamlit app
st.pyplot(plt)