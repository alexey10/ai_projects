# Re-importing necessary libraries after code execution state reset
import matplotlib.pyplot as plt
import numpy as np

# Define input range
x = np.linspace(-10, 10, 400)

# Activation functions
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
softmax_input = np.vstack([x, x - 1, x - 2])
softmax = np.exp(softmax_input) / np.sum(np.exp(softmax_input), axis=0)

# Plotting
plt.figure(figsize=(12, 8))

# ReLU
plt.subplot(2, 2, 1)
plt.plot(x, relu, label='ReLU', color='blue')
plt.title('ReLU Activation')
plt.grid(True)

# Sigmoid
plt.subplot(2, 2, 2)
plt.plot(x, sigmoid, label='Sigmoid', color='green')
plt.title('Sigmoid Activation')
plt.grid(True)

# Tanh
plt.subplot(2, 2, 3)
plt.plot(x, tanh, label='Tanh', color='purple')
plt.title('Tanh Activation')
plt.grid(True)

# Softmax (3-class example)
plt.subplot(2, 2, 4)
plt.plot(x, softmax[0], label='Class 1')
plt.plot(x, softmax[1], label='Class 2')
plt.plot(x, softmax[2], label='Class 3')
plt.title('Softmax Activation (3-class)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

