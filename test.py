import matplotlib.pyplot as plt
import numpy as np

# Create a 2x2 grid of subplots and set the figure size
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Generate sample data
x = np.linspace(0, 5, 100)
y1 = x
y2 = x**2
y3 = x**3
y4 = np.sin(x)

# Plot the data in each subplot
ax[0].plot(x, y3, 'b-', label='Cubic')
ax[1].plot(x, y4, 'm-', label='Sine')

# Customize the appearance of each subplot
for j in range(2):
    ax[j].grid(True)
    ax[j].legend()

# Add a title for the entire figure
fig.suptitle('Example of Subplots')

# Display the figure
plt.show()
plt.savefig("plot.png")