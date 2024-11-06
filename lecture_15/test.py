import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import imageio

# Generate a dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate TSS, ESS, and RSS
y_mean = np.mean(y)
TSS = np.sum((y - y_mean) ** 2)
ESS = np.sum((y_pred - y_mean) ** 2)
RSS = np.sum((y - y_pred) ** 2)

# Create a series of plots
filenames = []
for i in range(101):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data points')
    ax.plot(X, y_pred, color='red', label='Regression line')
    
    # Plot TSS
    if i <= 33:
        for j in range(len(X)):
            ax.plot([X[j, 0], X[j, 0]], [y[j, 0], y_mean], color='green', alpha=0.5)
        ax.set_title(f'Total Sum of Squares (TSS)\nTSS = {TSS:.2f}')
    
    # Plot ESS
    elif i <= 66:
        for j in range(len(X)):
            ax.plot([X[j, 0], X[j, 0]], [y_pred[j, 0], y_mean], color='orange', alpha=0.5)
        ax.set_title(f'Explained Sum of Squares (ESS)\nESS = {ESS:.2f}')
    
    # Plot RSS
    else:
        for j in range(len(X)):
            ax.plot([X[j, 0], X[j, 0]], [y[j, 0], y_pred[j, 0]], color='purple', alpha=0.5)
        ax.set_title(f'Residual Sum of Squares (RSS)\nRSS = {RSS:.2f}')
    
    ax.legend()
    filename = f'plot_{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

# Create a GIF
with imageio.get_writer('sum_of_squares.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove the individual plot files
import os
for filename in filenames:
    os.remove(filename)

print("GIF created successfully!")