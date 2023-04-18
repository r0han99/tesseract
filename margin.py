import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

plt.style.use('seaborn-whitegrid')

font = {'family': 'poppins',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

# Generate a random dataset with 2 classes
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create a linear SVM classifier
clf = SVC(kernel='linear')

# Fit the classifier to the data
clf.fit(X, y)

# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create a grid of points to evaluate the model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Get the decision function values for the grid points
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.3,
           linestyles=['--', '-', '--'])
ax.set_xlabel('Feature 1',fontdict=font)
ax.set_ylabel('Feature 2',fontdict=font)
plt.title('Margins',fontdict=font)
plt.show()
