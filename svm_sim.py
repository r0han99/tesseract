import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

# Matplotlib colormap selection dropdown
# color_maps_list = ('BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu')

# Streamlit UI
col1, col2 = st.columns(2)

kernel_type = col1.selectbox("Kernel type", ["Linear", "RBF"])

col1.markdown('Kernel: ' + kernel_type)

# Plotting area
plothole = col1.empty()

color_map = "Blues"

col2.markdown("Data points distribution:")
n_samples = col2.slider("Samples", 2, 100)
noise = col2.slider("Noise", 0.01, 1.00)

X, y = make_circles(n_samples=n_samples,noise=noise, random_state=42)

if kernel_type == "Linear":
    # Linear kernel
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", SVC(kernel="linear", C=10, random_state=42))])
else:
    # RBF kernel
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=0.1, random_state=42))])

svm_clf.fit(X, y)

# Prediction and decision function
x0s = np.linspace(-1.5, 2.5, 100)
x1s = np.linspace(-1.0, 1.5, 100)
x0, x1 = np.meshgrid(x0s, x1s)
X_concat = np.c_[x0.ravel(), x1.ravel()]
y_prediction = svm_clf.predict(X_concat).reshape(x0.shape)
y_decision = svm_clf.decision_function(X_concat).reshape(x0.shape)

# Contour alpha value sliders
col2.markdown("Contour alpha sliders:")
alpha_prediction = col2.slider("Prediction", 0.0, 1.0)
alpha_decision = col2.slider("Decision Boundary", 0.0, 1.0, value=0.26)

# Plotting section
fig = plt.figure(figsize=(10,10))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bo")
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "rs")
plt.axis([-1.5, 2.5, -1, 1.5])
plt.grid(True, which='both')
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.contourf(x0, x1, y_prediction, cmap=color_map, alpha=alpha_prediction)
plt.contourf(x0, x1, y_decision, cmap=color_map, alpha=alpha_decision)

# Streamlit pyplot show
plothole.pyplot(fig)
