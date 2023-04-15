import streamlit as st
import datetime
import time
import base64
from pathlib import Path
import plotly.express as px

import plotly.figure_factory as ff


import numpy as np
from scipy.spatial import Delaunay



import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


category = st.selectbox('Select Sandbox Category', ['Select a Category','Support Vector Machines','Neural Networks',],key='category')
st.markdown('***')

if category == 'Neural Networks':
    st.markdown('''<center><h3>Neural Network Simulation</h3></center>''', unsafe_allow_html=True)
    
    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
    animation_frame="year", animation_group="country", range_y=[0,4000000000])
    
    st.plotly_chart(fig)

elif category == 'Support Vector Machines':
    st.markdown('''<center><h3>Support Vector Machines<sub>  by <a href="https://medium.com/@sinchan.s/support-vector-machine-svm-in-action-using-streamlit-e3bc56208a85">shinchan</a></sub></h3></center>''', unsafe_allow_html=True)
    st.markdown("***")

    

    
    # matplotlib colormap selection dropdown
    color_maps_list = ('Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',    'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r')
   
    col1, col2 = st.columns(2)
    col1.markdown('Linear Kernel')
    plothole = col1.empty()
    
    color_map = col1.selectbox("countour map", color_maps_list)

    # sample size & noise control sliders
    col2.markdown("Data points distribution:")
    n_samples = col2.slider("Samples", 2, 100)
    noise = col2.slider("Noise", 0.01, 1.00)

    # x & y variables assignment
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # prediction pipeline
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))])
    polynomial_svm_clf.fit(X, y)

    # prediction & decision function
    x0s = np.linspace(-1.5, 2.5, 100)
    x1s = np.linspace(-1.0, 1.5, 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_concat = np.c_[x0.ravel(), x1.ravel()]
    y_prediction = polynomial_svm_clf.predict(X_concat).reshape(x0.shape)
    y_decision = polynomial_svm_clf.decision_function(X_concat).reshape(x0.shape)

    # contour alpha value sliders
    col2.markdown("Contour alpha sliders:")
    alpha_prediction = col2.slider("Prediction", 0.0, 1.0)
    alpha_decision = col2.slider("Decision Boundary", 0.0, 1.0)

    # plotting section
    fig = plt.figure(figsize=(10,10))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bo")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "rs")
    plt.axis([-1.5, 2.5, -1, 1.5])
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.contourf(x0, x1, y_prediction, cmap=color_map, alpha=alpha_prediction)
    plt.contourf(x0, x1, y_decision, cmap=color_map, alpha=alpha_decision)

    # streamlit pyplot show
    plothole.pyplot(fig,)






else:
   

    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(-1, 1, 8)
    u,v = np.meshgrid(u,v)
    u = u.flatten()
    v = v.flatten()

    tp = 1 + 0.5*v*np.cos(u/2.)
    x = tp*np.cos(u)
    y = tp*np.sin(u)
    z = 0.5*v*np.sin(u/2.)

    points2D = np.vstack([u,v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices

    st.markdown('''<center><h3>The Mobius Strip<sub>  by <a href="https://plotly.com/python/trisurf/">Plotly (DASH)</a></sub></h3></center>''', unsafe_allow_html=True)
    fig = ff.create_trisurf(x=x, y=y, z=z,
                            colormap="Portland",
                            simplices=simplices,
                            title="",
                            )
    st.plotly_chart(fig)