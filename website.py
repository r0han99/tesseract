import streamlit as st
import datetime
import time
import base64
from pathlib import Path
import plotly.express as px
import streamlit_ext as ste

import plotly.figure_factory as ff


import numpy as np
from scipy.spatial import Delaunay



import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


category = ste.selectbox('Select Sandbox Category', ['Select a Category','Support Vector Machines','Neural Networks',],key='category')
st.markdown('***')

if category == 'Neural Networks':
    st.markdown('''<center><h3>Neural Network Simulation (Under Development)</h3></center>''', unsafe_allow_html=True)
    
    u=np.linspace(-np.pi/2, np.pi/2, 60)
    v=np.linspace(0, np.pi, 60)
    u,v=np.meshgrid(u,v)
    u=u.flatten()
    v=v.flatten()

    x = (np.sqrt(2)*(np.cos(v)*np.cos(v))*np.cos(2*u) + np.cos(u)*np.sin(2*v))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))
    y = (np.sqrt(2)*(np.cos(v)*np.cos(v))*np.sin(2*u) - np.sin(u)*np.sin(2*v))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))
    z = (3*(np.cos(v)*np.cos(v)))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))

    points2D = np.vstack([u, v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices

    fig = ff.create_trisurf(x=x, y=y, z=z,
                            colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                            show_colorbar=True,
                            simplices=simplices,
                            title="Sample Plot ~ Boy's Surface")
    
    st.plotly_chart(fig)


elif category == 'Support Vector Machines':
    #st.markdown('''<center><h3>Support Vector Machines<sub>  by <a href="https://medium.com/@sinchan.s/support-vector-machine-svm-in-action-using-streamlit-e3bc56208a85">sinchan</a></sub></h3></center>''', unsafe_allow_html=True)
    st.markdown('''<center><h3>Support Vector Machines</h3></center>''', unsafe_allow_html=True)
    st.markdown("***")

    

    
    # matplotlib colormap selection dropdown
    #color_maps_list = ( 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu')
   
    col1, col2 = st.columns(2)
    kernel = col1.selectbox('Choose Kernel',['Polynomial', 'Radial Bias', 'Linear',],key='kernels')
    plothole = col1.empty()

    if kernel == 'Polynomial':
        
        color_map = "Blues"

        # sample size & noise control sliders
        col2.markdown("Data points distribution:")
        n_samples = col2.slider("Samples", 2, 100, value=40)
        noise = col2.slider("Noise", 0.01, 1.00)

        # x & y variables assignment
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

        # prediction pipeline
        polynomial_svm_clf = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))])
        polynomial_svm_clf.fit(X, y)

        col1.markdown(f'<center><i>{kernel} Kernel</i></center>',unsafe_allow_html=True)

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
        alpha_decision = col2.slider("Decision Boundary", 0.0, 1.0,value=0.26)

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
        color_map = "Blues"

        col2.markdown("Data points distribution:")
        n_samples = col2.slider("Samples", 2, 100, value=40)
        noise = col2.slider("Noise", 0.01, 1.00)

        X, y = make_circles(n_samples=n_samples,noise=noise, random_state=42)

        if kernel == "Linear":
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

        col1.markdown(f'<center><i>{kernel} Kernel</i></center>',unsafe_allow_html=True)

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



    st.markdown("<center><i>The Shaded Region depicts the Decision Boulder, play with the Sliders to see changes.</center></i>",unsafe_allow_html=True)


else:
   
    st.markdown('''<center><h4>Select Sandbox Category ↑</h4></center>''', unsafe_allow_html=True)

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

    #st.markdown('''<center><h5>A Placeholder plot ~ The Mobius Strip<sub>  by <a href="https://plotly.com/python/trisurf/">Plotly (DASH)</a></sub></h5></center>''', unsafe_allow_html=True)
    fig = ff.create_trisurf(x=x, y=y, z=z,
                            colormap="Portland",
                            simplices=simplices,
                            title="",
                            )
    #st.plotly_chart(fig)