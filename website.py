import streamlit as st
import datetime
import time
import base64
from pathlib import Path
import plotly.express as px

import plotly.figure_factory as ff


import numpy as np
from scipy.spatial import Delaunay



category = st.selectbox('Select Sandbox Category', ['Support Vector Machines','Neural Networks',],key='category')


if category == 'Neural Networks':
    st.markdown('Testing Sample Plotly Animations')
    
    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
    animation_frame="year", animation_group="country", range_y=[0,4000000000])
    
    st.plotly_chart(fig)

elif category == 'Support Vector Machines':
    st.markdown('''<c><h3>Support Vector Machines</h3></c>''', unsafe_allow_html=True)


    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
    animation_frame="year", animation_group="country", range_y=[0,4000000000])

    st.plotly_chart(fig)