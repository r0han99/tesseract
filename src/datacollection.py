import streamlit as st
import hydralit_components as hc
import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def datacollection_cs(slot):

    slot.markdown(f'''<h1 style="text-align:center;font-size:80px;border-style:solid; border-width:5px;">Data Collection<br><p style="text-align:center;"><strong>Rohan Sai Nalla</strong></p></h1>''',unsafe_allow_html=True)

    st.sidebar.markdown('Option')