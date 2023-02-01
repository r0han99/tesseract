import streamlit as st
import hydralit_components as hc
import base64
from pathlib import Path

svg_slot = '''<svg xmlns="http://www.w3.org/2000/svg" width="75" height="75" fill="currentColor" class="bi bi-unity" viewBox="0 0 16 16">
  <path d="M15 11.2V3.733L8.61 0v2.867l2.503 1.466c.099.067.099.2 0 .234L8.148 6.3c-.099.067-.197.033-.263 0L4.92 4.567c-.099-.034-.099-.2 0-.234l2.504-1.466V0L1 3.733V11.2v-.033.033l2.438-1.433V6.833c0-.1.131-.166.197-.133L6.6 8.433c.099.067.132.134.132.234v3.466c0 .1-.132.167-.198.134L4.031 10.8l-2.438 1.433L7.983 16l6.391-3.733-2.438-1.434L9.434 12.3c-.099.067-.198 0-.198-.133V8.7c0-.1.066-.2.132-.233l2.965-1.734c.099-.066.197 0 .197.134V9.8L15 11.2Z"/>
</svg>'''

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def ml_cs(slot):
    
    slot.markdown(f'''<h1 style="text-align:center;font-family:georgia;font-size:80px;border-style:solid; border-width:5px;" >{svg_slot} Machine Learning CSCI 5622</h1>''',unsafe_allow_html=True)

    st.image('./images/chicago.jpeg',caption='by Christopher Alvarenga',use_column_width=True)
    st.markdown('***')

    problem_statement = '''Can data-driven approach help in reducing crime rate in Chicago, as analyzed through crime records and demographic data?'''
    st.markdown(f'''<span style="font-family:'Chakra Petch';font-weight:500;font-size:50px;">{problem_statement}</span>''',unsafe_allow_html=True)

    with open('./content/introduction.txt','r') as f:
        content = f.read()

    

    st.markdown(f'''<span style="font-family:'Chakra Petch';">{content}</span>''',unsafe_allow_html=True)







    st.markdown(f'''<a href="#top"> scroll to top </a>''',unsafe_allow_html=True)


    