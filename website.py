import streamlit as st
import hydralit_components as hc
import datetime
import time
from src.datacollection import datacollection_cs
from src.machinelearning import ml_cs
import base64
from pathlib import Path

def load_fonts():
    font_url = '''
    <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500&family=Martian+Mono:wght@300&family=Playfair+Display&display=swap" rel="stylesheet">
    '''
    st.markdown(f'{font_url}',unsafe_allow_html=True)


def font_test():
    test = '''
    Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
    '''
    st.markdown(f'''<span style="font-family:'Chakra Petch';font-weight:500;">{test}</span>''',unsafe_allow_html=True)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


#make it look nice from the start
st.set_page_config(layout='wide',page_title='tesseract')


# specify the primary menu definition
menu_data = [
    {'icon': "bi bi-hurricane",'label':"Machine Learning", 'submenu':[{'id':'introduction','icon': "bi bi-circle", 'label':"Introduction"},{'id':'datacollect','icon': "bi bi-boxes", 'label':"Data Collection"},{'id':'exploration','icon': "bi bi-graph-down-arrow", 'label':"Data Exploration"},{'id':'model-sandbox','icon': "bi bi-hourglass-split", 'label':"Model Sandbox"},{'id':'conclusion','icon': "bi bi-card-checklist", 'label':"Results & Conclusion"}]},
    # {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
    # {'id':' Crazy return value 💀','icon': "💀", 'label':"Calendar"},
    {'icon': "bi bi-github",'label':"Portal to My Projects", 'submenu':[{'id':'f1paddock','icon': "bi bi-controller", 'label':"Formula1 Web-Paddock"},{'id':'covid19','icon': "bi bi-snow2", 'label':"Covid19 Web Application"},{'id':'plant-disease','icon': "bi bi-card-image", 'label':"Plant Disease Classification"},{'id':'activity-log','icon': "bi bi-card-checklist", 'label':"Activity Logger"},]},
    # {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
    {'icon': "bi bi-chat-square-quote", 'label':"Contact"},
    # {'icon': "fa-solid fa-radar",'label':"Dropdown2", 'submenu':[{'label':"Sub-item 1", 'icon': "fa fa-meh"},{'label':"Sub-item 2"},{'icon':'🙉','label':"Sub-item 3",}]},
]



over_theme = {'txc_inactive': 'white','menu_background':'#144272','txc_active':'#30a3ff','font-family':'Chakra Petch'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
    use_animation=False
)
st.markdown("<span id='top'></span>",unsafe_allow_html=True)


title_slot = st.empty()
subtitle_slot = st.empty()
title_slot.markdown(f'''<h1 style="text-align:center;font-family:georgia;font-size:80px;border-top:solid;border-left:solid; border-right: solid; border-width:2px;"><img src='data:image/png;base64,{img_to_bytes('./assets/tesseract.png')}' class='img-fluid' width=150 >Tesseract<sub style="text-align:center;font-family:'Chakra Petch';"><strong>  A Repository of Knowledge.</strong></sub></h1>''',unsafe_allow_html=True)
subtitle_slot.markdown(f'''<p style="text-align:center;font-family:'Chakra Petch';border-bottom:solid;border-left:solid; border-right: solid; border-width:2px;">By Rohan Sai Nalla</p>''',unsafe_allow_html=True)
st.markdown('***')
#get the id of the menu item clicked
load_fonts()



if menu_id == "datacollect":
    subtitle_slot.markdown('')
    datacollection_cs(title_slot)

elif menu_id == 'introduction':
    subtitle_slot.markdown('')
    ml_cs(title_slot)





elif menu_id == 'Home':
    
    font_test()