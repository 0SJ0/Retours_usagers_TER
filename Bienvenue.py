import streamlit as st
from PIL import Image

global_css="""
<style>
[data-testid="stAppViewContainer"]{
opacity: 1;
#font : oblique 14px Arial, Helvetica, sans-serif;
background-image: url("https://i.pinimg.com/originals/ee/5a/d3/ee5ad3e82144c6f39073344ed5f9f1c7.gif");
background-repeat: no-repeat;
background-size: 1880px 880px;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
background-color: rgba(0,0,0,0);
}

</style>
"""
#[data-testid="stSidebar"]{
#color: black;
#opacity: 0.94;
#background-image: url("https://bhf-cdn.azureedge.net/bhf-blob-prod/0037888_wisp-gold-texture-wallpaper_600.jpeg");
#}


st.markdown(global_css,unsafe_allow_html=True)

#st.sidebar.markdown("<footer><p style='text-align:center;'> <img src='https://upload.wikimedia.org/wikipedia/fr/thumb/2/2e/R%C3%A9gion_Hauts-de-France_logo_2016.svg/1200px-R%C3%A9gion_Hauts-de-France_logo_2016.svg.png' width='100' height='100'> </p></footer>", unsafe_allow_html=True)

