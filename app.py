import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as plt
import json
st.set_page_config(
    page_title="Data Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
 )

st.title('Data Visualization')
st.markdown("# Uncertainty Score Analysis")

uploaded_file = st.file_uploader(label="Choose a file", type=['json'])

sidebar = st.sidebar

if uploaded_file is not None:
    col1, col2 = st.columns([0.5, 0.5])
    Dict = json.load(uploaded_file)
    Data = pd.DataFrame.from_dict({(i,j): Dict[i][j] 
                           for i in Dict.keys() 
                           for j in Dict[i].keys()})
    
    with col1:
        boxPlot = Data.plot.box(fontsize=9,figsize = (20,15))
        st.pyplot(boxPlot.figure)



