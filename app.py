import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as plt
import json
from Data_process import process

def main():
    st.set_page_config(
        page_title="Data Visualization",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    folder = ['test','alt_1km','alt_10km','alt_50km']

    st.title('Data Visualization')
    st.markdown("# Uncertainty Score Analysis")

    sidebar = st.sidebar
    with st.spinner('Wait for it...'):
        avg_dict,Dict =  process()


    col1, col2 = st.columns([0.5, 0.5])

    Data = pd.DataFrame.from_dict({(i,j): Dict[i][j] 
                            for i in Dict.keys() 
                            for j in Dict[i].keys()})

    data = pd.DataFrame(avg_dict, index=folder)

    with col1:
        boxPlot = Data.plot.box(fontsize=9,figsize = (20,15))
        st.pyplot(boxPlot.figure)
    with col2:
        chart = data.plot()
        st.pyplot(chart.figure)
    st.write(Data.describe())

if __name__ == '__main__':
    main()

