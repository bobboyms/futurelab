import os

import streamlit as st
from functions.data.load import load_histogram2d_data, load_scalar_data
from functions.directory import DirectoryTree
from functions.charts.chart import show_histogram2d, show_metrics
from lab.chart import Type, get_chart_type
from lab.write import scalar

root = "logs"
directory_tree = DirectoryTree(root)

#
st.set_page_config(layout="wide")

with st.container():
    with st.container():

        # Criando duas colunas
        st.sidebar.title("Menu")
        projects_options = directory_tree.get_projects()
        project_choice = st.sidebar.selectbox("Selecione um projeto", projects_options)

        labs_options = directory_tree.get_labs(project_choice)
        lab_choice = st.sidebar.selectbox("Selecione um Lab", labs_options)

        sections_options, _ = directory_tree.get_sections(project_choice, lab_choice)
        section_choice = st.selectbox("Selecione uma Sessão", sections_options)



charts, parent_folder = directory_tree.get_chart(project_choice, lab_choice, section_choice)

for i in range(0, len(charts), 2):
    cols = st.columns(2)

    for chart, col in zip(charts[i:i + 2], cols):

        chart_type, ok = get_chart_type(parent_folder, chart)

        if not ok:
            continue

        with col:

            if chart_type == Type.Histogram2d:
                histogram_df = load_histogram2d_data(parent_folder, chart)
                if not histogram_df.is_empty():
                    show_histogram2d(histogram_df, chart)

            if chart_type == Type.LineChart:
                scalar_df = load_scalar_data(parent_folder, chart)
                if not scalar_df.is_empty():
                    show_metrics(scalar_df, chart)




