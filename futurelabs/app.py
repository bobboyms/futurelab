import streamlit as st


from functions.data.load import load_histogram2d_data, load_scalar_data, load_audio_data
from functions.directory import DirectoryTree
from functions.charts.chart import show_histogram2d, show_metrics, show_audio
from lab.chart import Type, get_chart_type
from streamlit_js_eval import streamlit_js_eval

# import sys
# import os
#
# # Adiciona o diretório raiz ao sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


root = "logs"


directory_tree = DirectoryTree(root)



st.set_page_config(layout="wide")

page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)

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
num_columns = 1 if page_width < 1100 else 2


for i in range(0, len(charts), num_columns):

    cols = st.columns(num_columns)

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

            if chart_type == Type.AudioData:

                audio_list = load_audio_data(parent_folder, chart)
                show_audio(audio_list, chart)




