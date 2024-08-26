import streamlit as st


from futurelabs.functions.data.load import load_histogram2d_data, load_scalar_data, load_audio_data, load_classification
from futurelabs.functions.directory import DirectoryTree
from futurelabs.functions.charts.chart import show_histogram2d, show_metrics, show_audio, show_classification
from futurelabs.lab.chart import Type, get_chart_type
from streamlit_js_eval import streamlit_js_eval

def dashboard():
    root = "logs"
    directory_tree = DirectoryTree(root)

    if not directory_tree.directory_exists(root):
        st.write("""
        # 📁 Root folder not found
        
        You must first create a folder where the logs will be saved
        """)
        return

    # st.set_page_config(layout="wide")

    page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH', want_output=True, )

    with st.container():
        st.sidebar.title("Menu")
        projects_options, parent_folder = directory_tree.get_projects()
        project_choice = st.sidebar.selectbox("Selecione um projeto", projects_options)

        labs_options, parent_folder = directory_tree.get_labs(parent_folder, project_choice)
        lab_choice = st.sidebar.selectbox("Selecione um Lab", labs_options)

        sections_options, parent_folder = directory_tree.get_sections(parent_folder, lab_choice)
        section_choice = st.selectbox("Selecione uma Sessão", sections_options)

    charts, parent_folder = directory_tree.get_chart(parent_folder, section_choice)
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

                if chart_type == Type.Classification:
                    classification_df = load_classification(parent_folder, chart)
                    print(classification_df)

                    show_classification(classification_df, chart)


st.set_page_config(
    page_title="Future Labs Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

dashboard()

