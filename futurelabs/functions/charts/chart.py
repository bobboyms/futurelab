import librosa
import plotly
import streamlit as st
import numpy as np
import polars as pl
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import librosa.display
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, f1_score
from imblearn.metrics import specificity_score
from imblearn.metrics import sensitivity_score
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função para processar classificação binária
def process_binary_classification(data, threshold):
    sigmoid_values = sigmoid(data)
    return np.where(sigmoid_values > threshold, 1, 0)


def download_options(fig, label):
    # Adicionar um botão discreto para abrir as opções de download

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.expander("Download options"):
            download_format = st.radio(
                "Choose format", ("PNG", "SVG", "PDF"), key=f"{label}_format"
            )

            if download_format == "PNG":
                st.download_button(
                    label="Download PNG",
                    data=fig.to_image(format="png"),
                    file_name=f"{label}.png",
                    mime="image/png",
                    key=f"{label}_download_png"
                )
            elif download_format == "SVG":
                st.download_button(
                    label="Download SVG",
                    data=fig.to_image(format="svg"),
                    file_name=f"{label}.svg",
                    mime="image/svg+xml",
                    key=f"{label}_download_svg"
                )
            elif download_format == "PDF":
                st.download_button(
                    label="Download PDF",
                    data=fig.to_image(format="pdf"),
                    file_name=f"{label}.pdf",
                    mime="application/pdf",
                    key=f"{label}_download_pdf"
                )


def show_histogram2d(combined_df, label):
    if combined_df is not None and "value" in combined_df.columns:

        steps = len(combined_df["value"])
        values_per_step = len(combined_df["value"][0])

        # Definir taxa de amostragem dependendo do número de dados
        if values_per_step <= 100:
            sampling_rate = 1  # Subamostrar a cada step
        else:
            sampling_rate = max(steps * 0.01, 1)

        subsampled_steps = int(steps // sampling_rate)

        data_df = pl.DataFrame({
            "step": np.arange(subsampled_steps),
            "value": [combined_df["value"][int(i * sampling_rate)] for i in range(subsampled_steps)]
        })

        expanded_data = data_df.explode("value")

        # Gráfico 2D com Plotly
        fig = go.Figure(data=go.Histogram2d(
            x=expanded_data["step"],
            y=expanded_data["value"],
            colorscale=[[0, 'rgb(12,51,131)'], [0.25, 'rgb(10,136,186)'], [0.5, 'rgb(242,211,56)'], [0.75, 'rgb(242,143,56)'], [1, 'rgb(217,30,30)']],
            # nbinsx=subsampled_steps,
            # nbinsy=100,
        ))

        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="Value",
            title={
                'text': f"{label}",
                'x': 0.5,  # Centraliza o título
                'xanchor': 'center',  # Define a âncora de referência como o centro
                'yanchor': 'top'
            }
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)

        # Adicionar um botão discreto para abrir as opções de download
        download_options(fig, label)




def show_metrics(combined_df, label, start=1):

    available_columns = combined_df.columns[start:]
    selected_columns = st.multiselect(
        "Select metrics to display", available_columns, default=available_columns
    )
    df = combined_df.sort('step')
    fig = px.line(df, x='step', y=selected_columns, markers=True)
    st.plotly_chart(fig)
    download_options(fig, label)



def show_audio(audio_files, label):
    with st.container():
        selected_index = st.slider('Select audio file',
                                   min_value=0,
                                   max_value=len(audio_files) - 1,
                                   value=0)

        # Mostrar o nome do arquivo de áudio selecionado
        selected_file = audio_files[selected_index]

        # Carregar o áudio selecionado usando librosa
        y, sr = librosa.load(selected_file, sr=None)

        # Exibir o player de áudio para o arquivo selecionado
        st.audio(str(selected_file))

        fig = None
        st1, _, _ = st.columns(3)

        with st1:
            chart_format = st.radio(
                "Choose format", ("WAVE", "MEL"), key=f"{label}_format", horizontal=True
            )


            if chart_format == "WAVE":
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set(title=f"Waveform of {selected_file.name}")


            elif chart_format == "MEL":

                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=328)
                S_dB = librosa.power_to_db(S, ref=np.max)

                fig, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title=f"Mel Spectrogram of {selected_file.name}")



        if fig:
            st.pyplot(fig)





def show_confusion_matrix(df):
    specific_step = st.number_input('Select Step:', min_value=int(df['step'].min()), max_value=int(df['step'].max()),
                                    value=0)

    step_data = df.filter(pl.col("step") == specific_step)

    if not step_data.is_empty():
        real_labels = step_data["real_labels_int"].to_list()
        predicted_labels = step_data["predicted_labels"].to_list()

        conf_matrix = confusion_matrix(real_labels[0], predicted_labels[0])

        labels = np.unique(real_labels[0])

        z = conf_matrix
        x = [f"Predicted {label}" for label in labels]
        y = [f"Actual {label}" for label in labels]

        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')

        fig.update_layout(
            title=f"Confusion Matrix for Step {specific_step}",
            title_x=0.5,
        )

        st.plotly_chart(fig)
        download_options(fig, specific_step)
    else:
        st.write(f"No data available for Step {specific_step}")


@st.cache_data(ttl=120)
def apply_classification(_combined_df, threshold):

    # Explode as listas de rótulos
    df = _combined_df.explode(["real_label", "predicted_label"])

    # Mapear os rótulos previstos binários
    df = df.with_columns([
        pl.col("predicted_label").map_elements(
            lambda x: process_binary_classification(x, threshold),
            return_dtype=pl.Int8
        ).alias("predicted_label_binary")
    ])

    # Agrupar por step
    grouped_df = df.group_by("step").agg([
        pl.col("real_label").alias("real_labels"),
        pl.col("predicted_label_binary").alias("predicted_labels")
    ])

    # Converter a coluna 'real_labels' para uma lista de Int16
    grouped_df = grouped_df.with_columns([
        pl.col("real_labels").cast(pl.List(pl.Int8)).alias("real_labels_int")
    ])

    # Função para calcular a precisão
    def calculate_precision(row):
        return precision_score(row["real_labels_int"], row["predicted_labels"], average='macro', zero_division=0) * 100

    def calculate_accuracy(row):
        return accuracy_score(row["real_labels_int"], row["predicted_labels"]) * 100

    def calculate_sensitivity(row):
        return sensitivity_score(row["real_labels_int"], row["predicted_labels"], average='macro') * 100

    def calculate_specificity(row):
        return specificity_score(row["real_labels_int"], row["predicted_labels"], average='macro') * 100

    def calculate_f1(row):
        return f1_score(row["real_labels_int"],  row["predicted_labels"], average='macro', zero_division=0) * 100


    # Adicionar a coluna de precisão
    grouped_df = grouped_df.with_columns([
        pl.struct(["real_labels_int", "predicted_labels"]).map_elements(
            lambda x: calculate_precision(x),
            return_dtype=pl.Float64
        ).alias("precision"),
        pl.struct(["real_labels_int", "predicted_labels"]).map_elements(
            lambda x: calculate_accuracy(x),
            return_dtype=pl.Float64
        ).alias("accuracy"),

        pl.struct(["real_labels_int", "predicted_labels"]).map_elements(
            lambda x: calculate_sensitivity(x),
            return_dtype=pl.Float64
        ).alias("sensitivity"),
        pl.struct(["real_labels_int", "predicted_labels"]).map_elements(
            lambda x: calculate_specificity(x),
            return_dtype=pl.Float64
        ).alias("specificity"),
        pl.struct(["real_labels_int", "predicted_labels"]).map_elements(
            lambda x: calculate_f1(x),
            return_dtype=pl.Float64
        ).alias("f1_score")
    ]).sort("step")


    return grouped_df

@st.cache_data(ttl=120)
def calculate_metrics_for_thresholds(_combined_df, thresholds):
    results = []

    for threshold in thresholds:
        # Aplicar classificação com o threshold atual
        metrics_df = apply_classification(_combined_df, threshold)

        # Calcular médias das métricas para o threshold atual
        precision_avg = metrics_df['precision'].mean()
        accuracy_avg = metrics_df['accuracy'].mean()
        sensitivity_avg = metrics_df['sensitivity'].mean()
        specificity_avg = metrics_df['specificity'].mean()
        f1_avg = metrics_df['f1_score'].mean()

        # Adicionar resultados na lista
        results.append([
            threshold,
            precision_avg,
            accuracy_avg,
            sensitivity_avg,
            specificity_avg,
            f1_avg
        ])

    # Criar um DataFrame com os resultados
    result_df = pl.DataFrame(results, schema=["Threshold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score"])

    return result_df

def display_metrics_table(_combined_df, thresholds):
    # Calcular as métricas para cada threshold
    result_df = calculate_metrics_for_thresholds(_combined_df, thresholds)

    # Exibir a tabela no Streamlit
    st.write("### Classification Metrics by Threshold")
    st.dataframe(result_df.to_pandas())


def show_classification(combined_df, label):
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold = st.selectbox('Select a threshold:', options=thresholds, index=0)

    df = apply_classification(combined_df, threshold)
    droped_df = df.drop(["real_labels", "predicted_labels", "real_labels_int"])

    with st.container():
        chart_format = st.radio(
            "Select a chart", ("Scores", "AUC-ROC", "Confusion Matrix", "Table"), horizontal=True, key=f"{label}_format"
        )

        if chart_format == "Scores":
            show_metrics(droped_df, f"{label}_xs")

        if chart_format == "Confusion Matrix":
            show_confusion_matrix(df)

        if chart_format == "Table":
            display_metrics_table(combined_df, thresholds)


