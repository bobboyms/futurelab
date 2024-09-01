from cProfile import label

import librosa
import plotly
import streamlit as st
import numpy as np
import polars as pl
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import librosa.display
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, f1_score, \
    roc_auc_score
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

def show_percentile(df, label):


    lines_to_show = st.multiselect(
        "Escolha as linhas a serem exibidas no gráfico:",
        options=["Mean", "Standard Deviation", "25th Percentile", "50th Percentile (Median)", "75th Percentile"],
        default=["25th Percentile", "50th Percentile (Median)", "75th Percentile"],
        key=f"percentile_{label}"
    )

    def calculate_statistics(grad_tensor):

        if isinstance(grad_tensor, torch.Tensor):
            grad = grad_tensor.numpy().flatten()
        elif isinstance(grad_tensor, np.ndarray):
            grad = grad_tensor.flatten()
        else:
            grad = np.array(grad_tensor).flatten()

        return {
            "mean": grad.mean(),
            "std": grad.std(),
            "p25": np.percentile(grad, 25),
            "p50": np.percentile(grad, 50),
            "p75": np.percentile(grad, 75)
        }

    # Aplicar a função e expandir os resultados diretamente no DataFrame
    stats_df = df.with_columns(
        pl.col("value").map_elements(lambda x: calculate_statistics(x), return_dtype=pl.Struct).alias("stats")
    ).unnest("stats")



    # Plotar as curvas
    fig = go.Figure()
    times = len(stats_df)

    # Adiciona as curvas de estatísticas ao gráfico conforme seleção do usuário
    if "Mean" in lines_to_show:
        fig.add_trace(go.Scatter(
            x=np.arange(1, times + 1),
            y=stats_df["mean"],
            mode='lines',
            name='Mean Gradients',
            line=dict(color='blue')
        ))

    if "Standard Deviation" in lines_to_show:
        fig.add_trace(go.Scatter(
            x=np.arange(1, times + 1),
            y=stats_df["std"],
            mode='lines',
            name='Std Dev Gradients',
            line=dict(color='red')
        ))

    if "25th Percentile" in lines_to_show:
        fig.add_trace(go.Scatter(
            x=np.arange(1, times + 1),
            y=stats_df["p25"],
            mode='lines',
            name='25th Percentile',
            line=dict(dash='dot', color='green')
        ))

    if "50th Percentile (Median)" in lines_to_show:
        fig.add_trace(go.Scatter(
            x=np.arange(1, times + 1),
            y=stats_df["p50"],
            mode='lines',
            name='50th Percentile (Median)',
            line=dict(dash='dot', color='orange')
        ))

    if "75th Percentile" in lines_to_show:
        fig.add_trace(go.Scatter(
            x=np.arange(1, times + 1),
            y=stats_df["p75"],
            mode='lines',
            name='75th Percentile',
            line=dict(dash='dot', color='purple')
        ))

    # Configurações do layout
    fig.update_layout(
        title=f'Evolution of Gradients During Training - {label}',
        xaxis_title='Epoch',
        yaxis_title='Gradient Values',
        legend_title='Statistics',
        template='plotly_white',
        yaxis=dict(
            exponentformat='e',  # Notação científica
            showexponent='all'  # Mostrar o expoente para todos os ticks
        )
    )


    # Mostrar a figura
    st.plotly_chart(fig)

def show_norma(df, label):

    def converter(row):
        if isinstance(row, torch.Tensor):
            grad = row.numpy().flatten()
        elif isinstance(row, np.ndarray):
            grad = row.flatten()
        else:
            grad = np.array(row).flatten()

        return grad

    l2_norms = []
    for row in df.iter_rows():
        grad_tensor = converter(row[0])
        l2_norm = np.linalg.norm(grad_tensor)
        l2_norms.append(l2_norm)

    # Converter os resultados para arrays numpy
    # steps = np.array([row[0] for row in df.iter_rows()])
    l2_norms = np.array(l2_norms)

    # Criar o gráfico da magnitude total dos gradientes ao longo do tempo
    times = len(df)
    fig = go.Figure(data=go.Scatter(
        x=np.arange(1, times + 1),
        y=l2_norms,
        mode='lines+markers',
        name='L2 Norm'
    ))

    fig.update_layout(
        title=f'Magnitude Total (Norma L2) - {label}',
        xaxis_title='Step',
        yaxis_title='L2 Norm',
        template='plotly_dark',
        yaxis=dict(
            exponentformat='e',  # Notação científica
            showexponent='all'  # Mostrar o expoente para todos os ticks
        )
    )

    # fig.show()
    st.plotly_chart(fig)


def show_histogram(df, label):
    # options = list(range(len(df)))

    selected_index = st.number_input('Select Step:', min_value=0, max_value=len(df),
                                    value=len(df) -1, key=f"{label}_key_histogram")

    def converter(row):
        if isinstance(row, torch.Tensor):
            grad = row.numpy().flatten()
        elif isinstance(row, np.ndarray):
            grad = row.flatten()
        else:
            grad = np.array(row).flatten()

        return grad

    # Acessar o gradiente do registro selecionado
    selected_grad = converter(df[selected_index, "value"])


    # Configurar e exibir o gráfico
    min_val = selected_grad.min()
    max_val = selected_grad.max()
    fig = go.Figure(data=go.Histogram2d(
        x=np.arange(0, len(selected_grad)),
        y=selected_grad,
        nbinsx=100,
        nbinsy=30,
        # ybins=dict(start=min_val, end=max_val),
        colorscale=[
            [0, 'rgb(12,51,131)'],
            [0.25, 'rgb(10,136,186)'],
            [0.5, 'rgb(242,211,56)'],
            [0.75, 'rgb(242,143,56)'],
            [1, 'rgb(217,30,30)']
        ]
    ))

    fig.update_layout(
        # xaxis=dict(
        #     exponentformat='e',  # Notação científica
        #     showexponent='all'  # Mostrar o expoente para todos os ticks
        # ),
        yaxis=dict(
            exponentformat='e',  # Notação científica
            showexponent='all'  # Mostrar o expoente para todos os ticks
        )
    )

    st.plotly_chart(fig)


def show_histogram2d(combined_df, label):
    st.write(f"### {label}")
    if combined_df is not None and "value" in combined_df.columns:
        chart_format = st.radio(
            "Select a chart", ("Percentile", "Norma L2", "Histogram"), horizontal=True, key=f"{label}_histogram"
        )

        if chart_format == "Percentile":
            show_percentile(combined_df, label)
        if chart_format == "Norma L2":
            show_norma(combined_df, label)
        if chart_format == "Histogram":
            show_histogram(combined_df, label)




def show_metrics(combined_df, label ,show_title=True, start=1):
    if show_title:
        st.write(f"### {label}")

    available_columns = combined_df.columns[start:]
    selected_columns = st.multiselect(
        "Select metrics to display", available_columns, default=available_columns, key=f"show_metrics_{label}"
    )
    df = combined_df.sort('step')
    fig = px.line(df, x='step', y=selected_columns, markers=False)
    fig.update_layout(
        yaxis=dict(
            exponentformat='e',  # Notação científica
            showexponent='all'  # Mostrar o expoente para todos os ticks
        )
    )
    st.plotly_chart(fig)
    download_options(fig, label)



def show_audio(audio_files, label):
    st.write(f"### {label}")
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





def show_confusion_matrix(df, label):

    min = int(df['step'].min())
    max = int(df['step'].max())
    specific_step = st.number_input('Select Step:', min_value=min, max_value=max,
                                    value=max-1, key=f"{label}_confusion_matrix")

    step_data = df.filter(pl.col("step") == specific_step)

    if not step_data.is_empty():
        real_labels = step_data["real_labels_int"].to_list()
        predicted_labels = step_data["predicted_label_int"].to_list()

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


# @st.cache_data(ttl=120)
def apply_classification(_combined_df, threshold):

    # Explode as listas de rótulos
    df = _combined_df.explode(["real_label", "predicted_label"])

    df = df.with_columns([
        pl.col("predicted_label").alias("predicted_probs")
    ])


    df = df.with_columns([
        pl.col("predicted_label").map_elements(
            lambda x: process_binary_classification(x, threshold),
            return_dtype=pl.Int8
        ).alias("predicted_label_binary")
    ])



    # Agrupar por step
    grouped_df = df.group_by("step").agg([
        pl.col("real_label").alias("real_labels"),
        # pl.col("predicted_label").alias("predicted_label"),
        pl.col("predicted_label_binary").alias("predicted_label_int")

    ])

    print(grouped_df)

    # Converter a coluna 'real_labels' para uma lista de Int16
    grouped_df = grouped_df.with_columns([
        pl.col("real_labels").cast(pl.List(pl.Int8)).alias("real_labels_int")
    ])

    # Função para calcular a precisão
    def calculate_precision(row):
        return precision_score(row["real_labels_int"], row["predicted_label_int"], average='macro', zero_division=0) * 100

    def calculate_accuracy(row):
        return accuracy_score(row["real_labels_int"], row["predicted_label_int"]) * 100

    def calculate_sensitivity(row):
        return sensitivity_score(row["real_labels_int"], row["predicted_label_int"], average='macro') * 100

    def calculate_specificity(row):
        return specificity_score(row["real_labels_int"], row["predicted_label_int"], average='macro') * 100

    def calculate_f1(row):
        return f1_score(row["real_labels_int"],  row["predicted_label_int"], average='macro', zero_division=0) * 100


    # Adicionar a coluna de precisão
    grouped_df = grouped_df.with_columns([
        pl.struct(["real_labels_int", "predicted_label_int"]).map_elements(
            lambda x: calculate_precision(x),
            return_dtype=pl.Float64
        ).alias("precision"),
        pl.struct(["real_labels_int", "predicted_label_int"]).map_elements(
            lambda x: calculate_accuracy(x),
            return_dtype=pl.Float64
        ).alias("accuracy"),

        pl.struct(["real_labels_int", "predicted_label_int"]).map_elements(
            lambda x: calculate_sensitivity(x),
            return_dtype=pl.Float64
        ).alias("sensitivity"),
        pl.struct(["real_labels_int", "predicted_label_int"]).map_elements(
            lambda x: calculate_specificity(x),
            return_dtype=pl.Float64
        ).alias("specificity"),
        pl.struct(["real_labels_int", "predicted_label_int"]).map_elements(
            lambda x: calculate_f1(x),
            return_dtype=pl.Float64
        ).alias("f1_score")
    ]).sort("step")

    # print(grouped_df.head())
    return grouped_df

# @st.cache_data(ttl=120)
def calculate_metrics_for_thresholds(_combined_df, thresholds, specific_step):

    step_data = _combined_df.filter(pl.col("step") == specific_step)

    results = []
    for threshold in thresholds:
        # Aplicar classificação com o threshold atual
        metrics_df = apply_classification(step_data, threshold)

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
    st.write("### Classification Metrics by Threshold")

    min = int(_combined_df['step'].min())
    max = int(_combined_df['step'].max())
    specific_step = st.number_input('Select Step:', min_value=min, max_value=max,
                                    value=max, key=f"{label}_auc")
    # Calcular as métricas para cada threshold
    result_df = calculate_metrics_for_thresholds(_combined_df, thresholds, specific_step)

    # Exibir a tabela no Streamlit
    st.dataframe(result_df.to_pandas())


def show_auc(df, label):

    min = int(df['step'].min())
    max = int(df['step'].max())
    specific_step = st.number_input('Select Step:', min_value=min, max_value=max,
                                    value=max, key=f"{label}_auc")

    step_data = df.filter(pl.col("step") == specific_step)

    if not step_data.is_empty():
        real_labels = step_data["real_label"].to_list()
        predicted_labels = step_data["predicted_label"].to_list()

        # Calculando a AUC-ROC
        auc = roc_auc_score(real_labels[0], predicted_labels[0])

        # Gerando a curva ROC
        fpr, tpr, thresholds = roc_curve(real_labels[0], predicted_labels[0])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {auc:.2f})',
                                 line=dict(color='darkorange', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random chance',
                                 line=dict(color='navy', width=2, dash='dash')))

        # Configurando o layout
        fig.update_layout(
            title='Receiver Operating Characteristic',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0, 1.05])
        )

        # Exibindo o gráfico
        st.plotly_chart(fig)


    else:
        st.write(f"No data available for Step {specific_step}")

def show_classification(combined_df, label):
    st.write(f"### {label}")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold = st.selectbox('Select a threshold:', options=thresholds, index=0)

    df = apply_classification(combined_df, threshold)

    droped_df = df.drop(["real_labels", "predicted_label_int", "real_labels_int"])

    with st.container():
        chart_format = st.radio(
            "Select a chart", ("Scores", "AUC-ROC", "Confusion Matrix", "Table"), horizontal=True, key=f"{label}_format"
        )

        if chart_format == "Scores":
            show_metrics(droped_df, f"{label}_xs", show_title=False)

        if chart_format == "Confusion Matrix":
            show_confusion_matrix(df, label)

        if chart_format == "AUC-ROC":
            show_auc(combined_df, label)

        if chart_format == "Table":
            display_metrics_table(combined_df, thresholds)


