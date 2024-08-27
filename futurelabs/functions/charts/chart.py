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


def show_metrics(combined_df, label):
    # Criar o gráfico de linha com Plotly
    fig = go.Figure()

    # Obter a lista de colunas que não seja a coluna "step"
    metric_columns = combined_df.columns[1:]  # Ignora a primeira coluna que é "step"

    # Gerar cores automaticamente para cada linha (usando a paleta padrão do Plotly)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    # Adicionar uma linha para cada métrica dinamicamente
    for i, column in enumerate(metric_columns):
        fig.add_trace(go.Scatter(x=combined_df["step"].to_list(), y=combined_df[column].to_list(),
                                 mode='lines+markers', name=column,
                                 line=dict(color=colors[i % len(colors)])))

    # Configurando o layout do gráfico
    fig.update_layout(
        title=label,
        xaxis_title='Step',
        yaxis_title='Metric Value',
        title_x=0.5  # Centralizar o título
    )

    # Exibir o gráfico no Streamlit
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
            with st.expander("Opções de visualização"):
                chart_format = st.radio(
                    "Choose format", ("WAVE", "MEL"), key=f"{label}_format"
                )

                if chart_format == "WAVE":

                    if st.button("Visualize Waveform"):
                        # Plotar a waveform (forma de onda) do áudio
                        fig, ax = plt.subplots(figsize=(10, 4))
                        librosa.display.waveshow(y, sr=sr, ax=ax)
                        ax.set(title=f"Waveform of {selected_file.name}")

                elif chart_format == "MEL":

                    if st.button("Visualize Mel"):
                        # Plotar o Mel Spectrograma do áudio
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=328)
                        S_dB = librosa.power_to_db(S, ref=np.max)

                        fig, ax = plt.subplots(figsize=(10, 4))
                        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        ax.set(title=f"Mel Spectrogram of {selected_file.name}")

        if fig:
            st.pyplot(fig)

def show_accuracy(df, threshold):
    # Lista para armazenar acurácia e steps
    accuracies = []
    steps = []

    # Iterar sobre os steps únicos e calcular a acurácia para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_list()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)
        accuracy = accuracy_score(real_labels, predicted_labels)

        # Armazenar os valores de acurácia e step
        accuracies.append(accuracy * 100)  # Convertendo para percentual
        steps.append(step)

    # Criar um gráfico de linha para a acurácia
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy per Step'
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="Accuracy per Step",
        xaxis_title="Step",
        yaxis_title="Accuracy (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        margin=dict(t=50, b=50),
        title_x=0.5
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)
    download_options(fig, "Accuracies")

def show_confusion_matrix(df, threshold):
    # Selecionar o step específico
    specific_step = st.number_input('Select Step:', min_value=int(df['step'].min()), max_value=int(df['step'].max()),
                                    value=3)

    # Filtrar os dados para o step específico
    step_data = df.filter(pl.col("step") == specific_step)

    # Extrair os rótulos reais e previstos
    real_labels = step_data["real_label"].to_list()
    predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold).tolist()

    if real_labels and predicted_labels:
        # Calcular a matriz de confusão
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Determinar os rótulos únicos para os eixos
        labels = np.unique(real_labels)

        # Gerar a matriz de confusão usando Plotly
        z = conf_matrix
        x = [f"Predicted {label}" for label in labels]
        y = [f"Actual {label}" for label in labels]

        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')

        # Ajustar a layout para melhor visualização
        fig.update_layout(
            title=f"Confusion Matrix for Step {specific_step}",
            # margin=dict(t=200, b=10), # Aumenta o espaçamento superior e inferior
            title_x = 0.5,
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)

        download_options(fig, specific_step)
    else:
        st.write(f"No data available for Step {specific_step}")


def show_precision(df,threshold):

    precisions = []
    steps = []

    # Iterar sobre os steps únicos e calcular a precisão para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_numpy()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)
        precision = precision_score(real_labels, predicted_labels, average='macro', zero_division=0)

        # Armazenar os valores de precisão e step
        precisions.append(precision * 100)  # Convertendo para percentual
        steps.append(step)

    # Ordenar steps e precisions com base nos steps
    sorted_steps, sorted_precisions = zip(*sorted(zip(steps, precisions)))

    # Criar um gráfico de linha para a precisão
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_steps,
        y=sorted_precisions,
        mode='lines+markers',
        name='Precision per Step',
        line = dict(color='green'),
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="Precision per Step",
        xaxis_title="Step",
        yaxis_title="Precision (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        # margin=dict(t=50, b=50)
        title_x=0.5,
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)
    download_options(fig, "Precision")


# Função principal
def show_recall(df,threshold):
    recalls = []
    steps = []

    # Iterar sobre os steps únicos e calcular a revocação para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_numpy()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)
        recall = recall_score(real_labels, predicted_labels, average='macro', zero_division=0)

        # Armazenar os valores de revocação e step
        recalls.append(recall * 100)  # Convertendo para percentual
        steps.append(step)

    # Ordenar steps e recalls com base nos steps
    sorted_steps, sorted_recalls = zip(*sorted(zip(steps, recalls)))

    # Criar um gráfico de linha para a revocação
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_steps,
        y=sorted_recalls,
        mode='lines+markers',
        name='Recall per Step',
        line=dict(color='yellow')  # Define a cor da linha
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="Sensitivity(Recall) per Step",
        xaxis_title="Step",
        yaxis_title="Recall (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        margin=dict(t=50, b=50)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)
    download_options(fig, "Sensitivity")


# Função principal
def show_auc_roc(df,threshold):

    aucs = []
    steps = []

    # Iterar sobre os steps únicos e calcular a AUC-ROC para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_numpy()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)

        # Calcular a curva ROC e a AUC
        fpr, tpr, _ = roc_curve(real_labels, predicted_labels, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Armazenar os valores de AUC e step
        aucs.append(roc_auc * 100)  # Convertendo para percentual
        steps.append(step)

    # Ordenar steps e AUCs com base nos steps
    sorted_steps, sorted_aucs = zip(*sorted(zip(steps, aucs)))

    # Criar um gráfico de linha para a AUC-ROC
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_steps,
        y=sorted_aucs,
        mode='lines+markers',
        name='AUC-ROC per Step',
        line=dict(color='purple')  # Define a cor da linha
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="AUC-ROC per Step",
        xaxis_title="Step",
        yaxis_title="AUC-ROC (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        margin=dict(t=50, b=50)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)


# Função para calcular a especificidade
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []

    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # Verdadeiros negativos
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])  # Falsos positivos
        specificity = tn / (tn + fp)
        specificities.append(specificity)

    return np.mean(specificities)


# Função principal
def show_specificity(df, threshold):

    # Lista para armazenar especificidade e steps
    specificities = []
    steps = []

    # Iterar sobre os steps únicos e calcular a especificidade para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_numpy()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)
        specificity = calculate_specificity(real_labels, predicted_labels)

        # Armazenar os valores de especificidade e step
        specificities.append(specificity * 100)  # Convertendo para percentual
        steps.append(step)

    # Ordenar steps e specificities com base nos steps
    sorted_steps, sorted_specificities = zip(*sorted(zip(steps, specificities)))

    # Criar um gráfico de linha para a especificidade
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_steps,
        y=sorted_specificities,
        mode='lines+markers',
        name='Specificity per Step',
        line=dict(color='orange')  # Define a cor da linha
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="Specificity per Step",
        xaxis_title="Step",
        yaxis_title="Specificity (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        margin=dict(t=50, b=50)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)


def show_f1_score(df, threshold):
    f1_scores = []
    steps = []

    # Iterar sobre os steps únicos e calcular o F1 Score para cada um
    for step in df['step'].unique():
        step_data = df.filter(pl.col("step") == step)
        real_labels = step_data["real_label"].to_numpy()
        predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(),threshold)
        f1 = f1_score(real_labels, predicted_labels, average='macro', zero_division=0)

        # Armazenar os valores de F1 Score e step
        f1_scores.append(f1 * 100)  # Convertendo para percentual
        steps.append(step)

    # Ordenar steps e f1_scores com base nos steps
    sorted_steps, sorted_f1_scores = zip(*sorted(zip(steps, f1_scores)))

    # Criar um gráfico de linha para o F1 Score
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_steps,
        y=sorted_f1_scores,
        mode='lines+markers',
        name='F1 Score per Step',
        line=dict(color='red')  # Define a cor da linha
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title="F1 Score per Step",
        xaxis_title="Step",
        yaxis_title="F1 Score (%)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Configurando os steps como inteiros
        yaxis=dict(range=[0, 100], tickformat=".0f%%"),  # Configurando o eixo Y como percentual
        margin=dict(t=50, b=50)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)

def generate_metrics_table(df, threshold_values):
    # DataFrame para armazenar os resultados
    results = []

    # Iterar sobre cada threshold e calcular as métricas
    for threshold in threshold_values:
        row = {"Threshold": threshold}

        accuracy_list = []
        precision_list = []
        recall_list = []
        specificity_list = []
        auc_list = []
        f1_score_list = []

        # Iterar sobre os steps únicos e calcular as métricas para cada um
        for step in df['step'].unique():
            step_data = df.filter(pl.col("step") == step)
            real_labels = step_data["real_label"].to_numpy()
            predicted_labels = process_binary_classification(step_data["predicted_label"].to_numpy(), threshold)

            # Calcular as métricas
            accuracy_list.append(accuracy_score(real_labels, predicted_labels))
            precision_list.append(precision_score(real_labels, predicted_labels, average='macro', zero_division=0))
            recall_list.append(recall_score(real_labels, predicted_labels, average='macro', zero_division=0))
            specificity_list.append(calculate_specificity(real_labels, predicted_labels))
            fpr, tpr, _ = roc_curve(real_labels, predicted_labels, pos_label=1)
            auc_list.append(auc(fpr, tpr))
            f1_score_list.append(f1_score(real_labels, predicted_labels, average='macro', zero_division=0))

        # Média das métricas para cada threshold
        row["Accuracy"] = np.mean(accuracy_list)
        row["Precision"] = np.mean(precision_list)
        row["Sensitivity"] = np.mean(recall_list)
        row["Specificity"] = np.mean(specificity_list)
        row["AUC-ROC"] = np.mean(auc_list)
        row["F1 Score"] = np.mean(f1_score_list)

        results.append(row)

    # Converter os resultados em um DataFrame
    metrics_df = pl.DataFrame(results)


    # Exibir a tabela no Streamlit
    st.dataframe(metrics_df)

    # Criação do botão de download
    st.download_button(
        label="⬇️",  # Sem texto no botão
        data=metrics_df.to_pandas().to_csv(index=False),
        file_name="metrics_table.csv",
        mime='text/csv',
        help="Download CSV",
        key="download_csv"
    )

def show_classification(combined_df, label):
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold = st.selectbox('Select a threshold:', options=thresholds, index=0)
    df = combined_df.explode(["real_label", "predicted_label"])

    st1, _, _ = st.columns(3)

    with st.container():
        chart_format = st.radio(
            "Select a chart", ("Accuracy", "Precision", "Sensitivity", "Specificity", "AUC-ROC","F1 Score", "Confusion Matrix", "Table"), horizontal=True, key=f"{label}_format"
        )

        if chart_format == "Accuracy":
            show_accuracy(df, threshold)

        if chart_format == "Precision":
            show_precision(df, threshold)

        if chart_format == "Confusion Matrix":
            show_confusion_matrix(df, threshold)

        if chart_format == "Sensitivity":
            show_recall(df, threshold)

        if chart_format == "Specificity":
            show_specificity(df, threshold)

        if chart_format == "AUC-ROC":
            show_auc_roc(df, threshold)

        if chart_format == "F1 Score":
            show_f1_score(df, threshold)

        if chart_format == "Table":
            generate_metrics_table(df, thresholds)



