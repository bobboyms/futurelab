import librosa
import plotly
import streamlit as st
import numpy as np
import polars as pl
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import librosa.display


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


# Criar um slider no Streamlit para selecionar a quantidade de arquivos a exibir


# def show_audio(audio_files, label):
#     selected_index = st.slider('Select audio file',
#                           min_value=0,
#                           max_value=len(audio_files) -1,
#                           value=0)
#
#     st.write(audio_files[selected_index])

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
