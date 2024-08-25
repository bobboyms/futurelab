import plotly
import streamlit as st
import numpy as np
import polars as pl
import plotly.graph_objs as go

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
            colorscale='Blues',
            nbinsx=subsampled_steps,
            nbinsy=100
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

        # Passar o objeto `fig` corretamente para `st.plotly_chart`
        st.plotly_chart(fig)


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