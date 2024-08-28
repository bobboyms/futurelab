import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import polars as pl
from concurrent.futures import ThreadPoolExecutor
import streamlit as st


# @st.cache_data
def load_histogram2d_data(parent_folder, section):
    directory = os.path.join(parent_folder, section)
    parquet_files = list(Path(directory).glob("*.parquet"))

    # Função para extrair o timestamp do nome do arquivo
    def extract_timestamp(file):
        file_stem = file.stem  # Obtém o nome do arquivo sem a extensão
        try:
            return datetime.strptime(file_stem, "%Y_%m_%d_%H:%M:%S:%f")
        except ValueError as e:
            return None

    # Ordena os arquivos com base no timestamp extraído
    sorted_files = sorted(
        [file for file in parquet_files if extract_timestamp(file) is not None],
        key=extract_timestamp
    )

    # Função para carregar um arquivo Parquet
    def load_parquet(file):
        try:
            return pl.read_parquet(file)
        except Exception as e:
            return None

    # Carrega os arquivos Parquet em paralelo
    with ThreadPoolExecutor() as executor:
        dataframes = list(executor.map(load_parquet, sorted_files))

    # Filtra DataFrames válidos
    dataframes = [df for df in dataframes if df is not None]

    # Concatena todos os DataFrames em um único DataFrame
    if dataframes:
        combined_df = pl.concat(dataframes)
        return combined_df
    else:
        return pl.DataFrame()


# @st.cache_data
def load_scalar_data(parent_folder, section):
    directory = os.path.join(parent_folder, section)
    parquet_files = list(Path(directory).glob("*.parquet"))

    # def extract_timestamp(file):
    #     file_stem = file.stem
    #     try:
    #         return datetime.strptime(file_stem, "%Y_%m_%d_%H:%M:%S:%f")
    #     except ValueError as e:
    #         return None

    def load_parquet(file):
        try:
            return pl.read_parquet(file)
        except Exception as e:
            return None

    sorted_files = sorted(parquet_files, key=lambda x: x.name)

    with ThreadPoolExecutor() as executor:
        dataframes = list(executor.map(load_parquet, sorted_files))

    dataframes = [df for df in dataframes if df is not None]

    if dataframes:

        combined_df = pl.concat(dataframes)
        return combined_df
    else:
        return pl.DataFrame()



def load_audio_data(parent_folder, section):
    directory = os.path.join(parent_folder, section)
    audio_files = list(Path(directory).glob("*.wav"))
    return sorted(audio_files, key=lambda x: x.name)

# @st.cache_data
def load_classification(parent_folder, section):
    directory = os.path.join(parent_folder, section)
    parquet_files = list(Path(directory).glob("*.parquet"))
    sorted_files = sorted(parquet_files, key=lambda x: x.name)

    def load_parquet(file):
        try:
            return pl.read_parquet(file)
        except Exception as e:
            return None

    def group_by_step(df):
        grouped_data = defaultdict(lambda: {"real_label": [], "predicted_label": []})

        for row in df.to_dicts():
            step = row["step"]
            real_labels = row["real_label"]
            predicted_probs = row["predicted_label"]

            grouped_data[step]["real_label"].extend(real_labels)
            grouped_data[step]["predicted_label"].extend(predicted_probs)

        new_data = [{"step": step, "real_label": values["real_label"], "predicted_label": values["predicted_label"]}
                    for step, values in grouped_data.items()]

        return pl.DataFrame(new_data)


    with ThreadPoolExecutor() as executor:
        dataframes = list(executor.map(load_parquet, sorted_files))

    # Filtra apenas dataframes válidos
    dataframes = [df for df in dataframes if df is not None]

    # Converte tipos de dados antes de concatenar
    # dataframes = [convert_dtypes(df) for df in dataframes]

    if dataframes:
        combined_df = pl.concat(dataframes)
        return combined_df #group_by_step(combined_df)
    else:
        return pl.DataFrame()
