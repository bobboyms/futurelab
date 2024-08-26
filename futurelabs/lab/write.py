from datetime import datetime

import polars as pl
import soundfile as sf


def write_chart_info(folder, chart):
    file_path = folder / "chart_info"
    with open(file_path, "w") as file:
        file.write(str(chart.value))

def histogram(values, folder, chart):

    if len(values) > 0:
        write_chart_info(folder, chart)

        df = pl.DataFrame({
            "value": values
        })

        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")[:-3]
        file = folder / f"{timestamp}.parquet"
        df.write_parquet(file)


def classification(data, folder):
    write_chart_info(folder, data["chart_type"])

    df = pl.DataFrame({
        "step": data["step"],
        "real_label": [data["real_label"]],
        "predicted_label": [data["predicted_label"]],
    })

    file = folder / f"{data['step']}.parquet"
    df.write_parquet(file)

def scalar(values, folder, chart):

    write_chart_info(folder, chart)

    steps = []
    metrics_columns = {}

    # Iterar sobre cada dicionário na lista e organizar em colunas
    for entry in values:
        steps.append(entry["step"])  # Adiciona o valor de step à lista steps
        for key, value in entry["value"].items():
            if key not in metrics_columns:
                metrics_columns[key] = []
            metrics_columns[key].append(value)

    # Criar um dicionário organizado com os dados em colunas
    organized_data = {
        "step": steps,
        **metrics_columns  # Expande as colunas de metrics dinamicamente
    }

    # Criar o DataFrame Polars
    df = pl.DataFrame(organized_data)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")[:-3]
    file = folder / f"{timestamp}.parquet"
    df.write_parquet(file)


def audio(value, sr, step, folder, chart):
    write_chart_info(folder, chart)
    file = folder / f"{step}.wav"
    sf.write(file, value, sr)

