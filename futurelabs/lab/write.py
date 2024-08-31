import os
from datetime import datetime

import polars as pl
import soundfile as sf
from sympy.stats.sampling.sample_numpy import numpy


def write_chart_info(folder, chart):
    file_path = folder / "chart_info"

    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write(str(chart.value))

def histogram(values, folder, chart):
    write_chart_info(folder, chart)

    if len(values) > 0:

        df = pl.DataFrame({
            "value": [values]
        })

        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")[:-3]
        file = folder / f"{timestamp}.parquet"
        df.write_parquet(file)


def classification(data, folder):
    write_chart_info(folder, data["chart_type"])

    df = pl.DataFrame(
        {
            "step": data["step"],
            "real_label": [data["real_label"]],
            "predicted_label": [data["predicted_label"]],
        }
    )

    # print(df)



    file = folder / f"{data['step']}.parquet"
    df.write_parquet(file)


def scalar(values, step, folder, chart):
    write_chart_info(folder, chart)

    data = {
        "step": step
    }

    for key in values:
        data[key] = values[key]

    df = pl.DataFrame(data)

    file = folder / f"{step}.parquet"
    df.write_parquet(file)


def audio(value, sr, step, folder, chart):
    write_chart_info(folder, chart)
    file = folder / f"{step}.wav"
    sf.write(file, value, sr)

