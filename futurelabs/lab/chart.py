import os
from enum import Enum
from pathlib import Path


class Type(Enum):
    Histogram2d = 1
    LineChart = 2
    AudioData = 3
    Classification = 4


def get_chart_type(parent_folder, chart):

    directory = os.path.join(parent_folder, chart, "chart_info")

    if Path(directory).exists():
        with open(directory, "r") as file:
            chart_id = file.read()

        return Type(int(chart_id)), True


    return None, True