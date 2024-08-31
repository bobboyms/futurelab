import json
import os.path
import time
import threading
import queue
from datetime import datetime

from pathlib import Path
from typing import Union

import torch

from futurelabs.lab.write import histogram, scalar, audio, classification
import numpy as np
from futurelabs.lab.chart import Type
from futurelabs.lab.functions import get_section_folder, DirectoryManager, Serializable, get_project_folder

TensorOrArray = Union[torch.Tensor, np.ndarray]


def get_current_time():
    # Obter a hora atual do computador
    now = datetime.now()

    # Formatar como HH:MM:SS
    time_str = now.strftime("%H:%M:%S")
    return time_str

class Project(Serializable):
    def __init__(self, project_name: str, work_folder: str, laboratory_name=None):
        self._directory_manager = DirectoryManager()
        self.laboratory_name = laboratory_name
        self.project_name = project_name
        self.work_folder = work_folder

        self.__serialize()

    def __serialize(self):
        project_dict = self.to_dict()

        if self.laboratory_name is None:
            self.laboratory_name = get_current_time()


        folder = get_project_folder(config={
            "work_folder": self.work_folder,
            "laboratory_name": self.laboratory_name,
            "project_name": self.project_name,
        })

        self._directory_manager.create_folder_if_not_exists(folder)

        src = folder / f'{self.laboratory_name}.json'

        with open(src, 'w') as json_file:
            json.dump(project_dict, json_file, indent=4)

    def log(self) -> 'Log':
        return Log(self)



class Logger:
    def __init__(self, section_name: str, description: str, project_config: 'Project', buffer_sleep:int):
        self.section_name = section_name
        self.description = description
        self.has_folder = False
        self.project_config = project_config
        self.directory_manager = DirectoryManager()
        self.buffer_sleep = buffer_sleep
        self.section_folder = get_section_folder(config={
            "laboratory_name": self.project_config.laboratory_name,
            "project_name": self.project_config.project_name,
            "work_folder": self.project_config.work_folder,
            "section_name": self.section_name,
        })
        self.queue = queue.Queue()
        self.consumer_thread = threading.Thread(target=self.__consume)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()

    def __ensure_folder_exists(self):
        if not self.has_folder:
            self.directory_manager.create_folder_if_not_exists(self.section_folder)
            self.has_folder = True


    def log_histogram(self, description: str, value: TensorOrArray):
        self.__ensure_folder_exists()

        value = value.numpy().flatten()
        if isinstance(value, torch.Tensor):
            value = value.numpy().flatten()
        elif isinstance(value, np.ndarray):
            value = value.flatten()
        else:
            raise RuntimeError("Type not supported")

        data = {
            "chart_type": Type.Histogram2d,
            "description": description,
            "value": value.flatten(),
        }
        self.queue.put(data)


    def log_classification(self, description: str, real_label: list[int], predicted_label: list[float], step: int):

        # is_ints = all(isinstance(x, int) for x in real_label)
        # if not is_ints:
        #     raise RuntimeError("The real label needs to be a list of integer")


        data = {
            "chart_type": Type.Classification,
            "description": description,
            "real_label": real_label,
            "predicted_label": predicted_label,
            "step": step,
        }
        self.queue.put(data)

    def log_scalar(self, description: str, value: dict[str, float], step:int):
        data = {
            "chart_type": Type.LineChart,
            "description": description,
            "value": value,
            "step": step,
        }
        self.queue.put(data)

    def log_audio(self, description: str, value: np.array, sr:int,  step:int):
        data = {
            "chart_type": Type.AudioData,
            "description": description,
            "value": value,
            "sr": sr,
            "step": step,
        }
        self.queue.put(data)

    def __consume(self):
        buffer = []  # Buffer para acumular os dados
        delay_seconds = self.buffer_sleep  # Define o intervalo de tempo para gravação
        while True:
            time.sleep(delay_seconds)
            for _ in range(self.queue.qsize()):

                data = self.queue.get()

                if data is None:
                    break  # Sai do loop se receber um sinal de término

                buffer.append(data)
                self.queue.task_done()

            if buffer:
                self.__write_to_file(buffer)



    def __write_to_file(self, buffer):
        # Dicionário para agrupar histogram_values por pasta
        histogram_data_by_folder = {}
        scalar_data_by_folder = {}
        classification_data_by_folder = {}

        for data in buffer:
            # Definindo a pasta com base na "description"
            folder = Path(os.path.join(self.section_folder, data["description"]))
            self.directory_manager.create_folder_if_not_exists(folder)

            # Se for um Histogram2d, adiciona os valores ao dicionário
            if data["chart_type"] == Type.Histogram2d:

                histogram(data["value"], folder, Type.Histogram2d)

            if data["chart_type"] == Type.LineChart:

                scalar(data["value"], data["step"], folder, Type.LineChart)

            if data["chart_type"] == Type.AudioData:
                audio(data["value"], data["sr"], data["step"], folder, Type.AudioData)

            if data["chart_type"] == Type.Classification:
                classification(data, folder)


        # Processa os histogramas para cada pasta
        # for folder, histogram_values in histogram_data_by_folder.items():


        # for folder, scalar_values in scalar_data_by_folder.items():
        #     scalar(scalar_values, folder, Type.LineChart)



    def stop(self):
        self.queue.put(None)
        self.consumer_thread.join()

class Log:
    def __init__(self, project_config: Project):
        self.project_config = project_config

    def new_logger(self, section_name: str, description: str, buffer_sleep=1) -> Logger:
        return Logger(section_name, description, self.project_config, buffer_sleep=buffer_sleep)

