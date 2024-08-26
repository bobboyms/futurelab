import json
import os.path
import time
import threading
import queue

from pathlib import Path

from futurelabs import histogram, scalar, audio
import numpy as np
from futurelabs import Type
from futurelabs import get_section_folder, DirectoryManager, Serializable, get_project_folder


class Project(Serializable):
    def __init__(self, project_name: str, laboratory_name: str, work_folder: str):
        self._directory_manager = DirectoryManager()
        self.laboratory_name = laboratory_name
        self.project_name = project_name
        self.work_folder = work_folder

        self.__serialize()

    def __serialize(self):
        project_dict = self.to_dict()

        folder = get_project_folder(config={
            "work_folder": self.work_folder,
            "laboratory_name": self.laboratory_name,
            "project_name": self.project_name,
        })

        self._directory_manager.create_folder_if_not_exists(folder)

        src = folder / 'project.json'

        with open(src, 'w') as json_file:
            json.dump(project_dict, json_file, indent=4)

    def log(self) -> 'Log':
        return Log(self)



class Logger:
    def __init__(self, section_name: str, description: str, chart_type: Type, project_config: 'Project'):
        self.section_name = section_name
        self.description = description
        self.chart_type = chart_type
        self.has_folder = False
        self.project_config = project_config
        self.directory_manager = DirectoryManager()
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



    def log_histogram(self, description: str, value: np.array, subsample_ratio=0.1):
        self.__ensure_folder_exists()

        # print(len(value) * subsample_ratio)
        subsampled_data = np.random.choice(value, size=int(len(value) * subsample_ratio), replace=False)

        data = {
            "chart_type": Type.Histogram2d,
            "description": description,
            "value": subsampled_data.flatten(),
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
        delay_seconds = 10  # Define o intervalo de tempo para gravação
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

        for data in buffer:
            # Definindo a pasta com base na "description"
            folder = Path(os.path.join(self.section_folder, data["description"]))
            self.directory_manager.create_folder_if_not_exists(folder)

            # Se for um Histogram2d, adiciona os valores ao dicionário
            if data["chart_type"] == Type.Histogram2d:

                if histogram_data_by_folder.get(folder) is None:
                    histogram_data_by_folder[folder] = []

                histogram_data_by_folder[folder].append(data["value"])

            if data["chart_type"] == Type.LineChart:


                if scalar_data_by_folder.get(folder) is None:
                    scalar_data_by_folder[folder] = []

                scalar_data_by_folder[folder].append({
                    "value": data["value"],
                    "step": data["step"],
                })

            if data["chart_type"] == Type.AudioData:
                audio(data["value"], data["sr"], data["step"], folder, Type.AudioData)


        # Processa os histogramas para cada pasta
        for folder, histogram_values in histogram_data_by_folder.items():
            histogram(histogram_values, folder, Type.Histogram2d)

        for folder, scalar_values in scalar_data_by_folder.items():
            scalar(scalar_values, folder, Type.LineChart)



    def stop(self):
        self.queue.put(None)
        self.consumer_thread.join()

class Log:
    def __init__(self, project_config: Project):
        self.project_config = project_config

    def new_logger(self, section_name: str, description: str, chart_type: Type) -> Logger:
        return Logger(section_name, description, chart_type, self.project_config)

