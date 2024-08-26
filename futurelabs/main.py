import time

import librosa
import numpy as np

from futurelabs import project
from futurelabs import Type

project_log = project.Project(
    project_name="teste",
    laboratory_name="lab 1",
    work_folder="logs"
).log()

project_log2 = project.Project(
    project_name="teste wav",
    laboratory_name="lab 2",
    work_folder="logs"
).log()



log_gradient = project_log.new_logger(
    section_name = "Gradientes ",
    description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    chart_type = Type.Histogram2d
)

log_loss = project_log.new_logger(
    section_name = "Teste Loss ",
    description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    chart_type = Type.Histogram2d
)

for _ in range(100):
    log_gradient.log_histogram(f"Grad 1", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))




###

log_gradient2 = project_log2.new_logger(
    section_name = "Gradientes ",
    description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    chart_type = Type.Histogram2d
)

log_scalar = project_log2.new_logger(
    section_name = "Loss e Acuracia",
    description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    chart_type = Type.Histogram2d
)

log_audio = project_log2.new_logger(
    section_name = "Amostras de Audio",
    description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    chart_type = Type.Histogram2d
)

audio_path = '../127_sample.wav'
data, sr = librosa.load(audio_path, sr=None)
for i in range(10):
    log_scalar.log_histogram(f"Grad 2", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))
    log_scalar.log_scalar("Loss", {"train":0.5 * i,"test":(0.87 * i) / 2}, i)
    log_scalar.log_scalar("Acuracia", {"valor": 0.5 * i}, i)

    # log_gradient2.log_scalar("Acuracia", {"valor": 0.5 * i}, i)

    log_scalar.log_audio("Amostra 1", data,sr, i)



time.sleep(20)