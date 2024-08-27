import time

import librosa
import numpy as np

from futurelabs.lab import project
from futurelabs.lab.chart import Type

# project_log = project.Project(
#     project_name="teste",
#     laboratory_name="lab 1",
#     work_folder="logs"
# ).log()
#
# project_log2 = project.Project(
#     project_name="teste wav",
#     laboratory_name="lab 2",
#     work_folder="logs"
# ).log()
#
#
#
# log_gradient = project_log.new_logger(
#     section_name = "Gradientes ",
#     description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
#     chart_type = Type.Histogram2d
# )
#
# log_loss = project_log.new_logger(
#     section_name = "Teste Loss ",
#     description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
#     chart_type = Type.Histogram2d
# )
#
# print("funcionou")
#
# for _ in range(100):
#     log_gradient.log_histogram(f"Grad 1", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))
#
#
#
#
# ###
#
# log_gradient2 = project_log2.new_logger(
#     section_name = "Gradientes ",
#     description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
#     chart_type = Type.Histogram2d
# )
#
# log_scalar = project_log2.new_logger(
#     section_name = "Loss e Acuracia",
#     description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
#     chart_type = Type.Histogram2d
# )
#
# log_audio = project_log2.new_logger(
#     section_name = "Amostras de Audio",
#     description ="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
#     chart_type = Type.Histogram2d
# )
#
# audio_path = '../127_sample.wav'
# data, sr = librosa.load(audio_path, sr=None)
# for i in range(10):
#     log_scalar.log_histogram(f"Grad 2", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))
#     log_scalar.log_scalar("Loss", {"train":0.5 * i,"test":(0.87 * i) / 2}, i)
#     log_scalar.log_scalar("Acuracia", {"valor": 0.5 * i}, i)
#
#     # log_gradient2.log_scalar("Acuracia", {"valor": 0.5 * i}, i)
#
#     log_scalar.log_audio("Amostra 1", data,sr, i)
#
#
#
# time.sleep(20)
def main():

    project_log = project.Project(
        project_name="Teste",
        work_folder="logs",
        laboratory_name="lab 2",
    ).log()

    log_classification = project_log.new_logger(
        section_name="Classificação",
        description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
        chart_type=Type.Classification
    )



    log_classification.log_classification("Predições", real_label=[0,0,0,1,0,1], predicted_label=[0.2,0.01,0.15,0.75,0.01,-0.1], step=1)
    log_classification.log_classification("Predições", real_label=[0, 1, 0, 1, 1, 0],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=1)
    log_classification.log_classification("Predições", real_label=[0, 0, 1, 1, 0, 1],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=2)
    log_classification.log_classification("Predições", real_label=[0, 0, 1, 1, 0, 1],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=3)
    log_classification.log_classification("Predições", real_label=[0, 1, 0, 0, 0, 0],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=4)
    log_classification.log_classification("Predições", real_label=[0, 1, 1, 1, 1, 1],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=5)
    log_classification.log_classification("Predições", real_label=[0, 1, 1, 0, 0, 0],
                                          predicted_label=[0.2, 0.01, 0.15, 0.75, 0.01, -0.1], step=5)



    # log_classification = project_log.new_logger(
    #     section_name="Gradientes das camadas",
    #     description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    #     chart_type=Type.Classification
    # )
    # for i in range(10):
    #     log_classification.log_histogram(f"Conv 1", np.random.normal(loc=0.0, scale=0.03, size=1 * 128 * 400))
    #     log_classification.log_histogram(f"Conv 2", np.random.normal(loc=0.0, scale=0.03, size=1 * 128 * 400) * 1.24)
    #     log_classification.log_histogram(f"Conv 3", np.random.normal(loc=0.5, scale=0.2, size=1 * 128 * 400))
    #     log_classification.log_histogram(f"Conv 4", np.random.normal(loc=0.0, scale=0.0, size=1 * 128 * 400))
    #
    # log_classification = project_log.new_logger(
    #     section_name="Amostra de audios geradas",
    #     description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    #     chart_type=Type.Classification
    # )
    # audio_path = '127_sample.wav'
    # data, sr = librosa.load(audio_path, sr=None)
    # log_classification.log_audio("Amostra 1", data, sr, 1)
    # log_classification.log_audio("Amostra 1", data, sr, 2)
    # log_classification.log_audio("Amostra 1", data, sr, 3)
    # log_classification.log_audio("Amostra 1", data, sr, 4)
    # log_classification.log_audio("Amostra 1", data, sr, 5)
    #
    # log_classification = project_log.new_logger(
    #     section_name="Dados do treinamento",
    #     description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    #     chart_type=Type.Classification
    # )
    #
    # log_classification.log_scalar("Loss", {"Teste": 0.5, "treino": 0.5}, 1)
    # log_classification.log_scalar("Loss", {"Teste": 0.49, "treino": 0.5}, 2)
    # log_classification.log_scalar("Loss", {"Teste": 0.38, "treino": 0.4}, 3)
    # log_classification.log_scalar("Loss", {"Teste": 0.30, "treino": 0.4}, 4)
    # log_classification.log_scalar("Loss", {"Teste": 0.28, "treino": 0.3}, 5)
    # log_classification.log_scalar("Loss", {"Teste": 0.29, "treino": 0.29}, 6)
    # log_classification.log_scalar("Loss", {"Teste": 0.20, "treino": 0.25}, 7)
    # log_classification.log_scalar("Loss", {"Teste": 0.27, "treino": 0.27}, 8)

    time.sleep(15)

main()