# Future Labs

## Instal the Lib
Para instalar a biblioteca Future Labs, utilize o comando abaixo:
```Bash
pip install git+https://github.com/bobboyms/futurelab.git@main
```

# Log
A biblioteca Future Labs oferece uma estrutura para monitorar e registrar experimentos de machine learning. Você pode criar projetos, seções, e registrar diferentes tipos de dados como histogramas, scalars, áudios e classificações.
## Create a project
O primeiro passo é criar um projeto onde os logs serão armazenados. O projeto serve como um contêiner para todas as seções e logs relacionados a um experimento.
```Python
project_log = project.Project(
                project_name="Teste",
                laboratory_name="lab 1",
                work_folder="logs"
              ).log()
```

## Create a section
Depois de criar um projeto, você pode criar seções dentro dele. Uma seção pode ser usada para agrupar logs de um mesmo tipo ou relacionados a um aspecto específico do experimento.
```Python
log_classification = project_log.new_logger(
        section_name="Classificação",
        description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
        chart_type=Type.Classification
    )
```

### Type of logs
Dentro de uma seção, você pode registrar diferentes tipos de logs. Abaixo estão os tipos suportados:

#### log_histogram
Usado para registrar dados em forma de histograma, geralmente utilizado para monitorar distribuições de valores como gradientes ou ativações.

```Python
log_gradients.log_histogram(f"Grad 2", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))
```

#### log_scalar
Usado para registrar valores numéricos simples, como perda (loss) ou acurácia ao longo de várias iterações ou épocas.

```Python
log_scalar.log_scalar("Loss", {"train":0.5 * i,"test":(0.87 * i) / 2}, i)
log_scalar.log_scalar("Acuracia", {"valor": 0.5 * i}, i)
```

#### log_audio
Usado para registrar amostras de áudio. Pode ser útil para monitorar a qualidade de saídas de modelos que geram áudio.
```Python
audio_path = '../127_sample.wav'
data, sr = librosa.load(audio_path, sr=None)
log_audio.log_audio("Amostra 1", data,sr, i)
```

#### log_classification
Usado para registrar resultados de classificações, comparando rótulos reais com predições.


```Python
log_classification.log_classification("Predições", real_label=[0,0,0,1,0,1], predicted_label=[0.2,0.01,0.15,0.75,0.01,-0.1], step=1)
```



