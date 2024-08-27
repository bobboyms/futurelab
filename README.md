# Future Labs

## Instal the Lib
pip install git+https://github.com/bobboyms/futurelab.git@main

# Log
## Create a project
```Python
project_log = project.Project(
                project_name="Teste",
                laboratory_name="lab 1",
                work_folder="logs"
              ).log()
```

## Create a section

```Python
log_classification = project_log.new_logger(
        section_name="Classificação",
        description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
        chart_type=Type.Classification
    )
```

### Type of logs
#### log_histogram
```Python
log_gradients.log_histogram(f"Grad 2", np.random.normal(loc=0.0, scale=0.03, size=1* 128 * 400))
```

#### log_scalar
```Python
log_scalar.log_scalar("Loss", {"train":0.5 * i,"test":(0.87 * i) / 2}, i)
log_scalar.log_scalar("Acuracia", {"valor": 0.5 * i}, i)
```

#### log_audio
```Python
audio_path = '../127_sample.wav'
data, sr = librosa.load(audio_path, sr=None)
log_audio.log_audio("Amostra 1", data,sr, i)
```

#### log_classification
```Python
log_classification.log_classification("Predições", real_label=[0,0,0,1,0,1], predicted_label=[0.2,0.01,0.15,0.75,0.01,-0.1], step=1)
```



