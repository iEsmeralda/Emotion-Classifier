Para este clasificador (Emotion_Classifier_v1) se utilizaron las imágenes de SAMM (conjunto de datos de movimiento microfacial espontáneo) el cual se encuentra disponible en el siguiente link: https://www.kaggle.com/datasets/sadeeshameed/samm-v3

Para el modelo entrenado en Emotion_Classifier_v2 se utilizo el conjunto de datos obtenido en el siguiente link: https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition
Además se utilizó la base de datos Human Face Emotions disponible en el siguiente link: https://universe.roboflow.com/emotions-dectection/human-face-emotions, esta base de datos se puede utilizar con el siguiente código: 

```bash
!pip install roboflow
```

```python
from roboflow import Roboflow

rf = Roboflow(api_key="qoFwl0R0L9WDcXjtBjP2")
project = rf.workspace("emotions-dectection").project("human-face-emotions")
version = project.version(30)
dataset = version.download("tensorflow")
```

                

