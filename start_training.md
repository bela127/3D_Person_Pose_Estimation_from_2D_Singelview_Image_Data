Das Training kann einfach gestartet werden indem das notebook *tests/Slim_Training.ipynb* gestartet und ausgeführt wird.

Hier sind alle für das Trainings-Framework nötigen Callbacks implementiert, mit diesen kann das Training angepasst werden.

Das Laden des Models dauert auf manchen Systemen sehr lange! (manchmal über 30min) aufgrund von Graphoptimierungen die TensorFlow vornimmt. Also immer geduldig warten.

Standart Hyperparameter fürs Training sind alle in der Configurationsdatei in *ShAReD_Net/configure.py* angegeben.

Diese Konfiguration kann einfach geladen und andere Parameter gesetzt werden.

```python
from ShAReD_Net.configure import config

config.dataset.IMG_PATH = "/dataset/jta/images_jpg"
config.dataset.ANNO_PATH = "/dataset/jta/new_image_annotations"

config.checkpoint.path = "/tf/pose3D/checkpoints/run18_hard_batch_loss"
config.tensorboard.path = "/tf/pose3D/logdir/run18_hard_batch_loss"
```

über die *set_config(config)* Methode kann die Konfiguration mit einem Dictonary geupdated werden.

Ein gutes Vorgehen ist es in einer separaten Datei die Konfiguration zu laden und alle Einstellungen einzeln zu setzen. Durch Versionierung dieser Datei für jedes Experiment sind so immer alle Hyperparameter eines Experiments an einem Ort gespeichert.
