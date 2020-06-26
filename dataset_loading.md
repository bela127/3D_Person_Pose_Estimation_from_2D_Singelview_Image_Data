Das laden eines Datensatzes kann einfach über einen bereitgestellten Dataset-Loader erreicht werden. Es wurde ein Beispiel-Loader für den JTA Datensatz implementiert.

Die Vorbereitung des Datensatzes ist in der entsprechenden *dataset_loader_* Datei beschrieben.

Wurde der Datensatz entsprechend vorbereitet kann der Datensatz einfach mit dem Loader geladen werden.

in der Configurationsdatei *configure.py* kann einfach der Loader ausgewählt werden und der Pfad zum vorbereiteten Datensatz gesetzt werden:

```python
config.dataset = dataset_jta
config.dataset.IMG_PATH = "/media/inferics/DataSets/Public_Datasets/JTA-Dataset/images"
config.dataset.ANNO_PATH = "/media/inferics/DataSets/Public_Datasets/JTA-Dataset/new_image_annotations"
```

Im code kann der bereits Vorverarbeitete Datensatz dann wie folgt geladen werden:

```python
import ShAReD_Net.data.transform.transform as transform

data_split = "test"
batchsize = 4
load_shuffled = False
train_ds = transform.create_dataset(data_split, batchsize, load_shuffled)
```

Wobei die Gesamtbatchsize angegeben wird (diese wird auf die GPUs verteilt und muss ein vielfaches der GPUs betragen).

Der Data-Split wird wie vom Loader gefordert als string angegeben

Es kann angegeben werden ob der Datensatz vor der Vorverarbeitung gemischt werden soll.
