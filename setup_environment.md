zum aufsetzen der Arbeitsumgebung bitte wie folgt vorgehen:

Clone das Reposotory, speichere es in *Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data*

```
git clone https://github.com/bela127/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data.git
```

beziehe das neuste Tensorflow docker image

```
docker pull tensorflow/tensorflow:nightly-gpu-py3-jupyter
```

erzeuge einen docker container der unter dem momentanen user ausgeführt wird, nutze alle GPUs, gebe ports für jupyter und tensorboard frei, binde den Datensatz Ortner ein, lege Arbeitsverzeichnis bei start fest und nenne den Container pose3D

```
docker run -u $(id -u):$(id -g) -it --gpus all -p 8888:8888 -p 6006:6006 -v /media/inferics/DataSets/Public_Datasets/JTA-Dataset/:/dataset/jta -v ~/Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data:/tf/pose3D -w /tf/pose3D --name pose3D tensorflow/tensorflow:nightly-gpu-py3-jupyter
```

liste alle container die existieren auf

```
docker container ls
```

starte den erstellten container

```
docker start pose3D
```

starte ein interaktives bash als momentaner user

```
docker exec -it pose3D bash
```

manchmal braucht man root rechte im container, öffne so ein root bash

```
docker exec -it -u root:root pose3D bash
```

manchmal wird man vom notebook server abgemeldet, um das Zugangstoken wieder zu erhalten, lisste alle notebook server auf die im container laufen

```
docker exec -it pose3D jupyter notebook list
```

führe nvidia-smi aus um GPU infos zu erhalten

```
docker exec -it pose3D nvidia-smi
```

führe nvidia-smi aus und update es alle 0.2 Sekunden um den echtzeit GPU-Status zu überprüfen

```
docker exec -it pose3D watch -n 0.2 nvidia-smi
```

Stoppe den container

```
docker stop pose3D
```

Lösche den container um einen neuen container zu erzeugen

```
docker rm pose3D
```

Starte Tensorboard auf allen IP-Adressen damit es auch von außen erreichbar ist

```
tensorboard --host 0.0.0.0 --logdir=./logdir
```

Fals eine GPU abgestürtzt ist, resete sie

```
sudo nvidia-smi --gpu-reset
sudo nvidia-smi --gpu-reset -i 0
```

Da nur alle GPUs genutzt werden können wenn keine GUI/X-Server läuft sorge dafür dass die workstation immer ohne X startet

In order to make text boot the default under systemd (regardless of which distro, really):

```
systemctl set-default multi-user.target
```

Fals doch mal wieder GUI zugriff nötig wird

To change back to booting to the GUI,

```
systemctl set-default graphical.target
```

Erhalte Systeminformationen über die Workstation

```
Inxi -F
```

wenn Linux aufgrund von speicherproblemen wichtige Prozesse abschiest, setze die prioritätsstuffe für diese Prozesse höher, so dass sie nicht mehr abgeschossen werden

```
echo -17 > /proc/$PID/oom_adj
```

Sollte IBM Large-Model-Support für die neuste Tensorflow Version anbieten können deutlich größere Modelle trainiert werden, LMS ist nur über conda beziebar und es müssen dafür folgende Conda-Cannel hinzugefügt werden

```
conda config --env --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
```

```
conda config --env --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access
```
