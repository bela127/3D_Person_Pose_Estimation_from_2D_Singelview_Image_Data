Hier wird die Vorbereitung des Datensatzes JTA beschrieben, damit er mit dem Dataset-Loader automatisch geladen werden kann.

Läd man den Datensatz von der offiziellen Quelle sind die Skripte:

*to_imgs.py* und *to_pose.py* vorgegeben und beschrieben wie man diese verwendet, diese skripte wurden angepasst und für multi-cpu systeme optimiert damit die Vorbereitung schneller geht, prinzipiell lassen sie sich aber gleich bedienen wie die offiziellen Skripte.

Die angepassten Skripte sind in: *ShAReD_Net/data/load/dataset_jta* zu finden.

Beide Skripte bieten ein CMD-Interface mit --help Argument mit dem alle Einstellungen erklärt werden.

Die Posen müssen dabei im *numpy* Format gespeichert werden.

Die Bilder können im *png* oder *jpg* Format verwendet werden, *png* hat dabei eine deutlich bessere Qualität braucht aber auch deutlich mehr Festblattenspeicher.
