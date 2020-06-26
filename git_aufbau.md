Im reposotory befindet sich der *src* Ordner, dieser stellt die Wurzel allen quellcodes dar. Darunter befindet sich:

- demos
  
   - Voruntersuchungen zu einzelnen Netzbausteinen

- extern
  
   - code aus externen quellen (getestet teilweise angepasst aber final nicht verwendet)

- issues
  
   - es wurden mehrere Fehler in Tensorflow endeckt, diese wurden auf github gemeldet, zur meldung gehört immer Testcode der den Bug reproduziert, dieser Testcode ist hier zu finden

- ShAReD_Net
  
   - der code des eigendlichen Netzes sowie Trainings-Framework und Datenvorbereitung

- tests
  
   - Jpyter-Notebooks mit code zur betrachtung des Datensatzes, tests zu einzelnen Modulen, sowie trainings und evaluations Scripten

Die relevanten Ordner unterhalb dieser Struktur:

- extern
  
   - gradient-checkpointing
     
      - Implementierung von Gradientcheckpointing, nicht mit der aktuellen TensorFlow Version kombatiebel
  
   - tensorflow_update
     
      - von IBM bereitgestelltes git Patch für TensorFlow um LMS zu unterstützen

- ShAReD_Net
  
   - training
     
      - trainings framework
     
      - trainings model
     
      - loss funktionen
     
      - original trainings scripts, sind nun durch Notebooks ersetzt
  
   - model
     
      - model code, aufgeteilt in mehrere dateien und unterordner nach Aufgabenbereichen. Das Hauptmodel liegt direkt in diesem Ordner. Es ist modular aufgenaut, die einzelnen Module liegen in den unterordnern
  
   - inferenc
     
      - inference model
  
   - framework
     
      - enthält noch nichts, muss aus training und evaluation extrahiert werden
  
   - evaluation
     
      - evaluation model
     
      - evaluation framework
  
   - data
     
      - code zum laden und aufbereiten der daten aus einem datensatz
     
      - enthält unterordner die Teilaufgaben beinhalten

Datensatz code befindet sich in:

- ShAReD_Net/data
  
   - load
     
      - datensatz speziefischer code zum laden eines datensatzes
     
      - es wurde hier ein Interface geschaffen so dass nur dieser Load-Code ausgetauscht werden muss und damit ein anderer Datensatz verwendet werden kann, der restliche Code zur Datenvorverarbeitung kann dann wieder verwendet werden
  
   - transform
     
      - enthält allen code der für die Datenvorverarbeitung benötigt wird.
     
      - benötigt einen Dataset-Loader der den Datensatz mit einem speziellen Interface bereitstellt, ein Loader ist in *load* implementiert
