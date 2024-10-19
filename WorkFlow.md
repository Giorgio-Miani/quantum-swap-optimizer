Da fare:

Da controllare:
- Modificare compatibility graph perchè funzioni solo per gruppi di moduli.
    -> è stata rimossa la classe CompatibilityGraph ed è stata creata la classe QubitMapping che gestisce il tutto. La creazione del compatibility graph per ogni gruppo di moduli avviene tramite la funzione build_compatibility_graph(...) all'interno della classe.
- Modificare i pesi del compatibility graph tenendo conto della distanza tra i qubit degli output dei moduli (la distanza va calcolata solo se i due output convergono allo stesso modulo nelle dipendenze).
- Dato un determinato modulo, implementare una funzione per la generazione di layouts in un intorno di un determinato qubit di uscita
    -> Idea: Fare sampling dei nodi della topologia a distanza minore di d (scelto da noi) dal qubit di uscita e ricercare i possibili layouts all'interno dei nodi selezionati.
- Aggiungere ai pesi degli edge del compatibility graph una componente che rappresenti il numero di SWAP necessari per spostare l'output dello step precedente in input al layout del modulo dipendente dello step corrente. (Minimizzare distanza sugli output quando servono per un modulo successivo)
- Trovare topologia simile a griglia regolare.
    -> Il backend con griglia regolare di qubit è stato creato nel file backend_gen.py
- Modificare implementazione peso associato alla distanza tra qubit di moduli con dipendenze in comune in modo tale da non considerare solamente le dipendenze al timestep immediatamente successivo ma anche le dipendenze che avvengono ai timestep seguenti.
    -> Non è necessario implementare questa parte, poiché il suo scopo è già soddisfatto dall'implementazione del peso corrispondente al numero di SWAP gate richiesti per trasferire gli output degli step precedenti agli input del modulo corrente.
- Implementazione basata sulla ricerca del max clique con algoritmo approssimato (ricerca in tempo polinomiale).

Concluse:
- Ottenere una divisione dei moduli raggruppandoli in moduli indipendenti ad ogni timestep. 
- Aggiungere condizione nella costruzione del compatibility graph tale per cui non si procede all'inserimento dell'edge quando la distanza tra 2 layout è superiore ad un certo valore.

Obiettivo: A partire dal dependency graph, l'obiettivo è identificare i moduli del circuito che possono essere eseguiti in parallelo e determinare il mapping ottimale per ciascun gruppo di moduli identificato. Questo mapping dovrebbe comportare l'introduzione del minor numero possibile di porte SWAP. Per identificare il mapping ottimale, è necessario costruire un compatibility graph in cui, ad ogni timestep, si tenga conto della distanza tra i qubit di ingresso dei moduli e i qubit di uscita del timestep precedente. Inoltre, è importante considerare la distanza tra i qubit di uscita al timestep corrente che presentano una dipendenza comune nei timestep successivi. Per ottimizzare ulteriormente l'algoritmo a livello temporale, intendiamo esplorare i possibili layout per il mapping, focalizzandoci su un intorno dei qubit di uscita dai quali un certo modulo dipende.
