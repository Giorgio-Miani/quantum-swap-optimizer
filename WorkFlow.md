Da fare:
- Fare in modo tale che venga scelto il layout che occupa più qubit sulla topologia quando vi sono più max clique con lo stesso peso.
- Aggiungere l'utilizzo di Clifford+T durante la fase di traspiling del circuito.
- Effettuare dei benchmark:
    -> metriche: depth; # totale qubit; # gate; (scomponi i gate con clifford T [lo fa qiskit] e per ciasuno estrai T-depth e T-count).
- Report/Presentazione (5-6+ pag):
    -> plottare qualche grafico "carino" ed effettuarne un'analisi sensata:
        -> usando un paper come esempio, introduci: introduzione; motivazioni; piccolo background (state of the art); tecnologie usate; algoritmi; implementazione.

Da controllare:

Concluse:
- Per ogni step temporale stabilito, implementare una strategia per gestire le situazioni in cui non ci sono sufficienti qubit disponibili per procedere con il mapping.
- Trovare una strategia per settare la reduced_distance.
    -> Settare la reduced_distance pari al numero di qubit di cui è composto ogni modulo.
- Ottenere una divisione dei moduli raggruppandoli in moduli indipendenti ad ogni timestep. 
- Aggiungere condizione nella costruzione del compatibility graph tale per cui non si procede all'inserimento dell'edge quando la distanza tra 2 layout è superiore ad un certo valore.
- Modificare compatibility graph perchè funzioni solo per gruppi di moduli.
    -> è stata rimossa la classe CompatibilityGraph ed è stata creata la classe QubitMapping che gestisce il tutto. La creazione del compatibility graph per ogni gruppo di moduli avviene tramite la funzione build_compatibility_graph(...) all'interno della classe.
- Modificare i pesi del compatibility graph tenendo conto della distanza tra i qubit degli output dei moduli (la distanza va calcolata solo se i due output convergono allo stesso modulo nelle dipendenze).
- Dato un determinato modulo, implementare una funzione per la generazione di layouts in un intorno di un determinato qubit di uscita
    -> Idea: Fare sampling dei nodi della topologia a distanza minore di d (scelto da noi) dal qubit di uscita e ricercare i possibili layouts all'interno dei nodi selezionati.
- Aggiungere ai pesi degli edge del compatibility graph una componente che rappresenti il numero di SWAP necessari per spostare l'output dello step precedente in input al layout del modulo dipendente dello step corrente. (Minimizzare distanza sugli output quando servono per un modulo successivo)
- Trovare topologia simile a griglia regolare.
    -> Il backend con griglia regolare di qubit è stato creato nel file backend_gen.py
- Implementazione basata sulla ricerca del max clique con algoritmo approssimato (ricerca in tempo polinomiale).

Testing:
- Selezionare diversi seed (quindi diversi circuiti) da utilizzare durante la fase di testing.
- Usare le metriche di qiskit per effettuare una comparatzione tra la nostra implementazione e quella associata al transpiling con qiskit sul circuito complessivo (Metriche: Numero totale di gate, numero totale di qubits, depth, T-depth e T-count).

Report (5/6 pagine): Utilizzare la struttura di un report scientifico, includendo un'introduzione al problema, una panoramica sullo stato dell'arte, la descrizione dell'implementazione e la presentazione dei risultati.

Obiettivo: A partire dal dependency graph, l'obiettivo è identificare i moduli del circuito che possono essere eseguiti in parallelo e determinare il mapping ottimale per ciascun gruppo di moduli identificato. Questo mapping dovrebbe comportare l'introduzione del minor numero possibile di porte SWAP. Per identificare il mapping ottimale, è necessario costruire un compatibility graph in cui, ad ogni timestep, si tenga conto della distanza tra i qubit di ingresso dei moduli e i qubit di uscita del timestep precedente. Inoltre, è importante considerare la distanza tra i qubit di uscita al timestep corrente che presentano una dipendenza comune nei timestep successivi. Per ottimizzare ulteriormente l'algoritmo a livello temporale, intendiamo esplorare i possibili layout per il mapping, focalizzandoci su un intorno dei qubit di uscita dai quali un certo modulo dipende.
