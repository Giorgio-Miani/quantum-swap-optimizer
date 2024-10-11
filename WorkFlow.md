- Ottenere una divisione dei moduli raggrupandoli in moduli parallelizzabili
- modificare compatibility graph perchè funzioni solo per gruppi di moduli
- modificare il peso del compatibility graph tenendo conto della distanza tra i qubit degli output dei moduli (la distanza va calcolata solo se i due output convergono allo stesso modulo nelle dipendenze).
- capire come ottenere un intorno dei qubit di output dalla topologia.
- capire come trovare i layout solo nell'intorno della topologia.
- trovare max clique tenendo conto della distanza degli output (e del numero di swaps) tra gli output trovati

Obiettivo: partire dalla topologia intera e posizionare i moduli in parallelo andando a minimizzare la distanza degli output per moduli con una dipendenza succesiva in comune (utilizzando il max clique gia creato). Poi per ogni gruppo parallelo, prima mi ricavo l'intorno degli input nella topologia anche in base al numero di qubit del modulo corrente, poi cerco i layout nell'intorno, per ogni layout ottenuta cerco di minimizzare di nuovo la distanza tra i qubit in uscita con dipendenza in comune (e con il minimo numero di swaps). Continuo fino alla fine dei moduli.
