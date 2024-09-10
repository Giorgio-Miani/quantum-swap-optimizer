- costruzione circuito
- dividere circuito in set di moduli parallelizzabili tra di loro.
- Mapomatic per calcolare per ogni modulo una lista di layout sul circuito con i relativi scores
- calcola distanza tra layout di ogni modulo (algoritmo 2 del paper)
    - calcola distanza tra qubit
- costruire control flow graph con relative distanze calcolate
- crea compatibility graph dove ogni nodo è un layout di un modulo e due nodi sono collegati se NON sono b-overlapping, ogni lato è pasato sulla base del mapomatic score e di P (vedere note)
- trovare il massimo clique del grafo (massismo sottogarfo completo), massimizzando i pesi

PROBLEMA: nel paper i circuiti sono indipendenti, noi vogliamo che moduli che interagiscano tra di loro siano vicini