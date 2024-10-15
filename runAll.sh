#!/bin/bash

# Numero di esecuzioni in parallelo
NUM_JOBS=100

# Path del file Python
PYTHON_SCRIPT="testing/testing.py"

# Esegui il file Python 100 volte in parallelo con il PYTHONPATH corretto
for i in $(seq 1 $NUM_JOBS); do
    PYTHONPATH=.. python3 "$PYTHON_SCRIPT" &
done

# Attendi che tutti i processi siano completati
wait

echo "Tutte le istanze di testing.py sono state eseguite."
