Author: Martin Pilat, Roman Neruda
Contact: Martin.Pilat@mff.cuni.cz
GitHub Repository: https://github.com/martinpilat/asyncea/

# Parallel Evolutionary Algorithm with Interleaving Generations

## Archive contents

This archive contains the source files used to create the results in the paper 
exactly in the form in which they were used.

The main file `asyncea.py` runs all the experiments described in the paper, the
logs are stored in the directory `run_logs`.

The `parsim.py` file contains the simulator of the parallel computer, which was
used in the experiments.

The `utils.py` file contains few utilities needed in the experiments

The `algorithms.py` file contains the implementation of all the algorithms
described in the paper. The implementations are mostly compatible with the 
deap library.

## Requirements

python 3.6
deap
