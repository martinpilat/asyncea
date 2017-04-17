import math
import random

import algorithms

import json
import math

IND_SIZE = 2
MIN_POP_SIZE = 50
POP_SIZE = 100
MAX_EVALS = 100000
CXPB = 0.8
MUTPB = 0.1
SIGMA = 2.5

def rastrigin_fitness(x):
    return 10*len(x) + sum(map(lambda xi: xi*xi - 10*math.cos(2*math.pi*xi), x)),

def constant_fitness(individual):
    return 1,

def constant_tf(value, *args):
    return 1

def random_tf10(value, *args):
    return 1 + random.random() * 9

def random_tf100(value, *args):
    return 1 + random.random() * 99

def random_tf1000(value, *args):
    return 1 + random.random() * 999

def exponential_tf(value, *args):
    return random.expovariate(1.0)

def pos_correlated_tf(value, *args):
    return 1 + value[0]

def neg_correlated_tf(value, *args):
    return 1 + max(100-value[0], 0)

for ncpus in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for i in range(20):

        args = {'fitness_function': constant_fitness,
               'time_function': constant_tf,
               'max_evals': 10000,
               'max_gen': 100000,
               'n_cpus': ncpus}
        best, log = algorithms.run_clever_tournament(**args)
        with open('../run_logs/ea_clever_const_const_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
             json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_ea(**args)
        with open('../run_logs/ea_plus_const_const_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_interleaving_generations(**args)
        with open('../run_logs/inter_const_const_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
             json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_interleaving_generations(**args)
        with open('../run_logs/plus_inter_const_const_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
             json.dump(log, f, indent=1)


        args = {'fitness_function': constant_fitness,
               'time_function': random_tf100,
               'max_evals': 10000,
               'max_gen': 100000,
               'n_cpus': ncpus}
        # best, log = algorithms.run_clever_tournament(**args)
        with open('../run_logs/ea_clever_const_unif100_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_ea(**args)
        with open('../run_logs/ea_plus_const_unif100_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_interleaving_generations(**args)
        with open('../run_logs/inter_const_unif100_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_interleaving_generations(**args)
        with open('../run_logs/plus_inter_const_unif100_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)

        args = {'fitness_function': constant_fitness,
                'time_function': exponential_tf,
                'max_evals': 10000,
                'max_gen': 100000,
                'n_cpus': ncpus}
        best, log = algorithms.run_clever_tournament(**args)
        with open('../run_logs/ea_clever_const_exp_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_ea(**args)
        with open('../run_logs/ea_plus_const_exp_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_interleaving_generations(**args)
        with open('../run_logs/inter_const_exp_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_interleaving_generations(**args)
        with open('../run_logs/plus_inter_const_exp_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)

        args = {'fitness_function': rastrigin_fitness,
                'time_function': pos_correlated_tf,
                'max_evals': 10000,
                'max_gen': 100000,
                'n_cpus': ncpus}
        best, log = algorithms.run_clever_tournament(**args)
        with open('../run_logs/ea_clever_rastr_poscor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_ea(**args)
        with open('../run_logs/ea_plus_rastr_poscor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_interleaving_generations(**args)
        with open('../run_logs/inter_rastr_poscor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_interleaving_generations(**args)
        with open('../run_logs/plus_inter_rastr_poscor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_asyncea(**args)
        with open('../run_logs/async_rastr_poscor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
             json.dump(log, f, indent=1)

        args = {'fitness_function': rastrigin_fitness,
                'time_function': neg_correlated_tf,
                'max_evals': 10000,
                'max_gen': 100000,
                'n_cpus': ncpus}
        best, log = algorithms.run_clever_tournament(**args)
        with open('../run_logs/ea_clever_rastr_negcor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
        json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_ea(**args)
        with open('../run_logs/ea_plus_rastr_negcor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_interleaving_generations(**args)
        with open('../run_logs/inter_rastr_negcor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)
        best, log = algorithms.run_plus_interleaving_generations(**args)
        with open('../run_logs/plus_inter_rastr_negcor_{ncpus}_{run}.json'.format(run=i, ncpus=ncpus), 'w') as f:
            json.dump(log, f, indent=1)