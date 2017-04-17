import datetime
import random
from operator import attrgetter

from deap import base, tools, creator, algorithms

from parsim import ParallelSimulator
import utils
from utils import Generation

IND_SIZE = 5


def run_asyncea(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    MIN_POP_SIZE = pop_size // 2

    toolbox = create_toolbox(fitness_function)

    pop = toolbox.population(n=pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)
    for ind in pop:
        pool.submit(ind)

    pop = [] # remove the individuals from the pop, will get them again once they are evaluated

    start = datetime.datetime.now()
    last = start

    best_ind = None

    while pool.evals < max_evals:

        result = pool.next_finished()

        result.args[0].fitness.values = result.value
        ind = result.args[0]
        pop.append(result.args[0])

        if not best_ind or ind.fitness > best_ind.fitness:
            best_ind = ind

        if len(pop) < MIN_POP_SIZE:
            continue

        if len(pop) > pop_size:
            pop = list(sorted(pop, key=attrgetter('fitness')))[1:]

        valid = True
        offspring = None
        while valid:
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, 2))
            # Apply crossover and mutation to the individuals; the operators create 2 offspring, we need only 1
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)[0]
            valid = offspring.fitness.valid
            if valid and len(pop) > pop_size:
                pop = sorted(pop, key=attrgetter('fitness'))[1:]

        pool.submit(offspring)

        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            t = (now - start).total_seconds()
            print('TIME:', t, 'SPEED:', pool.evals/t)
            last = now

    end = datetime.datetime.now()
    print(max_evals / (end - start).total_seconds())
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def run_ea(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    toolbox = create_toolbox(fitness_function)
    initial_pop = toolbox.population(n=pop_size)
    aspirant_pairs = utils.generate_aspirant_pairs(pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)

    start = datetime.datetime.now()
    last = start

    gens = [Generation(pop_size, tournament_aspirants=aspirant_pairs, pop=initial_pop)]

    t = 0
    best_ind = None

    while pool.evals < max_evals:

        # submit individuals for current generation
        for idx, ind in enumerate(gens[t].pop):
            if not ind.fitness.valid:
                ind.pop_idx = idx
                pool.submit(ind)

        while not pool.all_evaluations_finished():
            result = pool.next_finished()
            ind = result.args[0]
            gens[t].pop[ind.pop_idx].fitness.values = result.value
            if not best_ind or ind.fitness > best_ind.fitness:
                best_ind = ind

        pop = gens[t].pop

        # finish tournament selection
        gens[t].selected = [max(pop[asp1], pop[asp2], key=attrgetter('fitness')) for asp1, asp2 in gens[t].tournament_aspirants]

        # run operators
        gens[t].offspring = map(toolbox.clone, gens[t].selected)
        gens[t].offspring = algorithms.varAnd(gens[t].offspring, toolbox, cxpb, mutpb)

        # create next generation
        gens.append(Generation(pop_size, tournament_aspirants=utils.generate_aspirant_pairs(pop_size), pop=gens[t].offspring[:]))

        t += 1

        # print logging information every second
        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            runtime = (now - start).total_seconds()
            print('TIME:', runtime, 'SPEED:', pool.evals / runtime)
            last = now

    end = datetime.datetime.now()
    print(pool.evals / (end - start).total_seconds())
    print(pool.evals)
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def run_plus_ea(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    toolbox = create_toolbox(fitness_function)
    initial_pop = toolbox.population(n=pop_size)
    aspirant_pairs = utils.generate_aspirant_pairs(pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)

    start = datetime.datetime.now()
    last = start

    gens = [Generation(pop_size, tournament_aspirants=aspirant_pairs, pop=initial_pop)]

    t = 0
    best_ind = None

    # submit individuals for current generation
    for idx, ind in enumerate(gens[t].pop):
        if not ind.fitness.valid:
            ind.pop_idx = idx
            pool.submit(ind)

    while not pool.all_evaluations_finished():
        result = pool.next_finished()
        ind = result.args[0]
        gens[0].pop[ind.pop_idx].fitness.values = result.value
        if not best_ind or ind.fitness > best_ind.fitness:
            best_ind = ind

    while pool.evals < max_evals:

        pop = gens[t].pop

        # finish tournament selection
        gens[t].selected = [max(pop[asp1], pop[asp2], key=attrgetter('fitness')) for asp1, asp2 in gens[t].tournament_aspirants]

        # run operators
        gens[t].offspring = map(toolbox.clone, gens[t].selected)
        gens[t].offspring = algorithms.varAnd(gens[t].offspring, toolbox, cxpb, mutpb)

        for idx, ind in enumerate(gens[t].offspring):
            ind.pop_idx = idx
            if not ind.fitness.valid:
                pool.submit(ind)

        while not pool.all_evaluations_finished():
            result = pool.next_finished()
            ind = result.args[0]
            gens[t].offspring[ind.pop_idx].fitness.values = result.value
            if ind.fitness > best_ind.fitness:
                best_ind = ind

        next_pop = sorted(gens[t].pop + gens[t].offspring, key=attrgetter('fitness'))[pop_size:]

        # create next generation
        gens.append(Generation(pop_size, tournament_aspirants=utils.generate_aspirant_pairs(pop_size), pop=next_pop[:]))

        t += 1

        # print logging information every second
        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            runtime = (now - start).total_seconds()
            print('TIME:', runtime, 'SPEED:', pool.evals / runtime)
            last = now

    end = datetime.datetime.now()
    print(pool.evals / (end - start).total_seconds())
    print(pool.evals)
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def run_clever_tournament(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    toolbox = create_toolbox(fitness_function)
    initial_pop = toolbox.population(n=pop_size)
    aspirant_pairs = utils.generate_aspirant_pairs(pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)

    start = datetime.datetime.now()
    last = start

    gens = [Generation(pop_size, tournament_aspirants=aspirant_pairs, pop=initial_pop)]

    t = 0
    best_ind = None

    while pool.evals < max_evals:

        # submit individuals for current generation
        for asp in gens[t].aspirants:
            ind = gens[t].pop[asp]
            if not ind.fitness.valid:
                ind.pop_idx = asp
                ind.gen = t
                pool.submit(ind)

        while not pool.all_evaluations_finished():
            result = pool.next_finished()
            ind = result.args[0]
            gens[ind.gen].pop[ind.pop_idx].fitness.values = result.value
            if not best_ind or ind.fitness > best_ind.fitness:
                best_ind = ind

        # finish tournament selection
        gens[t].selected = [max(gens[t].pop[asp1], gens[t].pop[asp2], key=attrgetter('fitness')) for asp1, asp2 in gens[t].tournament_aspirants]

        # run operators
        gens[t].offspring = map(toolbox.clone, gens[t].selected)
        gens[t].offspring = algorithms.varAnd(gens[t].offspring, toolbox, cxpb, mutpb)

        # create next generation
        gens.append(Generation(pop_size, tournament_aspirants=utils.generate_aspirant_pairs(pop_size), pop=gens[t].offspring[:]))

        t += 1

        # print logging information every second
        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            runtime = (now - start).total_seconds()
            print('TIME:', runtime, 'SPEED:', pool.evals / runtime)
            last = now

    end = datetime.datetime.now()
    print(pool.evals / (end - start).total_seconds())
    print(pool.evals)
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def propagate(gens, ind, toolbox, pool, pop_size, cxpb, mutpb):
    working_gen = gens[ind.gen]
    asp_idx = [i for i, (a1, a2) in enumerate(working_gen.tournament_aspirants) if a1 == ind.pop_idx or a2 == ind.pop_idx]
    for a in asp_idx:
        a1, a2 = working_gen.tournament_aspirants[a]
        if working_gen.pop[a1] and working_gen.pop[a2] and working_gen.pop[a1].fitness.valid and working_gen.pop[a2].fitness.valid:
            working_gen.selected[a] = max(working_gen.pop[a1], working_gen.pop[a2], key=attrgetter('fitness'))
            if a % 2 == 0:
                start, stop = a, a + 2
            else:
                start, stop = a - 1, a + 1
            if working_gen.selected[start] is not None and working_gen.selected[stop - 1] is not None and working_gen.offspring[start] is None and working_gen.offspring[stop - 1] is None:
                working_gen.offspring[start:stop] = map(toolbox.clone, working_gen.selected[start:stop])
                working_gen.offspring[start:stop] = algorithms.varAnd(working_gen.offspring[start:stop], toolbox, cxpb, mutpb)
                if len(gens) <= ind.gen + 1:
                    gens.append(Generation(pop_size, tournament_aspirants=utils.generate_aspirant_pairs(pop_size), pop=[None] * pop_size))
                gens[ind.gen + 1].pop[start:stop] = working_gen.offspring[start:stop]  # comma selection
                for idx, i in enumerate(gens[ind.gen + 1].pop[start:stop]):
                    i.pop_idx = start + idx
                    i.gen = ind.gen + 1
                    if not i.fitness.valid and i.pop_idx in gens[i.gen].aspirants:
                        pool.submit(i, order=(i.gen, gens[i.gen].aspirant_order[i.pop_idx]))
                        gens[i.gen].waiting.add(i.pop_idx)
                    else:
                        propagate(gens, i, toolbox, pool, pop_size, cxpb, mutpb)


def run_interleaving_generations(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    toolbox = create_toolbox(fitness_function)
    initial_pop = toolbox.population(n=pop_size)
    aspirant_pairs = utils.generate_aspirant_pairs(pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)

    start_time = datetime.datetime.now()
    last = start_time

    initial_generation = Generation(pop_size, tournament_aspirants=aspirant_pairs, pop=initial_pop)
    gens = [initial_generation]

    for asp in initial_generation.aspirants: # submit the initial generation for evaluation
        ind = initial_generation.pop[asp]
        ind.pop_idx = asp
        ind.gen = 0
        pool.submit(ind, order=(0, initial_generation.aspirant_order[asp]))
        initial_generation.waiting.add(ind.pop_idx)

    best_ind = None

    while pool.evals < max_evals:
        result = pool.next_finished()
        ind = result.args[0]
        working_gen = gens[ind.gen]
        ind.fitness.values = result.value
        if not best_ind or ind.fitness > best_ind.fitness:
            best_ind = ind
        working_gen.waiting.remove(ind.pop_idx)
        # find all pairs of parents containing this individual and if both are evaluated, perform selection
        propagate(gens, ind, toolbox, pool, pop_size, cxpb, mutpb)

        # print logging information every second
        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            runtime = (now - start_time).total_seconds()
            print('TIME:', runtime, 'SPEED:', pool.evals / runtime)
            last = now

    end = datetime.datetime.now()
    print(pool.evals / (end - start_time).total_seconds())
    print(pool.evals)
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def propagate_plus(gens, ind_gen, toolbox, pool, pop_size, cxpb, mutpb):
    working_gen = gens[ind_gen] # take the current generation

    if len(gens) == ind_gen + 1: # if we do not have the next generation, expand the list of generations
        g = Generation(pop_size, tournament_aspirants=utils.generate_aspirant_pairs(pop_size))
        g.in_pop = set()
        gens.append(g)

    next_gen = gens[ind_gen + 1] # the next generation

    # perform the partial plus selection (select parents and offspring which are definitely selected, remember index and parent/offspring info
    evaluated_parents = [((0, idx), par) for idx, par in enumerate(working_gen.pop) if par and par.fitness.valid]
    evaluated_offspring = [((1, idx), off) for idx, off in enumerate(working_gen.offspring) if off and off.fitness.valid]
    evaluated = evaluated_offspring + evaluated_parents
    if len(evaluated) < pop_size:  # not enough evaluated individuals, cannot select anything
        return
    sorted_evaluated = sorted(evaluated, key=lambda x: x[1].fitness)
    selected = sorted_evaluated[pop_size:]
    new_selected = [i for i in selected if i[0] not in next_gen.in_pop]  # these are the new individuals certain to be selected

    for i in new_selected:  # add each newly selected individual as parent to the next generation
        index = next_gen.pop.index(None)  # replace the first free space in population with a clone of this individual
        next_gen.pop[index] = toolbox.clone(i[1])
        next_gen.in_pop.add(i[0])

        relevant_aspirants = [(idx, (asp1, asp2)) for idx, (asp1, asp2) in enumerate(next_gen.tournament_aspirants) if next_gen.pop[asp1] and next_gen.pop[asp2] and not next_gen.selected[idx]]
        for idx, (asp1, asp2) in relevant_aspirants:  # check if we can perform a selection
            next_gen.selected[idx] = max([next_gen.pop[asp1], next_gen.pop[asp2]], key=attrgetter('fitness')) # perform the tournament selection
            if idx % 2 == 0: # compute the start and end indices for offspring used in genetic operators
                start, stop = idx, idx + 2
            else:
                start, stop = idx - 1, idx + 1
            # perform the genetic operators if both individuals are available and operators have not yet been performed
            if next_gen.selected[start] is not None and next_gen.selected[stop - 1] is not None and next_gen.offspring[start] is None and next_gen.offspring[stop - 1] is None:
                next_gen.offspring[start:stop] = map(toolbox.clone, next_gen.selected[start:stop])
                next_gen.offspring[start:stop] = algorithms.varAnd(next_gen.offspring[start:stop], toolbox, cxpb, mutpb)
                # evaluate the new individuals if it is needed
                for pos, new_ind in enumerate(next_gen.offspring[start:stop]):
                    new_ind.pop_idx = start + pos
                    new_ind.gen = ind_gen + 1
                    if not new_ind.fitness.valid:
                        pool.submit(new_ind, order=(new_ind.gen,))
                        next_gen.waiting.add(new_ind.pop_idx)
        propagate_plus(gens, ind_gen + 1, toolbox, pool, pop_size, cxpb, mutpb) # we added a new parent, propagate the info


def run_plus_interleaving_generations(fitness_function, time_function, max_evals, max_gen, n_cpus=50, pop_size=100, cxpb = 0.8, mutpb= 0.1):

    toolbox = create_toolbox(fitness_function)
    initial_pop = toolbox.population(n=pop_size)
    dummy_parents = toolbox.population(n=pop_size) # these are never used or evaluated, they are just to ease the implementation of first generation
    aspirant_pairs = utils.generate_aspirant_pairs(pop_size)

    pool = ParallelSimulator(n_cpus=n_cpus, eval_func=toolbox.evaluate, time_func=time_function)

    start_time = datetime.datetime.now()
    last = start_time

    for ind in dummy_parents: # set large fitness for dummy parents so they are worse than all offspring in the initial generation
        ind.fitness.values = (1e10, )

    initial_generation = Generation(pop_size, tournament_aspirants=aspirant_pairs, pop=dummy_parents, offspring=initial_pop)
    initial_generation.in_pop = set()
    gens = [initial_generation]

    # submit all individuals from initial population
    for idx, ind in enumerate(initial_generation.offspring):
        ind.pop_idx = idx
        ind.gen = 0
        pool.submit(ind, order=(0, ))
        initial_generation.waiting.add(idx)

    best_ind = None

    while pool.evals < max_evals:
        result = pool.next_finished()
        ind = result.args[0]
        working_gen = gens[ind.gen]
        ind.fitness.values = result.value
        if not best_ind or ind.fitness > best_ind.fitness:
            best_ind = ind
        working_gen.waiting.remove(ind.pop_idx)
        # propagete the evaluated individual
        propagate_plus(gens, ind.gen, toolbox, pool, pop_size, cxpb, mutpb)

        # print logging information every second
        now = datetime.datetime.now()
        if now - last > datetime.timedelta(seconds=1):
            runtime = (now - start_time).total_seconds()
            print('TIME:', runtime, 'SPEED:', pool.evals / runtime)
            last = now

    end = datetime.datetime.now()
    print(pool.evals / (end - start_time).total_seconds())
    print(pool.evals)
    print(pool.current_time)
    pool.print_stats()

    return best_ind, pool.log


def create_toolbox(fitness_function):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", fitness_function)

    return toolbox
