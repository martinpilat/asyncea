from deap import tools

def generate_aspirant_pairs(pop_size):
    pop_idx = list(range(pop_size))
    return [tools.selRandom(pop_idx, 2) for _ in range(pop_size)]

class Generation:

    def __init__(self, pop_size, tournament_aspirants, pop=None, selected=None, offspring=None, aspirants=None, waiting=None):
        self.tournament_aspirants = tournament_aspirants
        self.pop = pop or [None] * pop_size
        self.selected = selected or [None] * pop_size
        self.offspring = offspring or [None] * pop_size
        self.aspirants = aspirants or set(sum(tournament_aspirants, []))
        self.waiting = waiting or set()
        self.aspirant_order = [None] * pop_size
        for i, (a1, a2) in enumerate(self.tournament_aspirants):
            if self.aspirant_order[a1] is None:
                self.aspirant_order[a1] = i
            if self.aspirant_order[a2] is None:
                self.aspirant_order[a2] = i