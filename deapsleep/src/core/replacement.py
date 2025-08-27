from deap.tools import selBest, selTournament, selWorst, selNSGA2
import random

def elitism(pop, offspring):
    '''
    Reinserts archived elites into the offspring by replacing the worst individuals
    '''
    n_elites = len(pop)//10 # 10% elites
    elites = selBest(pop, n_elites)
    sorted_off = sorted(
        offspring, 
        key=lambda ind: ind.fitness.values[0]
    )
    survivors = sorted_off[:len(pop) - len(elites)]
    return survivors + elites

def generational(pop, offspring) -> list:
    '''
    Replaces the entire parent population with the new offspring.
    '''
    assert len(pop) == len(offspring), \
        'Population and offspring must have the same size'
    pop = offspring[:]
    return pop

def mu_plus_lambda(pop, offspring) -> list:
    '''
    (μ + λ): selects the best μ from parents and offspring
    '''
    combined = pop + offspring
    pop = selBest(combined, len(pop))
    return pop

def mu_plus_lambda_nsga2(pop, offspring, nd='standard') -> list:
    combined = pop + offspring
    pop = selNSGA2(combined, len(pop), nd=nd)
    return pop

def mu_comma_lambda(pop, offspring) -> list:
    '''
    (μ, λ): selects the best μ individuals from offspring only
    '''
    pop = selBest(offspring, len(pop))
    return pop

def steady_state(pop, offspring, k=1) -> list:
    '''
    Iteratively replaces the worst individuals in the population with each new offspring.
    '''
    pop = pop[:]
    worst_idxs = list(range(len(pop)))
    for child in offspring:
        worst = selWorst(pop, k)[-1] 
        idx = pop.index(worst)
        pop[idx] = child
    return pop

def random_replacement(pop, offspring) -> list:
    '''
    Replaces random members of the current population with new offspring.
    '''
    pop = pop[:]

    slots = random.sample(range(len(pop)), len(offspring))
    for slot, child in zip(slots, offspring):
        pop[slot] = child

    return pop

def rand_tourn_repl(pop, offspring, k=1):
    
    pop = pop[:]   
    off = offspring[:]   

    for _ in range(k):

        if not pop or not off:
            break

        child = random.choice(off)
        off.remove(child)

        rival = random.choice(pop)
        pop.remove(rival)

        winner = selTournament(
            [child, rival],      
            k=1,              
            tournsize=2,    
            fit_attr='fitness'
        )[0]
        pop.append(winner)

    return pop

def iter_tourn_repl(pop, offspring, k=1):
    
    pop = pop[:]
    off = offspring[:]

    for _ in range(k):

        if not pop or not off:
            break

        for child in off:

            rival = random.choice(pop)
            pop.remove(rival)

            winner = selTournament(
                [child, rival],
                k=1,
                tournsize=2,
                fit_attr='fitness'
            )[0]

            pop.append(winner)

    return pop