from warnings import warn
from functools import wraps
import numpy as np

def feasibility(individual, eqtol=1e-6) -> bool:

    eqtol = float(eqtol)
    G = individual.constraints.get('G', None) # type: ignore
    H = individual.constraints.get('H', None)
    
    if G is None and H is None:
        return True  # Consider valid until evaluated
        
    ieq_ok = (G is None) or all(g <= 0 for g in G)
    eq_ok = (H is None) or all(abs(h) <= eqtol for h in H)
    
    return ieq_ok and eq_ok

def distance(individual):
    G = individual.constraints.get('G', [])
    H = individual.constraints.get('H', [])
    return sum(max(0, g) for g in G) + sum(abs(h) for h in H)

def manhattan_distance(feasible_ind, original_ind):
    x_f = np.array(feasible_ind, dtype=float)
    x_o = np.array(original_ind, dtype=float)
    return float(np.sum(np.abs(x_f - x_o)))

def make_closest_feasible(problem, generator, eqtol=1e-6, max_retries=100):
    """
    Returns a function `closest_feasible(ind)` that:
        1) Tries up to max_retries times to generate a feasible individual with generator()
        2) If none is feasible, clamps the individual to the box-bounds only.
    """
    xl = np.array(problem.xl)
    xu = np.array(problem.xu)

    def closest_feasible(individual):

        for _ in range(max_retries):
            cand = generator()
            if feasibility(cand, eqtol):
                return cand

        x = np.clip(np.array(individual, dtype=float), xl, xu)
        new = type(individual)(x.tolist())
        return new

    return closest_feasible

def addViolation(eqtol=1e-6):
    def decorator(func):
        @wraps(func)

        def wrapper(individual, *args, **kwargs):
            
            F = func(individual, *args, **kwargs)
            G = individual.constraints.get('G', None)
            H = individual.constraints.get('H', None)

            feas = feasibility(individual, eqtol)
            if feas:
                individual.violation = 0.0            
            else:
                individual.violation = sum(max(0, g) for g in G) + sum(abs(h) for h in H)
                
            return F
        return wrapper
    return decorator

def deathPenalty(generator, max_retries=1000):
    def decorator(func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):

            for _ in range(max_retries):
                F = func(individual, *args, **kwargs)
                if individual.violation == 0.0: # then feasible
                    return F
                newind = generator()
                individual[:] = newind[:]

            warn(
                f'\nUnable to find a feasible solution in {max_retries} attempts. '
                'Keeping the unfeasible individual.',
                RuntimeWarning
            )

            return F
        return wrapper
    return decorator

def cdp(ind1, ind2):
    feas1 = (ind1.violation == 0)
    feas2 = (ind2.violation == 0)

    if feas1 and not feas2:
        return ind1
    if feas2 and not feas1:
        return ind2

    if not feas1 and not feas2:
        return ind1 if ind1.violation < ind2.violation else ind2

    return ind1 if ind1.fitness < ind2.fitness else ind2