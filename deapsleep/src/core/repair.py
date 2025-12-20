# repair.py

# to be filled with custom repair functions
# if necessary

def repair_start_from_zero(individual):
    '''
    Repair function that rotates a permutation individual
    so that it starts from 0. For TSP-like problems.
    '''
    if 0 in individual:
        i = individual.index(0)
        individual[:] = individual[i:] + individual[:i]
    else:
        raise ValueError(
            f'No starting position (0) found in the solution: {individual}'
        )
    return individual