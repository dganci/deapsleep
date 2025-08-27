import pandas as pd
import numpy as np

class Epistasis:

    def __init__(self, problem, population, fitnesses):
        self.problem = problem
        self.population = population
        self.fitnesses = fitnesses
        self.history = []

    def _compute_epistasis(self, gene_idx=0):
        '''
        Compute the epistasis of a specific gene in a population.
        '''
        var_total = np.var(self.fitnesses)

        fixed_value = np.mean([
            x[gene_idx] 
            for x in self.population
        ])

        for x in self.population:
            x[gene_idx] = fixed_value

        fitness_fixed = [
            self.problem.evaluate(np.array(x).reshape(1, -1))['F'][0] 
            for x in self.population
        ]
        var_fixed = np.var(fitness_fixed)

        if var_total == 0:
            return 0.0
        return 1 - (var_fixed / var_total)

    def store_epistasis(self, gene_idx=0):
        
        self.history.append(
            self._compute_epistasis(gene_idx)
        )
    
    def show_epistasis(self):
        records = []
        for run_idx, run_data in enumerate(self.history):
            for gen_idx, epistasis_dict in enumerate(run_data):
                for gene_idx, epistasis_value in epistasis_dict.items():
                    records.append({
                        'Run': run_idx,
                        'Generation': gen_idx,
                        'Gene': gene_idx,
                        'Epistasis': epistasis_value
                    })

        df = pd.DataFrame(records)