import copy
import types
import random
import numpy as np
from pymoo.core.problem import Problem
from deap import algorithms

class PopulationDropout:

    def __init__(self, strategy=1):
        self.strategy = strategy

    def apply(self, population):
        '''
        Apply dropout to the population.
        '''
        self.ori_pop = population
        nd, d = [], []
        for el in population:
            if random.random() >= self.rate:
                nd.append(el)
            else:
                d.append(el)
        self.nondropped, self.dropped = nd, d

    def restore(self, offspring, toolbox):
        '''
        Restore dropped individuals according to one of this strategy (to be specified):
          1 - non-dropped + dropped (best)
          2 - select(non-dropped offspring, non-dropped parents) + dropped
          3 - select(non-dropped offspring, full parents)
        '''

        if self.strategy == 1:
            # 1st strategy: nd offspring + dropped
            return list(offspring) + self.dropped

        elif self.strategy == 2:
            # 2nd strategy: select(nd offspring, nondropped) + dropped
            selected = map(
                toolbox.clone,
                toolbox.select(
                    list(offspring) + list(self.nondropped), 
                    len(self.nondropped)
                )
            )
            return list(selected) + self.dropped
        
        elif self.strategy == 3:
            # 3rd strategy: select(nd offspring, original population)
            selected = map(
                toolbox.clone,
                toolbox.select(
                    list(offspring) + list(self.ori_pop), 
                    len(self.ori_pop)
                )
            )
            return list(selected)

        else:
            raise ValueError(f"Unknown dropout replacement strategy: {self.strategy}")

class IndividualDropout(Problem):

    def __init__(self, problem: Problem):
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            n_eq_constr=problem.n_eq_constr,
            xl=problem.xl,
            xu=problem.xu,
            vtype=problem.vtype
        )
        self.problem = problem  
        self._prepare_done = False # for strategy 2

    def _generate_mask(self):
        '''
        Generate a random boolean mask.
        '''
        mask = np.random.rand(self.n_var) >= self.rate
        if not mask.any():
            idx = np.random.randint(self.n_var)
            mask[idx] = True
        return mask
    
    def _prepare(self, x):
        '''
        Identify variables that when zeroed lead to zero fitness
        '''
        crits = []
        x = np.asarray(x).reshape(-1)
        for i in range(self.problem.n_var):
            x_tmp = x.copy()
            x_tmp[i] = 0
            tmp_out = {"F": None, "G": None, "H": None}
            # evaluate single solution
            self.problem._evaluate(x_tmp.reshape(1, -1), tmp_out)
            # tmp_out["F"] may be array
            fval = tmp_out["F"]
            if np.any(fval < 1e-6):
                crits.append(i)
        self._critical_idxs = crits
        self._prepare_done = True

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        A modified version of the evaluation function, taking into account dropping variables, according to one of these strategies:
          1 - Variables removal
          2 - Preparation and variable sobstitution (best) 
        '''
        if self.use_dropout:
            # Save original number of variables
            ori_n_var = self.problem.n_var
            try:
                mask = self._generate_mask()

                if sum(mask) < 2:
                    selected = np.where(mask)[0]
                    available = np.arange(len(mask))
                    needed = 2 - len(selected)
                    extra = np.random.choice(
                        np.setdiff1d(available, selected), 
                        needed, 
                        replace=False
                    )
                    mask[extra] = True
                
                out['mask'] = mask.astype(int).tolist()

                if self.strg == 1: # variables removal
                    sub_x = x[:, mask]
                    self.problem.n_var = int(mask.sum())

                elif self.strg == 2: # variables substitution
                    # preparation phase to find critical positions
                    if not self._prepare_done:
                        self._prepare(x.copy())
                    # substitution
                    sub_x = np.asarray(x.copy()).reshape(-1)
                    for i in np.where(mask)[0]:
                        sub_x[i] = 1 if i in self._critical_idxs else 0
                    sub_x = sub_x.reshape(1, -1)

                else:
                    raise ValueError(
                        'Invalid value for individual dropout strategy. ' \
                        'Please choose 1 for variables removal '
                        'or 2 for variables substitution.' \
                    )       
                
                # Evaluate the modified individual
                self.problem._evaluate(sub_x, out, *args, **kwargs)
                p = 1 - self.rate
                corr = 1 / p

                # Apply the correction factor to the fitness and constraints
                out["F"] *= corr
                if "G" in out and out["G"] is not None:
                    G = np.array(out["G"], dtype=float)
                    G *= corr
                    out["G"] = G.tolist()
                if "H" in out and out["H"] is not None:
                    H = np.array(out["H"], dtype=float)
                    H *= corr
                    out["H"] = H.tolist()                
            finally:
                # Restore the original number of variables
                self.problem.n_var = ori_n_var
        
        else:
            self.problem._evaluate(x, out, *args, **kwargs)
        return out

    def varAnd(self, pop, toolbox, cxpb, mutpb):
        '''
        Variation operators adjusted for dropout.
        '''
        orig_m = toolbox.mate
        orig_u = toolbox.mutate

        def sel_mate(ind1, ind2):
            '''
            Crossover operator only acting on non-dropped variables
            '''
            # read masks from the individuals (fallback: all True)
            m1 = getattr(ind1, 'dropped_mask', np.ones(self.problem.n_var, dtype=bool))
            m2 = getattr(ind2, 'dropped_mask', np.ones(self.problem.n_var, dtype=bool))

            # intersection: only positions that both parents had non-dropped
            idxs = np.where(np.logical_and(m1, m2))[0]
            if idxs.size == 0:
                idxs = np.arange(self.problem.n_var)

            # perform original mate on copies, then copy back only allowed positions
            _c1, _c2 = orig_m(copy.deepcopy(ind1), copy.deepcopy(ind2))
            for i in idxs:
                ind1[i], ind2[i] = _c1[i], _c2[i]
            try:
                del ind1.fitness.values
                del ind2.fitness.values
            except AttributeError:
                pass
            return ind1, ind2

        def sel_mut(ind):
            '''
            Mutation operator only acting on non-dropped variables
            '''
            m = getattr(ind, 'dropped_mask', np.ones(self.problem.n_var, dtype=bool))
            idxs = np.where(m)[0]
            if idxs.size == 0:
                idxs = np.arange(self.problem.n_var)
            (cl, ) = copy.deepcopy(ind),
            (mu, ) = orig_u(cl)
            for i in idxs:
                ind[i] = mu[i]
            try:
                del ind.fitness.values
            except AttributeError:
                pass
            return ind,

        toolbox.mate   = types.MethodType(lambda self_tb, a, b: sel_mate(a, b), toolbox)
        toolbox.mutate = types.MethodType(lambda self_tb, a: sel_mut(a), toolbox)

        try:
            offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        finally:
            toolbox.mate   = orig_m
            toolbox.mutate = orig_u

        return offspring