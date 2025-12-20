import inspect
import numpy as np

from functools import partial
from collections.abc import Iterable

from deap import tools, algorithms
from deap.base import Toolbox
from deap.tools import HallOfFame, Logbook, ParetoFront

from deapsleep.src.core import repair as rp
from deapsleep.src.core import replacement
from deapsleep.src.dropout import *

class Evolver:
    '''
    Evolver class for DEAP-based optimization.
    '''
    def __init__(self, toolbox: Toolbox, **kwargs):
        
        self.toolbox = toolbox
        self.__dict__.update(kwargs)

        # add evaluation operator
        self.toolbox.register('evaluate', self._evaluate)

        # add evolutionary operators
        for alias, op in zip(
            ['mate', 'mutate', 'select'], 
            [self.mate, self.mutate, self.select]
        ):
            try:
                self._addOperator(
                    alias,
                    getattr(tools, op)
                )
            except AttributeError:
                raise ValueError(f"Unknown operator: {op}")
        
    def _addOperator(self, opname: str, opfunc: callable) -> None:
        '''
        Registers an evolutionary operator (i.e., selection, crossover or mutation) in the toolbox,
        automatically handling the required parameters.
        '''
        # handling repeated parameters in different operators
        OP_PARAM_MAPPING = {
            'mate': {'eta_cx': 'eta'},   
            'mutate': {'eta_mut': 'eta'}  
        }
        param_map = OP_PARAM_MAPPING.get(opname, {})

        try:
            sig = inspect.signature(opfunc)
            kwargs = {}
            for param in sig.parameters:
                if param == 'individual':
                    continue       
                if param == "low":
                    kwargs["low"] = self.problem.xl.tolist()
                    continue
                elif param == "up":
                    kwargs["up"] = self.problem.xu.tolist()
                    continue
                attr_name = next((src for src, dest in param_map.items() if dest == param), param)
                if hasattr(self, attr_name):
                    kwargs[param] = getattr(self, attr_name)
            self.toolbox.register(opname, opfunc, **kwargs)
        except AttributeError as e:
            raise ValueError(f"Missing parameter for {opname}: {e}") from e

    def _addStats(self) -> None:
        '''
        Registers statistics to be computed at fitnesses level.
        '''
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        for name, func in self.statparams.items():
            self.stats.register(name, func, axis=0)

    def _set_dropout(self, flag: bool) -> None:
        '''
        Set if to use individual-dropout with a boolean parameter.
        '''
        self.problem.use_dropout = flag

    def _set_rate(self, obj: object, rates: list[float], gen: int) -> None:
        '''
        Set the dropout rate using an annealing exponential function, for the current generation.
        Rate edges have to be specified, otherwise dropout rate will be fixed along the generations.
        '''
        if isinstance(rates, str):
            raise TypeError('Found a string. Try: --param="[start, end]".')
        start, end = rates

        H = getattr(self, 'H', 0.1*self.ngen)
        q0 = getattr(self, 'q0', 0.25)
        delta0 = getattr(self, 'delta0', 0.25)
        
        decay = 2.0 ** (-gen / H)
        curr = end + (start - end) * decay

        if np.random.rand() < q0:
            delta = np.random.uniform(0.0, delta0 * decay)
            curr += delta

        curr = np.clip(curr, 0.0, 1.0)
        setattr(obj, 'rate', curr)

    def _evaluate(self, individual: object) -> tuple[float]:
        '''
        Evaluation operator. Passes the individual to the problem's evaluation function,
        and returns the fitness(es) as a tuple of floats. Eventually, it saves
        constraint values in the individual instance to be later used.
        '''

        X = np.asarray(individual, dtype=self.problem.vtype).reshape(1, -1)

        # call to the problem evaluation operator:
        out = self.problem.evaluate(
            X,
            return_as_dictionary=True,
            use_dropout=self.problem.use_dropout
        )

        # saving the results
        F = out["F"][0] if isinstance(out["F"], (list, np.ndarray)) else out["F"]
        G = np.ravel(out.get("G", []))
        H = np.ravel(out.get("H", []))

        individual.constraints = {'G': G, 'H': H}

        # compute total violation
        _g = np.sum(np.maximum(0, G))      # inequality constraints (g(x) <= 0)
        _h = np.sum(np.abs(H))             # equality constraints (h(x) == 0)
        individual.violation = _g + _h

        # apply penalty if specified
        penalty_factor = getattr(self, 'penalty_factor', 1)
        F += float(penalty_factor) * individual.violation # assuming minimization

        mask = out.get('mask', None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            return ((F,) if np.isscalar(F) else tuple(F), list(mask))
        else:
            return ((F,) if np.isscalar(F) else tuple(F))

    def _evalInvalid(self, individuals: Iterable[object], force_all: bool=False) -> None:
        '''
        Calls the evaluation operator on individuals with invalid (i.e., absent) fitness, also saving 
        the current number of evaluations done. Eventually, it can force the evaluation of the whole population.
        '''

        if force_all:
            invalid = list(individuals)
            for ind in invalid:
                if hasattr(ind.fitness, 'values'):
                    del ind.fitness.values
        else:
            invalid = [
                ind 
                for ind in individuals 
                if not ind.fitness.valid
            ]

        # apply a before-evaluation repair if specified
        if hasattr(self, 'before_evaluation_repair'):
            repair = getattr(rp, self.before_evaluation_repair, None)
            if repair is None:
                raise ValueError(f"Unknown repair function: {self.before_evaluation_repair}")
            for ind in invalid:
                repair(ind)
        
        # evaluate the invalid individuals
        fits = self.toolbox.map(
            self.toolbox.evaluate, 
            invalid
        )
        
        for ind, fit in zip(invalid, fits):
            # dropout case
            if isinstance(fit, tuple) and len(fit) == 2 and isinstance(fit[1], (list, tuple, np.ndarray)):
                fit_vals, mask = fit
                ind.fitness.values = fit_vals
                ind.dropped_mask = np.asarray(mask, dtype=bool)
            # full evaluation case
            else:  
                ind.fitness.values = fit
                ind.dropped_mask = np.ones(self.problem.n_var, dtype=bool)

        self.evals = len(invalid)

    def _evolve(self, archive: HallOfFame | ParetoFront) -> tuple[Logbook, HallOfFame | ParetoFront]:
        '''
        Core method of the Evolver class. It runs a basic or a dropout version of the genetic algorithm for
        single- or multi-objective optimization. Depending on that, it receives a Hall of Fame or Pareto front instance,
        in which it updates the best found solutions over n generations. Also, it saves a logbook, carrying statistical
        information about the results.
        '''
        assert hasattr(self, 'replacement_operator'), \
            'Replacement operator not set.'
        
        ispopD = hasattr(self, 'popD_rate')
        ispopD_ann = ispopD and isinstance(self.popD_rate, Iterable)
        isindD = hasattr(self, 'indD_rate')
        isindD_ann = isindD and isinstance(self.indD_rate, Iterable)

        # initialization
        self.pop = self.toolbox.population()
        self._addStats()
        logbook = Logbook()
        logbook.header = ['evals'] + self.stats.fields

        if ispopD:
            popdrop = PopulationDropout()
            popD_strg = getattr(self, 'popD_strg', 1)
            setattr(popdrop, 'strategy', popD_strg)

        if isindD:    
            self.problem = IndividualDropout(self.problem)
            indD_strg = getattr(self, 'indD_strg', 'substitute')
            setattr(self.problem, 'strg', indD_strg)

        setattr(self.problem, 'use_dropout', False)
        
        # 1st evaluation (full fitness, no dropout)
        self._evalInvalid(self.pop)
        record = self.stats.compile(self.pop)
        logbook.record(evals=self.evals, **record)

        # start the evolution process
        for gen in range(1, self.ngen):

            # ----------------------------------------------------
            # 1) Set population or individual dropout if specified
            # ----------------------------------------------------
            if ispopD:
                if ispopD_ann:  # annealing
                    self._set_rate(popdrop, self.popD_rate, gen)
                else:
                    setattr(popdrop, 'rate', self.popD_rate)
                popdrop.apply(self.pop)
                _basepop = popdrop.nondropped
            else:
                _basepop = self.pop
            population = [self.toolbox.clone(ind) for ind in _basepop]

            if isindD:
                if isindD_ann: # annealing
                    self._set_rate(self.problem, self.indD_rate, gen)
                else:
                    setattr(self.problem, 'rate', self.indD_rate)

            # ---------------------------------------------------------------------------
            # 2) Evaluate under individual‐level dropout (if configured)
            #    After this call, 'population[i].fitness' is a noisy estimate via dropout
            # ---------------------------------------------------------------------------
            if isindD:
                self._set_dropout(True)
                # force_all=True ensures we re‐evaluate everyone under dropout every gen
                self._evalInvalid(population, force_all=True)

            # ------------------------------------------------------------------
            # 3) Selection (based on the dropout fitness stored in 'population')
            # ------------------------------------------------------------------
            selected = list(
                map(
                    self.toolbox.clone, 
                    self.toolbox.select(population, len(population))
                )
            )

            # ----------------------------------------------------------------------------------
            # 4) Crossover & mutation → generate offspring (still "clones" from dropout fitness)
            # ----------------------------------------------------------------------------------
            if isindD:
                # vary only nondropped variables
                varAnd = getattr(self.problem, 'varAnd')
            else:
                # vary all variables
                varAnd = getattr(algorithms, 'varAnd')

            offspring = varAnd(
                selected,
                self.toolbox,
                self.cxpb,
                self.mutpb
            )

            # apply a after-variation repair if specified
            if hasattr(self, 'after_variation_repair'):
                repair = getattr(rp, self.after_variation_repair, None)
                if repair is None:
                    raise ValueError(f"Unknown repair function: {self.after_variation_repair}")
                for ind in offspring:
                    repair(ind)

            # ------------------------------------------------------
            # 5) Evaluate offspring with "full" fitness (no dropout)
            # ------------------------------------------------------
            if isindD:
                self._set_dropout(False)
                self._evalInvalid(offspring, force_all=True)
            else:
                self._evalInvalid(offspring)

            # -----------------------------------------------------------------
            # 6) Handle population‐level dropout restoration (if applicable).
            #    Restore a part of parents into self.pop, 
            #    replacing the corresponding offspring via popdrop.restore(...).
            # -----------------------------------------------------------------
            repl = True
            if ispopD:
                self.pop = popdrop.restore(
                    offspring,
                    self.toolbox
                )
                repl = False

            # --------------------------------------------------------------
            # 7) Replacement
            # If popdrop.restore is not used, ensure a fair parent-offspring 
            # comparison by using only fully evaluated individuals.
            # --------------------------------------------------------------
            archive.update(self.pop)
            if repl:
                func = getattr(replacement, self.replacement_operator)
                if hasattr(self, 'nd'):
                    func = partial(func, nd=self.nd)           
                self.pop[:] = func(self.pop, offspring)

            # ---------------------------
            # 8) Archive update & logging
            # ---------------------------
            record = self.stats.compile(self.pop)
            logbook.record(evals=self.evals, **record)
            front = list(archive)

        return logbook, archive
        
    def soo(self) -> tuple[Logbook, HallOfFame]:
        '''
        Calls the _evolve method using an HallOfFame instance, for single-objective optimization problems.
        '''
        hof = tools.HallOfFame(self.hof)
        return self._evolve(hof)

    def moo(self) -> tuple[Logbook, ParetoFront]:
        '''
        Calls the _evolve method using a ParetoFront instance, for multi-objective optimization problems.
        '''
        pareto = tools.ParetoFront()
        return self._evolve(pareto)