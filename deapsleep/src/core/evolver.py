import inspect
import numpy as np

from functools import partial
from collections.abc import Iterable

from deap import tools, algorithms
from deap.base import Toolbox
from deap.tools import HallOfFame, Logbook, ParetoFront

from deapsleep.src.core import replacement
from deapsleep.src.core.constraint import *
from deapsleep.src.dropout import *

class Evolver:
    '''
    Evolver class for DEAP-based optimization.
    '''
    def __init__(self, toolbox: Toolbox, **kwargs):
        
        self.toolbox = toolbox
        self.__dict__.update(kwargs)

        # add evaluation (and constraint handling) operators
        self._addEvaluation()

        # add evolutionary operators
        for alias, op in zip(
            ['mate', 'mutate', 'select'], 
            [self.mate, self.mutate, self.select]
        ):
            self._addOperator(
                alias,
                getattr(tools, op)
            )

    def _addEvaluation(self) -> None:
        '''
        Adds the evaluation operator defined in the _evolve self method,
        and constraint handler if specified.
        '''
        
        # evaluation operator
        self.toolbox.register('evaluate', self._evaluate)

        # constraint handling decoration
        if hasattr(self, 'penalty'):
            self.toolbox.decorate('evaluate', addViolation(self.eqtol))
            if self.penalty == 'death':
                pnltfunc = partial(deathPenalty, self.toolbox.individual)
            elif self.penalty == 'delta':
                pnltfunc = partial(tools.DeltaPenalty, feasibility, float(self.delta), distance)                
            elif self.penalty == 'closest':
                closest_feasible = make_closest_feasible(
                    self.problem,
                    generator=self.toolbox.individual,
                    eqtol=self.eqtol
                )
                pnltfunc = partial(tools.ClosestValidPenalty, feasibility, closest_feasible, alpha=float(self.alpha_penalty), distance=manhattan_distance)
            else:
                raise ValueError(f"Unknown penalty type: {self.penalty}")           
            self.toolbox.decorate('evaluate', pnltfunc())
        
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
        offset = 2 if getattr(self, 'penalty', None) == 'cdp' else 0 
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values[offset:])
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
        t = gen / self.ngen
        # exponential:
        curr = start * ((end / start) ** t)
        setattr(obj, 'rate', max(0.0, curr))

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
        F = out["F"][0]
        G = out.get("G", [])[0] if self.problem.n_ieq_constr > 0 else []
        H = out.get("H", [])[0] if self.problem.n_eq_constr > 0 else []
        individual.constraints['G'] = np.array(G)
        individual.constraints['H'] = np.array(H)

        mask = out.get('mask', None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if isinstance(F, np.ndarray):
                return (tuple(F), list(mask))
            else:
                return ((F,), list(mask))
        else:
            if isinstance(F, np.ndarray):
                return tuple(F)
            else:
                return (F,)
    
    def _evalInvalid(self, individuals: Iterable[object], force_all: bool=False) -> None:
        '''
        Calls the evaluation operator on individuals with invalid (i.e., absent) fitness, also saving 
        the current number of evaluations done. Eventually, it can force the evaluation of the whole population.
        '''
        if force_all:
            invalid = list(individuals)
            for ind in invalid:
                del ind.fitness.values
        else:
            invalid = [ind for ind in individuals if not ind.fitness.valid]
        
        fits = self.toolbox.map(self.toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            if isinstance(fit, tuple) and len(fit) == 2 and isinstance(fit[1], (list, tuple, np.ndarray)):
                fit_vals, mask = fit
                ind.fitness.values = fit_vals
                ind.dropped_mask = np.asarray(mask, dtype=bool)
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
            if hasattr(self, 'drop_replacement_strg'):
                setattr(popdrop, 'strategy', self.popD_restoring_strg)

        if isindD:    
            self.problem = IndividualDropout(self.problem)
            setattr(self.problem, 'strg', getattr(self, 'indD_strg', 1))

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
                # if individual-dropout level is active, only nondropped variables will be exchanged/mutated
                varAnd = getattr(self.problem, 'varAnd')
            else:
                varAnd = getattr(algorithms, 'varAnd')
            offspring = varAnd(
                selected,
                self.toolbox,
                self.cxpb,
                self.mutpb
            )

            # ------------------------------------------------------
            # 5) Evaluate offspring with “full” fitness (no dropout)
            # ------------------------------------------------------
            if isindD:
                self._set_dropout(False)
                self._evalInvalid(offspring, force_all=True)
            else:
                self._evalInvalid(offspring)
            
            if hasattr(self, 'linear_ranking') and self.linear_ranking:
                self._linear_ranking(offspring)

            # -----------------------------------------------------------------
            # 6) Handle population‐level dropout restoration (if applicable)
            #    If 'drop_replacement_strg != 3', we restore certain parents
            #    into 'self.pop' in place of offspring via popdrop.restore(...)
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
            fis = [ind.fitness.values for ind in front]
        
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