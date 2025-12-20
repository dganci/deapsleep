from deap import base, creator, tools
import warnings as warn

class Initializer:
    '''
    A class for initializing a population of individuals, based on predefined parameters
    '''
    def __init__(self, *args, **kwargs):

        self.args = args
        self.__dict__.update(kwargs)
        self.toolbox = base.Toolbox()
        self._defineInd()
        self._generateInd()
        self._generatePop()

    def _defineInd(self) -> None:
        '''
        Shapes an individual, based on given parameters.
        '''
        
        if getattr(self, 'is_permutation', False):
            if self.ind_type is not list:
                warn(
                    "\nIndividual type changed to 'list' \
                       because permutation encoding is used.\n"
                    )
                self.ind_type = list
        
        # delete already existing classes
        for cls in ['Fitness', 'Individual']:
            if hasattr(creator, cls):
                delattr(creator, cls)

        # fitness class definition
        creator.create(
            'Fitness',
            base.Fitness,
            weights=self.weights
            )
        self.fitness = getattr(creator, 'Fitness')

        # container creation
        creator.create(
            'Individual',
            self.ind_type,
            fitness=self.fitness,
            constraints=dict,
            violation=float,
            dropped=bool
            )
        self.container = getattr(creator, 'Individual')

    def _generateInd(self) -> None:
        '''
        Generates an individual based on previous definition.
        '''
        # permutation individual generation
        if getattr(self, "is_permutation", False):
            if not hasattr(self, 'n_var') or self.n_var is None:
                raise ValueError(
                    '\nFor permutation problems, the number of variables \
                       must be specified in the initializer parameters.\n'
                    )
            import random
            self.toolbox.register(
                "individual",
                tools.initIterate,
                self.container,
                lambda: random.sample(range(self.n_var), self.n_var)
            )
        # default individual generation
        else:
            # attribute generator
            attr_gen = []
            for i, (name, func, *bounds) in enumerate(self.args):
                gen_name = f"{name}_{i}"
                self.toolbox.register(gen_name, func, *bounds)
                attr_gen.append(getattr(self.toolbox, gen_name))
                
            # structure initializers
            self.toolbox.register(
                'individual',
                tools.initCycle,
                self.container,
                attr_gen,
                1 # number of cycles
                )

    def _generatePop(self) -> None:
        '''
        Generates a population by calling _generateInd a *pop_size* number of times.
        '''    
        self.toolbox.register(
            'population',
            tools.initRepeat,
            self.pop_type,
            self.toolbox.individual,
            self.pop_size
            )