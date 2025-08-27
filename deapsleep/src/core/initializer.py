from deap import base, creator, tools

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
        Generates an individual by calling _defineInd.
        '''
        # attribute generator
        attr_gen = []
        for i, (base_name, func, *bounds) in enumerate(self.args):
            gen_name = f"{base_name}_{i}"
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