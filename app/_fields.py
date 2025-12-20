opt_FIELDS = [
    {
        "label": "Problem name*",
        "key": "problem_name",
        "tooltip": (
            "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n"
            "'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'."
        ),
    },
    {
        "label": "Problem type (single or multi)*",
        "key": "problem_type",
        "tooltip": "Accepted: 'single', 'multi'.",
    },
    {
        "label": 'Experiment name (e.g. "baseline")*',
        "key": "version",
        "tooltip": (
            "Accepted: any version containing 'base', 'idrop',\n"
            "'indd', 'pdrop', 'popd' (case insensitive)."
        ),
    },
    {
        "label": "Number of runs*",
        "key": "n_runs",
        "tooltip": "",
    },
    {
        "label": "Number of generations*",
        "key": "ngen",
        "tooltip": "",
    },
    {
        "label": "Number of variables*",
        "key": "n_var",
        "tooltip": "",
    },
    {
        "label": "Individual dropout rate(s):",
        "key": "indD_rate",
        "tooltip": "Accepted: float or list of floats in [0, 1] (e.g., 0.3 for 30% dropout).",
    },
    {
        "label": "Population dropout rate(s):",
        "key": "popD_rate",
        "tooltip": "Accepted: float or list of floats in [0, 1] (e.g., 0.3 for 30% dropout).",
    },
]

opt_EXTRA_FIELDS = [
    {
        "label": "Population size",
        "key": "pop_size",
        "default": "200",
        "tooltip": None
    },
    {
        "label": "Hall of Fame (n. of top solutions to save)",
        "key": "hof",
        "default": "20",
        "tooltip": "Has to be <= population size."
    },
    {
        "label": "Crossover operator",
        "key": "mate",
        "default": "cxOnePoint",
        "tooltip": (
            "Chance that two individuals will be mated.\n"
            "Accepted: DEAP crossover operators (e.g., 'cxOnePoint',\n"
            "'cxTwoPoint', 'cxUniform', 'cxSimulatedBinaryBounded', etc.).\n"
            "For 'cxSimulatedBinaryBounded', add 'eta_cx' (float ~5-20; "
            "lower → offspring more diverse from parents)."
        )
    },
    {
        "label": "Crossover probability",
        "key": "cxpb",
        "default": "0.75",
        "tooltip": "Accepted: float in [0, 1]."
    },
    {
        "label": "Mutation operator",
        "key": "mutate",
        "default": "mutGaussian",
        "tooltip": (
            "Accepted: DEAP mutation operators (e.g., 'mutFlipBit',\n"
            "'mutPolynomialBounded', 'mutGaussian', 'mutShuffleIndexes').\n"
            "For 'mutPolynomialBounded', add 'eta_mut' (float ~5-20; "
            "lower → offspring more diverse from parents)."
        )
    },
    {
        "label": "Mutation probability",
        "key": "mutpb",
        "default": "0.1",
        "tooltip": "Chance that an individual is mutated.\nAccepted: float in [0, 1]."
    },
    {
        "label": "Individual probability",
        "key": "indpb",
        "default": "0.75",
        "tooltip": "Chance that each gene of a mutated individual is changed.\nAccepted: float in [0, 1]."
    },
    {
        "label": "Selection operator",
        "key": "select",
        "default": "selTournament",
        "tooltip": "Accepted: DEAP selection operators (e.g., 'selTournament',\n'selBest', 'selWorst', 'selRoulette', 'selRandom', 'selNSGA2')."
    },
    {
        "label": "Replacement operator",
        "key": "replacement_operator",
        "default": "mu_plus_lambda",
        "tooltip": "Accepted: 'elits', 'generational', 'mu_plus_lambda', 'mu_plus_lambda_nsga2', 'mu_comma_lambda',\n'steady_state', 'random_replacement', 'rand_tourn_repl', 'iter_tourn_repl'."
    },
    {
        "label": "Logbook aggregation operator (mean or median)",
        "key": "aggregation_op",
        "default": "median",
        "tooltip": "Accepted: 'mean' or 'median'."
    },
    {
        "label": "Seed",
        "key": "seed",
        "default": "42",
        "tooltip": "Accepted: any integer."
    }
]

plot_FIELDS = [
    {
        "label": "Problem name*",
        "key": "problem",
        "tooltip": (
            "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n"
            "'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'."
        )
    },
    {
        "label": 'Experiment name (e.g. "baseline")*',
        "key": "version",
        "tooltip": (
            "Accepted: any version containing 'base', 'idrop',\n"
            "'indd', 'pdrop', 'popd' (case insensitive)."
        )
    },
    {
        "label": "Logbook aggregation operator (mean or median)*",
        "key": "op",
        "default": "median",
        "tooltip": (
            "Accepted: 'mean' or 'median'."
        )
    }
]

comp_FIELDS = [
    {
        "label": "Problem name*",
        "key": "problem_name",
        "tooltip": "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'."
    },
    {
        "label": "Problem type (single or multi)*",
        "key": "problem_type",
        "tooltip": "Accepted: 'single', 'multi'."
    },
    {
        "label": '1st experiment name (e.g., "baseline")*',
        "key": "version1",
        "tooltip": "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive)."
    },
    {
        "label": '2nd experiment name (e.g., "IDrop")*',
        "key": "version2",
        "tooltip": "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive)."
    }
]

comp_EXTRA_FIELDS = [
    {
        "label": "Logbook aggregation operator (mean or median)",
        "key": "aggregation_op",
        "default": "median",
        "tooltip": "Accepted: 'mean' or 'median'."
    },
    {
        "label": "N. of objectives (for reference set)",
        "key": "n_obj",
        "default": "1",
        "tooltip": "Specify the number of objectives (e.g., 1 for single-objective)."
    },
    {
        "label": "N. of divisions (for reference set)",
        "key": "P",
        "default": "0",
        "tooltip": "Choose the integer P such that:\n C(P + n.obj - 1, n.obj - 1) ≈ population size.\n\n \
    - For 2 objectives: P≈99→100, 199→200, 499→500, 999→1000\n(Population size ≈ P+1)\n\n\
    - For 3 objectives: P≈13→105, 19→210, 23→300, 30→496, 43→990\n(Population size ≈ (P+2)*(P+1)/2)"
    },
    {
        "label": "Save full comparison results (True or False)",
        "key": "save_full",
        "default": "False",
        "tooltip": "Accepted: 'True' or 'False'."
    }
]