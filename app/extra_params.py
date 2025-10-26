opt_params = {
    "Population size": ("200", "pop_size"),
    "Hall of Fame (n. of top solutions to save)": ("20", "hof"),
    "Crossover operator": ("cxOnePoint", "mate"),
    "Crossover probability": ("0.75", "cxpb"),
    "Mutation operator": ("mutGaussian", "mutate"),
    "Mutation probability": ("0.1", "mutpb"),
    "Individual probability": ("0.75", "indpb"),
    "Selection operator": ("selTournament", "select"),
    "Replacement operator": ("mu_plus_lambda", "replacement_operator"),
    "Logbook aggregation operator (mean or median)": ("median", "aggregation_op"),
    "Seed": ("42", "seed"),
}

opt_tooltips = {
    "Hall of Fame (n. of top solutions to save)": "Has to be <= population size.",
    "Crossover operator": "Chance that two individuals will be mated.\nAccepted: DEAP crossover operators (e.g., 'cxOnePoint',\n'cxTwoPoint', 'cxUniform', 'cxSimulatedBinaryBounded', etc.).\nFor 'cxSimulatedBinaryBounded', add 'eta_cx' (float ~5-20; lower → offspring more diverse from parents).",
    "Crossover probability": "Accepted: float in [0, 1].",
    "Mutation operator": "Accepted: DEAP mutation operators (e.g., 'mutFlipBit',\n'mutPolynomialBounded', 'mutGaussian', 'mutShuffleIndexes').\nFor 'mutPolynomialBounded', add 'eta_mut' (float ~5-20; lower → offspring more diverse from parents).'",
    "Mutation probability": "Chance that an individual is mutated.\nAccepted: float in [0, 1].",
    "Individual probability": "Chance that each gene of a mutated individual is changed.\nAccepted: float in [0, 1].",
    "Replacement operator": "Accepted: 'elits', 'generational', 'mu_plus_lambda', 'mu_plus_lambda_nsga2', 'mu_comma_lambda',\n'steady_state', 'random_replacement', 'rand_tourn_repl', 'iter_tourn_repl'.",
    "Selection operator": "Accepted: DEAP selection operators (e.g., 'selTournament',\n'selBest', 'selWorst', 'selRoulette', 'selRandom', 'selNSGA2').",
    "Logbook aggregation operator (mean or median)": "Accepted: 'mean' or 'median'.",
    "Seed": "Accepted: any integer."
}

plot_params = {
    "Logbook aggregation operator (mean or median)": ("median", "op")
}

plot_tooltips = {
    "Logbook aggregation operator (mean or median)": "Accepted: 'mean' or 'median'."
}

comp_params = {
    "Logbook aggregation operator (mean or median)": ("median", "aggregation_op"),
    "N. of objectives (for reference set)": ("1", "n_obj"),
    "N. of divisions (for reference set)": ("0", "P"),
    "Save full comparison results (True or False)": ("False", "save_full")
}

comp_tooltips = {
    "Logbook aggregation operator (mean or median)": "Accepted: 'mean' or 'median'.",
    "N. of divisions (for reference set)": "Choose the integer P such that:\n C(P + n.obj - 1, n.obj - 1) ≈ population size.\n\n \
    - For 2 objectives: P≈99→100, 199→200, 499→500, 999→1000\n(Population size ≈ P+1)\n\n\
    - For 3 objectives: P≈13→105, 19→210, 23→300, 30→496, 43→990\n(Population size ≈ (P+2)*(P+1)/2)",
    "Save full comparison results (True or False)": "Accepted: 'True' or 'False'."
}