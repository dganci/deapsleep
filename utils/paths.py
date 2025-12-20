import os

ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), 
            ".."
            )
        ) # points to DeapSleep

CONFIG_DIR = os.path.join(
    ROOT, 
    'deapsleep', 
    'experiments', 
    'configurations'
)

RESULTS_DIR = os.path.join(
    ROOT, 
    'results'
)