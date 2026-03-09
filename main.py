from experiments.config import EXPERIMENT_CONFIG
from experiments.runner import run_experiments

if __name__ == '__main__':
    results = run_experiments(EXPERIMENT_CONFIG)
