"""
Запуск экспериментов и сбор результатов.
"""
from typing import Dict
from datetime import datetime

import os
import numpy as np
import pandas as pd

from algorithms.adaptive_dijkstra import AdaptiveDijkstra
from algorithms.dijkstra import Dijkstra
from algorithms.expected_dijkstra import ExpectedDijkstra
from graph.factory import GraphFactory


def run_experiments(config: Dict, output_dir: str = "results") -> str:
    """
    Запускает эксперименты и сохраняет результаты в CSV.

    Каждый алгоритм находит путь и возвращает реальное время прохождения.
    Для устойчивости результатов используем много независимых прогонов (num_runs).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"experiment_{timestamp}.csv")
    results = []

    for size in config['graph_sizes']:
        for topo in config['topologies']:
            for dist_name, dist_params in config['distributions'].items():
                for run in range(config['num_runs']):
                    seed = config['seed'] + run * 100
                    if topo == 'random':
                        graph = GraphFactory.create_random(
                            n=size,
                            edge_prob=config['random_edge_prob'],
                            mu_func=dist_params['mu'],
                            p_func=dist_params['p'],
                            seed=seed
                        )
                    else:  # grid
                        graph = GraphFactory.create_grid(
                            rows=config['grid_rows'],
                            cols=config['grid_cols'],
                            mu_func=dist_params['mu'],
                            p_func=dist_params['p'],
                            directed=False,
                            seed=seed
                        )

                    start_time = dist_params.get('start_time', 0.0)
                    graph.current_time = start_time
                    graph.update_all_weights(0)

                    source, target = np.random.choice(graph.n, size=2, replace=False)

                    dijkstra_path = None

                    for alg_name in config['algorithms']:
                        graph_copy = graph.copy()

                        if alg_name == 'dijkstra':
                            alg = Dijkstra()
                        elif alg_name == 'adaptive_dijkstra':
                            alg = AdaptiveDijkstra()
                        elif alg_name == 'expected_dijkstra':
                            alg = ExpectedDijkstra()
                        else:
                            continue

                        time_taken, path = alg.find_path(graph_copy, source, target)

                        if alg_name == 'dijkstra':
                            dijkstra_path = set(path) if path else set()

                        path_set = set(path) if path else set()
                        path_differs = path_set != dijkstra_path if dijkstra_path is not None else False

                        results.append({
                            'graph_size': size,
                            'topology': topo,
                            'distribution': dist_name,
                            'run': run,
                            'source': int(source),
                            'target': int(target),
                            'algorithm': alg_name,
                            'time': time_taken,
                            'path_length': len(path) if path else 0,
                            'path': '-'.join(map(str, path)) if path else '',
                            'path_differs_from_dijkstra': path_differs
                        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Результаты сохранены в {output_path}")
    return output_path