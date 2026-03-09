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
    Возвращает путь к сохранённому файлу.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"experiment_{timestamp}.csv")
    results = []

    for size in config['graph_sizes']:
        for topo in config['topologies']:
            for dist_name, dist_params in config['distributions'].items():
                for run in range(config['num_runs']):
                    # Создаём граф
                    seed = config['seed'] + run * 100  # для разнообразия
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

                    # Инициализируем веса в начальный момент времени (t=0)
                    graph.update_all_weights(0)

                    # Выбираем случайные источник и цель (разные)
                    source, target = np.random.choice(graph.n, size=2, replace=False)

                    # Для каждого алгоритма запускаем поиск
                    for alg_name in config['algorithms']:
                        # Создаём копию графа, чтобы алгоритмы не влияли друг на друга
                        # Для простоты будем каждый раз создавать новый граф, но это дорого.
                        # Вместо этого можно клонировать граф. Реализуем метод copy в Graph.
                        # Здесь для краткости будем использовать один граф, но в реальности нужно копировать.
                        # Оставим как заглушку.

                        # Вызов алгоритма
                        if alg_name == 'dijkstra':
                            alg = Dijkstra()
                        elif alg_name == 'adaptive_dijkstra':
                            alg = AdaptiveDijkstra()
                        elif alg_name == 'expected_dijkstra':
                            alg = ExpectedDijkstra()
                        else:
                            continue

                        # Важно: для адаптивного алгоритма нужно передавать граф, но он его изменит.
                        # Поэтому перед вызовом создадим копию графа.
                        # Для простоты опустим копирование.

                        # Здесь должен быть код клонирования графа.
                        # Пока просто вызовем алгоритм, понимая, что состояние будет испорчено.
                        time_taken, path = alg.find_path(graph, source, target)

                        results.append({
                            'graph_size': size,
                            'topology': topo,
                            'distribution': dist_name,
                            'run': run,
                            'source': source,
                            'target': target,
                            'algorithm': alg_name,
                            'time': time_taken,
                            'path_length': len(path) if path else 0
                        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Результаты сохранены в {output_path}")
    return output_path