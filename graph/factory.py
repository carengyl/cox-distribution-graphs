"""
Фабрика для создания графов различных топологий.
"""
import numpy as np
from typing import Callable, List, Tuple, Optional
from .graph import Graph
from .coxian import CoxianDistribution

class GraphFactory:
    """Статические методы для генерации графов."""

    @staticmethod
    def create_random(n: int, edge_prob: float,
                      mu_func: Callable, p_func: Callable,
                      seed: Optional[int] = None) -> Graph:
        """
        Создаёт случайный граф Эрдёша–Реньи G(n, edge_prob).
        Каждое возможное ребро (i,j) (i != j) добавляется с вероятностью edge_prob.
        Все рёбра имеют одинаковое распределение, заданное mu_func, p_func.
        """
        if seed is not None:
            np.random.seed(seed)
        graph = Graph(n)
        dist = CoxianDistribution(mu_func, p_func)
        for i in range(n):
            for j in range(n):
                if i != j and np.random.rand() < edge_prob:
                    graph.add_edge(i, j, dist)
        return graph

    @staticmethod
    def create_grid(rows: int, cols: int,
                    mu_func: Callable, p_func: Callable,
                    directed: bool = False,
                    seed: Optional[int] = None) -> Graph:
        """
        Создаёт граф-решётку rows x cols.
        Вершины нумеруются построчно: (r,c) -> index = r*cols + c.
        Связи: вправо и вниз (если directed=False, то добавляются обратные рёбра автоматически).
        """
        n = rows * cols
        graph = Graph(n)
        dist = CoxianDistribution(mu_func, p_func)

        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                # связь вправо
                if c + 1 < cols:
                    v = r * cols + (c + 1)
                    graph.add_edge(u, v, dist)
                    if not directed:
                        graph.add_edge(v, u, dist)
                # связь вниз
                if r + 1 < rows:
                    v = (r + 1) * cols + c
                    graph.add_edge(u, v, dist)
                    if not directed:
                        graph.add_edge(v, u, dist)
        return graph

    @staticmethod
    def create_from_adjacency(n: int, adj_list: List[List[Tuple[int, CoxianDistribution]]]) -> Graph:
        """
        Создаёт граф по заданному списку смежности.
        adj_list[i] = список кортежей (j, distribution) для рёбер из i в j.
        """
        graph = Graph(n)
        for u in range(n):
            for v, dist in adj_list[u]:
                graph.add_edge(u, v, dist)
        return graph