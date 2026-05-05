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
        Каждое ребро получает собственное распределение с параметрами,
        сгенерированными через mu_func(u, v) и p_func(u, v).
        Если mu_func/p_func принимают 2 аргумента — вызываются как mu_func(u, v),
        иначе как mu_func(t) (старый интерфейс).
        """
        if seed is not None:
            np.random.seed(seed)
        graph = Graph(n)
        for i in range(n):
            for j in range(n):
                if i != j and np.random.rand() < edge_prob:
                    mu, p = GraphFactory._resolve_params(mu_func, p_func, i, j)
                    dist = CoxianDistribution(mu, p)
                    graph.add_edge(i, j, dist)
        return graph

    @staticmethod
    def create_grid(rows: int, cols: int,
                    mu_func: Callable, p_func: Callable,
                    directed: bool = False,
                    seed: Optional[int] = None) -> Graph:
        """
        Создаёт граф-решётку rows x cols.
        Каждое ребро получает собственное распределение.
        """
        n = rows * cols
        if seed is not None:
            np.random.seed(seed)
        graph = Graph(n)

        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 1 < cols:
                    v = r * cols + (c + 1)
                    mu, p = GraphFactory._resolve_params(mu_func, p_func, u, v)
                    dist = CoxianDistribution(mu, p)
                    graph.add_edge(u, v, dist)
                    if not directed:
                        mu_r, p_r = GraphFactory._resolve_params(mu_func, p_func, v, u)
                        dist_r = CoxianDistribution(mu_r, p_r)
                        graph.add_edge(v, u, dist_r)
                if r + 1 < rows:
                    v = (r + 1) * cols + c
                    mu, p = GraphFactory._resolve_params(mu_func, p_func, u, v)
                    dist = CoxianDistribution(mu, p)
                    graph.add_edge(u, v, dist)
                    if not directed:
                        mu_r, p_r = GraphFactory._resolve_params(mu_func, p_func, v, u)
                        dist_r = CoxianDistribution(mu_r, p_r)
                        graph.add_edge(v, u, dist_r)
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

    @staticmethod
    def _resolve_params(mu_func, p_func, u, v):
        """Определяет, принимает ли функция 2 аргумента (u, v) или 1 (t)."""
        import inspect
        sig = inspect.signature(mu_func)
        n_params = len(sig.parameters)
        if n_params >= 2:
            return mu_func(u, v), p_func(u, v)
        else:
            return mu_func, p_func