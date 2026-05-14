# algorithms/astar.py

import heapq
from typing import List, Tuple, Callable, Optional

from graph.graph import Graph
from .base import Algorithm
from .utils import reconstruct_path, simulate_traversal


class AStar(Algorithm):
    """
    A* алгоритм, планирующий путь по математическим ожиданиям весов рёбер
    с использованием эвристической оценки оставшегося пути.
    После построения пути симулирует реальный проход с генерацией весов
    согласно распределению Кокса.
    """

    def __init__(self, heuristic: Optional[Callable[[Graph, int, int], float]] = None):
        """
        :param heuristic: функция вида heuristic(graph, node, target) -> float,
                          оценивающая ожидаемое оставшееся время от node до target.
                          Если не задана, используется нулевая эвристика
                          (алгоритм вырождается в ExpectedDijkstra).
        """
        if heuristic is None:
            self.heuristic = lambda graph, u, v: 0.0
        else:
            self.heuristic = heuristic

    def find_path(self, graph: Graph, source: int, target: int) -> Tuple[float, List[int]]:
        n = graph.n

        # g_score — фактическая ожидаемая стоимость от source до вершины
        g_score = [float('inf')] * n
        g_score[source] = 0.0

        # f_score = g_score + heuristic
        f_score = [float('inf')] * n
        f_score[source] = self.heuristic(graph, source, target)

        prev = [-1] * n

        # Очередь с приоритетом (f_score, vertex)
        pq = [(f_score[source], source)]

        while pq:
            current_f, u = heapq.heappop(pq)

            # Если в очереди устаревшая запись — пропускаем
            if current_f > f_score[u]:
                continue

            if u == target:
                break

            # Обходим все рёбра из u
            for idx in graph.adj[u]:
                edge = graph.edges[idx]
                # Математическое ожидание веса в текущий момент времени
                w_mean = edge.distribution.mean(graph.current_time)

                v = edge.v
                tentative_g = g_score[u] + w_mean
                if tentative_g < g_score[v]:
                    prev[v] = u
                    g_score[v] = tentative_g
                    f = tentative_g + self.heuristic(graph, v, target)
                    f_score[v] = f
                    heapq.heappush(pq, (f, v))

        if g_score[target] == float('inf'):
            return float('inf'), []

        path = reconstruct_path(prev, source, target)

        # Реальное время прохода (с генерацией весов при движении)
        real_time = simulate_traversal(graph, path)

        return real_time, path