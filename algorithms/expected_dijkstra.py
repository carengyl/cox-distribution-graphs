"""
Модификация Дейкстры, использующая математическое ожидание весов вместо текущих значений.
"""
import heapq
from typing import List, Tuple
from graph.graph import Graph
from .base import Algorithm
from .utils import reconstruct_path

class ExpectedDijkstra(Algorithm):
    """Дейкстра на математических ожиданиях весов (не меняет граф)."""

    def find_path(self, graph: Graph, source: int, target: int) -> Tuple[float, List[int]]:
        n = graph.n
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[source] = 0
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for idx in graph.adj[u]:
                edge = graph.edges[idx]
                # используем математическое ожидание в текущий момент времени
                w_mean = edge.distribution.mean(graph.current_time)
                nd = d + w_mean
                if nd < dist[edge.v]:
                    dist[edge.v] = nd
                    prev[edge.v] = u
                    heapq.heappush(pq, (nd, edge.v))

        if dist[target] == float('inf'):
            return float('inf'), []
        return dist[target], reconstruct_path(prev, source, target)