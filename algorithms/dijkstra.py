"""
Классический алгоритм Дейкстры.
Использует текущие веса рёбер графа (не меняет их).
"""
import heapq
from typing import List, Tuple
from graph.graph import Graph
from .base import Algorithm
from .utils import reconstruct_path

class Dijkstra(Algorithm):
    """Дейкстра для фиксированного набора весов."""

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
                w = edge.current_weight
                if w is None:
                    # TODO: вес не сгенерирован – пропускаем (или можно сгенерировать, но лучше требовать инициализации)
                    continue
                nd = d + w
                if nd < dist[edge.v]:
                    dist[edge.v] = nd
                    prev[edge.v] = u
                    heapq.heappush(pq, (nd, edge.v))

        if dist[target] == float('inf'):
            return float('inf'), []
        return dist[target], reconstruct_path(prev, source, target)