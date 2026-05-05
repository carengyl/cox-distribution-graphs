"""
Модификация Дейкстры, использующая математическое ожидание весов для планирования,
затем симулирует реальный проход по найденному пути.
"""
import heapq
from typing import List, Tuple
from graph.graph import Graph
from .base import Algorithm
from .utils import reconstruct_path, simulate_traversal

class ExpectedDijkstra(Algorithm):
    """Дейкстра на математических ожиданиях весов."""

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
                w_mean = edge.distribution.mean(graph.current_time)
                nd = d + w_mean
                if nd < dist[edge.v]:
                    dist[edge.v] = nd
                    prev[edge.v] = u
                    heapq.heappush(pq, (nd, edge.v))

        if dist[target] == float('inf'):
            return float('inf'), []

        path = reconstruct_path(prev, source, target)
        real_time = simulate_traversal(graph, path)
        return real_time, path