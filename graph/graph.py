"""
Класс графа с динамическими весами рёбер.
"""
from typing import List, Tuple, Optional
from .edge import Edge
from .coxian import CoxianDistribution

class Graph:
    """Граф с поддержкой текущего времени и обновления весов рёбер."""

    def __init__(self, n_vertices: int, initial_time: float = 0.0):
        """
        n_vertices: количество вершин (вершины нумеруются от 0 до n-1)
        initial_time: начальное модельное время
        """
        self.n = n_vertices
        self.edges: List[Edge] = []
        self.adj: List[List[int]] = [[] for _ in range(n_vertices)]  # для каждой вершины список индексов рёбер
        self.current_time = initial_time

    def add_edge(self, u: int, v: int, distribution: CoxianDistribution):
        """
        Добавляет ориентированное ребро u -> v с заданным распределением.
        Для неориентированного графа нужно добавить два ориентированных.
        """
        idx = len(self.edges)
        edge = Edge(u, v, distribution)
        self.edges.append(edge)
        self.adj[u].append(idx)

    def update_all_weights(self, time_increment: float = 0.0):
        """
        Обновляет веса всех рёбер, используя текущее время,
        затем увеличивает текущее время на time_increment.
        """
        for edge in self.edges:
            edge.update_weight(self.current_time)
        self.current_time += time_increment

    def get_weight(self, u: int, v: int) -> Optional[float]:
        """
        Возвращает текущий вес ребра (u,v), если оно существует.
        Если рёбер несколько (мультиграф) – вернёт первое попавшееся.
        """
        for idx in self.adj[u]:
            edge = self.edges[idx]
            if edge.v == v:
                return edge.current_weight
        return None

    def advance_time(self, delta: float):
        """Увеличивает текущее время на delta (без изменения весов)."""
        self.current_time += delta

    def get_current_edge_list(self) -> List[Tuple[int, int, float]]:
        """
        Возвращает список всех рёбер с актуальными весами.
        Если вес не сгенерирован, он будет сгенерирован с использованием текущего времени.
        """
        result = []
        for edge in self.edges:
            if edge.current_weight is None:
                edge.update_weight(self.current_time)
            result.append((edge.u, edge.v, edge.current_weight))
        return result

    def copy(self) -> 'Graph':
        """Создаёт глубокую копию графа с текущими весами и временем."""
        new_graph = Graph(self.n, self.current_time)
        # Копируем рёбра
        for edge in self.edges:
            # Создаём новое распределение с теми же функциями
            dist = CoxianDistribution(edge.distribution.mu, edge.distribution.p)
            new_edge = Edge(edge.u, edge.v, dist)
            new_edge.current_weight = edge.current_weight
            new_graph.edges.append(new_edge)
            new_graph.adj[edge.u].append(len(new_graph.edges)-1)
        return new_graph