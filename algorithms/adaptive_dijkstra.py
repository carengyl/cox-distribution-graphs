"""
Адаптивный алгоритм: после прохождения каждого ребра обновляет время графа и веса,
затем перезапускает Дейкстру из новой вершины.
"""
from typing import List, Tuple
from graph.graph import Graph
from .base import Algorithm
from .dijkstra import Dijkstra

class AdaptiveDijkstra(Algorithm):
    """
    Адаптивный алгоритм на основе Дейкстры.
    ВНИМАНИЕ: изменяет состояние графа (время и веса) в процессе работы.
    """

    def find_path(self, graph: Graph, source: int, target: int) -> Tuple[float, List[int]]:
        total_time = 0.0
        path = [source]
        current = source
        dijkstra = Dijkstra()   # используем обычную Дейкстру для промежуточных расчётов

        while current != target:
            # Находим путь от текущей вершины до цели в текущем состоянии графа
            seg_time, seg_path = dijkstra.find_path(graph, current, target)
            if seg_time == float('inf'):
                return float('inf'), []   # нет пути

            # Первый шаг найденного пути
            next_node = seg_path[1]
            # Получаем вес ребра (должен быть уже сгенерирован, т.к. Дейкстра его использовала)
            w = graph.get_weight(current, next_node)
            if w is None:
                # на всякий случай сгенерируем
                for idx in graph.adj[current]:
                    edge = graph.edges[idx]
                    if edge.v == next_node:
                        w = edge.update_weight(graph.current_time)
                        break

            total_time += w
            path.append(next_node)
            current = next_node

            # Обновляем время графа на пройденное время
            graph.advance_time(w)

            # Обновляем веса всех рёбер в новый момент времени
            # (без дополнительного увеличения времени)
            for edge in graph.edges:
                edge.update_weight(graph.current_time)

        return total_time, path