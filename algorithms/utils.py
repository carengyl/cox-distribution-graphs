"""
Вспомогательные функции для алгоритмов.
"""
from typing import List, Tuple
from graph.graph import Graph


def reconstruct_path(prev: List[int], source: int, target: int) -> List[int]:
    """Восстанавливает путь по массиву предков."""
    path = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path[0] == source:
        return path
    else:
        return []   # путь не найден


def compute_path_time(graph: Graph, path: List[int]) -> float:
    """
    Вычисляет суммарный вес пути по текущим (уже сгенерированным) весам рёбер.
    Не изменяет граф и не генерирует новые веса.
    """
    if not path or len(path) < 2:
        return float('inf')
    total = 0.0
    for i in range(len(path) - 1):
        w = graph.get_weight(path[i], path[i + 1])
        if w is None:
            return float('inf')
        total += w
    return total


def simulate_traversal(graph: Graph, path: List[int]) -> float:
    """
    Симулирует реальный проход по заданному пути: на каждом шаге
    продвигает время графа и пересчитывает веса рёбер.
    Возвращает суммарное реальное время прохождения.
    """
    total_time = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        w = None
        for idx in graph.adj[u]:
            edge = graph.edges[idx]
            if edge.v == v:
                w = edge.update_weight(graph.current_time)
                break
        if w is None:
            return float('inf')
        total_time += w
        graph.advance_time(w)
        for edge in graph.edges:
            edge.update_weight(graph.current_time)
    return total_time