"""
Абстрактный базовый класс для алгоритмов поиска пути.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List
from graph.graph import Graph

class Algorithm(ABC):
    """Интерфейс алгоритма поиска кратчайшего пути."""

    @abstractmethod
    def find_path(self, graph: Graph, source: int, target: int) -> Tuple[float, List[int]]:
        """
        Возвращает кортеж (total_time, path), где path — список вершин от source до target.
        Если путь не найден, возвращает (inf, []).
        """
        pass