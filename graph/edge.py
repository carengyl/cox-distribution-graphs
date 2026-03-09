"""
Класс ребра графа.
"""
from typing import Optional
from .coxian import CoxianDistribution

class Edge:
    """Ребро графа, связывающее вершины u и v."""

    def __init__(self, u: int, v: int, distribution: CoxianDistribution):
        self.u = u
        self.v = v
        self.distribution = distribution
        self.current_weight: Optional[float] = None   # текущий вес (обновляется вызовом update_weight)

    def update_weight(self, t: float) -> float:
        """
        Генерирует новый вес ребра для момента времени t и сохраняет его.
        Возвращает сгенерированный вес.
        """
        self.current_weight = self.distribution.sample(t)
        return self.current_weight

    def get_weight(self) -> Optional[float]:
        """Возвращает текущий вес, если он был сгенерирован, иначе None."""
        return self.current_weight