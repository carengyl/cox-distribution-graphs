"""
Вспомогательные функции для алгоритмов.
"""
from typing import List

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