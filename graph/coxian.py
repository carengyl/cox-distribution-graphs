"""
Распределение Кокса (Coxian distribution).
Поддерживает как фиксированные параметры, так и зависящие от времени.
"""
import numpy as np


class CoxianDistribution:
    """
    Распределение Кокса порядка k.
    Параметры могут быть заданы:
    - как фиксированные списки (mu, p)
    - как функции времени (mu(t), p(t))
    """

    def __init__(self, mu, p):
        """
        mu: список интенсивностей фаз длины k, или функция mu(t) -> List[float]
        p:  список вероятностей продолжения длины k-1, или функция p(t) -> List[float]
        """
        self._mu = mu
        self._p = p
        self._is_dynamic = callable(mu) or callable(p)
        # Проверим, что если один из параметров функция, то и другой тоже функция
        if callable(mu) != callable(p):
            raise ValueError("Both mu and p must be either callable or non-callable")

    def _get_params(self, t: float = 0.0):
        """Возвращает пару (mu, p) для момента времени t."""
        if self._is_dynamic:
            mu = self._mu(t)
            p = self._p(t)
        else:
            mu = self._mu
            p = self._p
        # Проверим размерности
        if len(p) != len(mu) - 1:
            raise ValueError(f"Length of p ({len(p)}) must be len(mu)-1 ({len(mu)-1})")
        return mu, p

    def sample(self, t: float = 0.0) -> float:
        """
        Генерирует одно значение из распределения в момент времени t.
        """
        mu, p = self._get_params(t)
        total = 0.0
        phase = 0
        k = len(mu)
        while phase < k:
            # длительность текущей фазы (экспоненциальное распределение)
            total += np.random.exponential(1.0 / mu[phase])
            if phase == k - 1:
                break
            # решение о переходе к следующей фазе
            if np.random.rand() < p[phase]:
                phase += 1
            else:
                break
        return total

    def mean(self, t: float = 0.0) -> float:
        """
        Математическое ожидание распределения в момент времени t.
        """
        mu, p = self._get_params(t)
        exp = 0.0
        prod = 1.0
        for i in range(len(mu)):
            exp += prod / mu[i]
            if i < len(mu) - 1:
                prod *= p[i]
            else:
                prod = 0.0
        return exp

    @property
    def mu(self):
        return self._mu

    @property
    def p(self):
        return self._p