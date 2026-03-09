"""
Конфигурация экспериментов.
"""

# Пример функции временной зависимости для утреннего часа пик
def mu_morning(t: float):
    hour = t % 24
    if 7 <= hour <= 9:
        return [0.3, 0.7]   # две фазы, низкие интенсивности -> большие задержки
    elif 17 <= hour <= 19:
        return [0.4, 0.8]
    else:
        return [1.0, 2.0]

def p_morning(t: float):
    # вероятность перехода постоянна
    return [0.6]

# Параметры экспериментов
EXPERIMENT_CONFIG = {
    'graph_sizes': [10, 20, 50],
    'topologies': ['random', 'grid'],
    'random_edge_prob': 0.2,
    'grid_rows': 5,
    'grid_cols': 5,
    'distributions': {
        'static_exp': {   # экспоненциальное (1 фаза)
            'mu': lambda t: [1.0],
            'p': lambda t: []
        },
        'static_cox2': {  # Кокса 2 фазы
            'mu': lambda t: [1.0, 2.0],
            'p': lambda t: [0.7]
        },
        'morning_peak': { # зависит от времени
            'mu': mu_morning,
            'p': p_morning
        }
    },
    'algorithms': ['dijkstra', 'adaptive_dijkstra', 'expected_dijkstra'],
    'num_runs': 10,        # число прогонов для статистики
    'seed': 42
}