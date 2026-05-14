"""
Конфигурация экспериментов.
"""
import numpy as np

# --- Однородные распределения (интерфейс mu(t), p(t)) ---

def mu_static_exp(t: float):
    return [1.0]

def p_static_exp(t: float):
    return []


# --- Неоднородные распределения (интерфейс mu(u, v), p(u, v)) ---
# Каждое ребро получает свои параметры — быстрый/медленный маршрут.

_rng = np.random.RandomState(99)
_edge_params_cache = {}

def _get_edge_mu(u, v):
    key = (u, v)
    if key not in _edge_params_cache:
        _edge_params_cache[key] = _rng.uniform(0.3, 3.0)
    return _edge_params_cache[key]


def mu_heterogeneous(u: float, v: float):
    mu_val = _get_edge_mu(int(u), int(v))
    return [mu_val]

def p_heterogeneous(u: float, v: float):
    return []


def mu_heterogeneous_cox2(u: float, v: float):
    mu_val = _get_edge_mu(int(u), int(v))
    return [mu_val, mu_val * 1.5]

def p_heterogeneous_cox2(u: float, v: float):
    return [0.6]


# --- Зависящие от времени распределения ---
# Два типа рёбер: «обычные» и «загруженные в час пик».
# Загруженные рёбра в пик замедляются в 5-10 раз.

def _is_rush_edge(u, v):
    return (int(u) * 31 + int(v) * 17) % 7 == 0

def _make_rush_mu(is_rush):
    if is_rush:
        return lambda t: [0.1] if 7 <= (t % 24) <= 10 else ([0.2] if 17 <= (t % 24) <= 20 else [1.5])
    else:
        return lambda t: [1.0]

def _make_rush_p(is_rush):
    return lambda t: []

def mu_rush_hour(u: float, v: float):
    is_rush = _is_rush_edge(u, v)
    return _make_rush_mu(is_rush)

def p_rush_hour(u: float, v: float):
    is_rush = _is_rush_edge(u, v)
    return _make_rush_p(is_rush)


# Аналогично, но с 2 фазами и более сильным эффектом
def _make_rush_mu2(is_rush):
    if is_rush:
        return lambda t: [0.08, 0.15] if 7 <= (t % 24) <= 10 else ([0.15, 0.3] if 17 <= (t % 24) <= 20 else [1.0, 2.0])
    else:
        return lambda t: [1.0, 2.0]

def _make_rush_p2(is_rush):
    if is_rush:
        return lambda t: [0.5] if 7 <= (t % 24) <= 10 else [0.6]
    else:
        return lambda t: [0.6]

def mu_rush_hour_cox2(u: float, v: float):
    is_rush = _is_rush_edge(u, v)
    return _make_rush_mu2(is_rush)

def p_rush_hour_cox2(u: float, v: float):
    is_rush = _is_rush_edge(u, v)
    return _make_rush_p2(is_rush)


# Параметры экспериментов
EXPERIMENT_CONFIG = {
    'graph_sizes': [10, 20, 50],
    'topologies': ['random', 'grid'],
    'random_edge_prob': 0.3,
    'grid_rows': 5,
    'grid_cols': 5,
    'distributions': {
        'static_exp': {
            'mu': mu_static_exp,
            'p': p_static_exp
        },
        'heterogeneous': {
            'mu': mu_heterogeneous,
            'p': p_heterogeneous
        },
        'heterogeneous_cox2': {
            'mu': mu_heterogeneous_cox2,
            'p': p_heterogeneous_cox2
        },
        'rush_hour': {
            'mu': mu_rush_hour,
            'p': p_rush_hour,
            'start_time': 8.0    # начинаем в час пик
        },
        'rush_hour_cox2': {
            'mu': mu_rush_hour_cox2,
            'p': p_rush_hour_cox2,
            'start_time': 8.0
        }
    },
    'algorithms': ['dijkstra', 'adaptive_dijkstra', 'expected_dijkstra', 'astar'],
    'num_runs': 10,
    'seed': 42
}

def get_heuristic(topology, graph):
    if topology == 'grid':
        cols = EXPERIMENT_CONFIG['grid_cols']
        def heuristic(graph, u, target):
            ur, uc = divmod(u, cols)
            tr, tc = divmod(target, cols)
            return abs(ur - tr) + abs(uc - tc)
        return heuristic
    else:
        # Для случайного графа используем эвристику на основе степени вершин
        def heuristic(graph, u, target):
            if u == target:
                return 0.0
            deg_u = len(graph.adj[u])
            deg_t = len(graph.adj[target])
            max_deg = max(len(adj) for adj in graph.adj) if graph.n > 0 else 1
            # Общие соседи
            neighbors_u = set(idx for idx in graph.adj[u])
            neighbors_t = set(idx for idx in graph.adj[target])
            common = len(neighbors_u & neighbors_t)
            if common > 0:
                # Ожидаемое минимальное время ребра (можно вычислить один раз)
                min_mean = min(e.distribution.mean(graph.current_time) for e in graph.edges)
                return 2.0 * min_mean
            # Иначе – оценка на основе степени
            return (deg_u + deg_t) / (2.0 * max_deg)
        return heuristic