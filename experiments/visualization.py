"""
Модуль для визуализации результатов экспериментов и графов с путями.
Использует pandas, matplotlib, seaborn, networkx.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional, List, Dict, Union
from datetime import datetime

# Настройка стиля
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ------------------------------------------
# Функции для работы с результатами
# ------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    """
    Загружает результаты из CSV-файла.
    """
    df = pd.read_csv(csv_path)
    # Преобразуем колонку времени в числовой тип
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    return df

def aggregate_results(df: pd.DataFrame,
                      group_by: List[str],
                      agg_funcs: Optional[Dict] = None) -> pd.DataFrame:
    """
    Агрегирует данные по заданным колонкам.
    По умолчанию: среднее, стандартное отклонение, количество.
    """
    if agg_funcs is None:
        agg_funcs = {
            'time': ['mean', 'std', 'count'],
            'path_length': ['mean', 'std']
        }
    return df.groupby(group_by).agg(agg_funcs).round(3)

# ------------------------------------------
# Функции для построения графиков
# ------------------------------------------

def plot_time_by_algorithm(df: pd.DataFrame,
                           title: str = "Сравнение времени пути по алгоритмам",
                           save_path: Optional[str] = None):
    """
    Box plot распределения времени пути для каждого алгоритма.
    """
    if df.empty:
        print("Нет данных для построения графика.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='algorithm', y='time')
    plt.title(title, fontsize=16)
    plt.ylabel("Время пути")
    plt.xlabel("Алгоритм")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_time_by_distribution(df: pd.DataFrame,
                              algorithm: Optional[str] = None,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None):
    """
    Bar plot среднего времени пути для разных распределений.
    Версия для старых версий seaborn (использует matplotlib напрямую).
    """
    if df.empty:
        print("Нет данных для построения графика.")
        return

    if algorithm:
        df_filtered = df[df['algorithm'] == algorithm]
        title = title or f"Среднее время пути для алгоритма {algorithm}"
    else:
        df_filtered = df
        title = title or "Среднее время пути по распределениям и алгоритмам"

    if df_filtered.empty:
        print("Нет данных после фильтрации.")
        return

    agg = df_filtered.groupby(['distribution', 'algorithm'])['time'].agg(['mean', 'std']).reset_index()

    if agg.empty:
        print("Нет данных для построения графика.")
        return

    plt.figure(figsize=(12, 6))
    # Получаем уникальные распределения и алгоритмы
    distributions = agg['distribution'].unique()
    algorithms = agg['algorithm'].unique()
    n_dist = len(distributions)
    n_alg = len(algorithms)
    width = 0.8 / n_alg  # ширина одного бара

    for i, alg in enumerate(algorithms):
        data = agg[agg['algorithm'] == alg]
        # Приводим данные к порядку distributions
        means = []
        stds = []
        for d in distributions:
            val = data[data['distribution'] == d]
            if not val.empty:
                means.append(val['mean'].values[0])
                stds.append(val['std'].values[0])
            else:
                means.append(0)
                stds.append(0)
        x = np.arange(n_dist) + i * width - (n_alg - 1) * width / 2
        plt.bar(x, means, width, yerr=stds, label=alg, capsize=3)

    plt.xticks(np.arange(n_dist), distributions, rotation=45)
    plt.ylabel("Среднее время пути")
    plt.xlabel("Распределение")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_time_vs_graph_size(df: pd.DataFrame,
                            algorithm: Optional[str] = None,
                            distribution: Optional[str] = None,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None):
    """
    Зависимость времени пути от размера графа (с ошибками).
    """
    if df.empty:
        print("Нет данных для построения графика.")
        return

    df_filtered = df.copy()
    if algorithm:
        df_filtered = df_filtered[df_filtered['algorithm'] == algorithm]
    if distribution:
        df_filtered = df_filtered[df_filtered['distribution'] == distribution]

    if df_filtered.empty:
        print("Нет данных после фильтрации.")
        return

    agg = df_filtered.groupby(['graph_size', 'algorithm'])['time'].agg(['mean', 'std']).reset_index()

    if agg.empty:
        print("Нет данных для построения графика.")
        return

    plt.figure(figsize=(10, 6))
    for alg in agg['algorithm'].unique():
        data = agg[agg['algorithm'] == alg]
        plt.errorbar(data['graph_size'], data['mean'], yerr=data['std'],
                     marker='o', label=alg, capsize=5, linewidth=2)
    plt.title(title or "Зависимость времени пути от размера графа", fontsize=16)
    plt.xlabel("Размер графа (число вершин)")
    plt.ylabel("Среднее время пути")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_adaptation_frequency(df: pd.DataFrame,
                              threshold: float = 0.1,
                              save_path: Optional[str] = None):
    """
    Оценивает частоту, с которой адаптивный алгоритм меняет путь по сравнению с ожидаемым.
    Для каждого запуска сравнивается путь, найденный адаптивным и ожидаемым алгоритмом.
    Если пути различаются (коэффициент различия > threshold), считаем перестроение.

    ВНИМАНИЕ: требует сохранённых путей в результатах (колонка 'path').
    """
    if df.empty:
        print("Нет данных для построения графика.")
        return

    if 'path' not in df.columns:
        print("Колонка 'path' не найдена. Невозможно вычислить частоту адаптации.")
        return

    def parse_path(path_str):
        if pd.isna(path_str) or path_str == '':
            return []
        return [int(x) for x in path_str.split('-')]

    def path_similarity(p1, p2):
        if not p1 or not p2:
            return 0.0
        set1, set2 = set(p1), set(p2)
        return len(set1 & set2) / max(len(set1), len(set2))

    # Собираем пары адаптивный-ожидаемый для каждого запуска
    # Предполагаем, что в данных есть колонки: run, algorithm, path
    runs = df['run'].unique()
    freqs = []
    for run in runs:
        df_run = df[df['run'] == run]
        # Получаем пути для каждого алгоритма
        paths = {}
        for alg in df_run['algorithm'].unique():
            path_data = df_run[df_run['algorithm'] == alg]['path'].values
            if len(path_data) > 0:
                paths[alg] = parse_path(path_data[0])
        if 'adaptive_dijkstra' in paths and 'expected_dijkstra' in paths:
            sim = path_similarity(paths['adaptive_dijkstra'], paths['expected_dijkstra'])
            if sim < (1 - threshold):
                freqs.append(1)
            else:
                freqs.append(0)
    if freqs:
        freq = np.mean(freqs)
        plt.figure(figsize=(6,4))
        plt.bar(['Частота перестроений'], [freq])
        plt.ylabel('Доля запусков с перестроением')
        plt.title(f'Адаптивность алгоритма (порог {threshold})')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("Недостаточно данных для расчёта частоты адаптации.")

# ------------------------------------------
# Функции для визуализации графов с путями
# ------------------------------------------

def _to_networkx(graph):
    """
    Преобразует наш кастомный граф (из модуля graph) в networkx.DiGraph.
    Если передан уже nx.Graph, возвращает его же.
    """
    if isinstance(graph, nx.Graph):
        return graph
    # Предполагаем, что у graph есть атрибуты n, edges
    try:
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.n))
        for edge in graph.edges:
            G.add_edge(edge.u, edge.v, weight=edge.current_weight if edge.current_weight is not None else 0.0)
        return G
    except AttributeError:
        raise TypeError("graph должен быть экземпляром класса Graph из graph.py или networkx.Graph")

def visualize_graph_with_paths(graph,
                               paths: Dict[str, List[int]],
                               title: str = "Пути на графе",
                               layout: str = "spring",
                               edge_labels: bool = False,
                               save_path: Optional[str] = None):
    """
    Визуализирует граф с выделенными путями.

    Параметры:
    - graph: объект нашего класса Graph или networkx.Graph
    - paths: словарь {имя_алгоритма: список вершин}
    - title: заголовок
    - layout: тип раскладки ('spring', 'kamada_kawai', 'circular', 'shell', 'planar')
    - edge_labels: показывать ли веса рёбер
    - save_path: путь для сохранения изображения
    """
    G = _to_networkx(graph)

    # Выбор раскладки
    layouts = {
        'spring': nx.spring_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'circular': nx.circular_layout,
        'shell': nx.shell_layout,
        'planar': nx.planar_layout,
        'random': nx.random_layout
    }
    pos = layouts.get(layout, nx.spring_layout)(G)

    plt.figure(figsize=(14, 10))

    # Рисуем основу графа
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5, width=1)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Если нужны подписи весов
    if edge_labels:
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8)

    # Цвета для путей
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
    for i, (alg_name, path) in enumerate(paths.items()):
        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            # Проверяем, что все рёбра существуют в графе
            path_edges = [e for e in path_edges if G.has_edge(e[0], e[1])]
            if path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                       edge_color=colors[i % len(colors)],
                                       width=3, label=alg_name)
                # Выделяем вершины пути
                nx.draw_networkx_nodes(G, pos, nodelist=path,
                                       node_color=colors[i % len(colors)],
                                       node_size=600, edgecolors='black')

    plt.title(title, fontsize=16)
    if paths:
        plt.legend(fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_graph_with_path(graph,
                              path: List[int],
                              title: str = "Кратчайший путь на графе",
                              layout: str = "spring",
                              edge_labels: bool = False,
                              save_path: Optional[str] = None):
    """Упрощённая версия для одного пути."""
    visualize_graph_with_paths(graph, {"Path": path}, title, layout, edge_labels, save_path)

def plot_path_comparison(graph,
                         source: int,
                         target: int,
                         paths_dict: Dict[str, List[int]],
                         save_path: Optional[str] = None):
    """
    Строит граф с путями и подписями весов рёбер.
    """
    G = _to_networkx(graph)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    # Подписи весов
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    colors = plt.cm.tab10.colors
    for i, (name, path) in enumerate(paths_dict.items()):
        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            path_edges = [e for e in path_edges if G.has_edge(e[0], e[1])]
            if path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                       edge_color=colors[i], width=3, label=f"{name}")
                nx.draw_networkx_nodes(G, pos, nodelist=path,
                                       node_color=colors[i], node_size=600)

    plt.title(f"Сравнение путей из {source} в {target}", fontsize=16)
    plt.legend()
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def save_paths_summary(graph, paths_dict, filename):
    """
    Сохраняет текстовое описание путей и их длин.
    """
    with open(filename, 'w') as f:
        f.write(f"Граф: {graph.n} вершин, {len(graph.edges)} рёбер\n")
        f.write("="*40 + "\n")
        for name, path in paths_dict.items():
            if path:
                # Вычислим длину пути (сумму весов рёбер)
                length = 0.0
                valid = True
                for i in range(len(path)-1):
                    w = graph.get_weight(path[i], path[i+1])
                    if w is None:
                        valid = False
                        break
                    length += w
                if valid:
                    f.write(f"{name}: путь {path}, длина {length:.3f}\n")
                else:
                    f.write(f"{name}: путь {path} (не все веса доступны)\n")
            else:
                f.write(f"{name}: путь не найден\n")
    print(f"Сводка сохранена в {filename}")

# ------------------------------------------
# Утилита для сохранения рисунков с меткой времени
# ------------------------------------------

def save_figure(plt, base_name: str, ext: str = "png") -> str:
    """
    Сохраняет текущую фигуру с именем, содержащим дату и время.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{ext}"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Рисунок сохранён как {filename}")
    return filename