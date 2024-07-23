import os
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
from prettytable import PrettyTable
from argparse import ArgumentParser
from utils import calc_distance, check_time, start_times
from metaheuristic import mutate_routes, vns
import logging
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

plt.style.use('default')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=10)

dir_path = os.path.dirname(os.path.realpath('__file__'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

RESET = "\033[0m"
COLORS = {
    'green': "\033[92m",
    'blue': "\033[94m",
    'yellow': "\033[93m",
    'red': "\033[91m"
}


def colorful_log(message: str, color: str) -> str:
    return f"{COLORS[color]}{message}{RESET}"


def graph(tours: List[List[int]], x: List[float], y: List[float], dist_matrix: np.ndarray, file_name: str,
          cust_size: int = 50):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x[0], y[0], 'r^', markersize=12, label='Depot')
    ax.scatter(x[1:], y[1:], s=10, c='k', marker='v', label='Customers')
    for i, (xi, yi) in enumerate(zip(x[1:], y[1:]), 1):
        ax.annotate(i, (xi + 0.2, yi + 0.2), size=8, color='k', fontweight='bold', fontfamily='serif')
    for tour in tours:
        ax.plot([x[i] for i in tour], [y[i] for i in tour], linewidth=2, zorder=0)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_title(
        f"Solomon’s I1: {file_name} wt {cust_size}, distance: {calc_distance(tours, dist_matrix):.2f}, vehicles: {len(tours)}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.legend(fontsize=10)
    os.makedirs('./figures/solomon/', exist_ok=True)
    fig.savefig(f'./figures/solomon/slm_{file_name}_{cust_size}.pdf', format='pdf')
    plt.close(fig)


def load_data(file_path: str, cust_size: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    data = np.genfromtxt(file_path, delimiter=',')
    df = data[1:cust_size + 2]
    customer, xcoor, ycoor, demand, readytime, duetime, servicetime, capacity = (
        df[:, 0], df[:, 1], df[:, 2], df[:, 3], df[:, 4], df[:, 5], df[:, 6], df[0, -1]
    )

    dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
    for i in range(cust_size + 1):
        for j in range(cust_size + 1):
            dist_matrix[i][j] = np.sqrt((xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)

    return customer, xcoor, ycoor, demand, readytime, duetime, servicetime, capacity, dist_matrix


def solve_initial_solution(cust_size: int, dist_matrix: np.ndarray, servicetime: List[float], readytime: List[float],
                           duetime: List[float], demand: List[int], capacity: int, muy: int, lam: int, a1: int,
                           a2: int) -> List[List[int]]:
    unrouted_customers = list(range(1, cust_size + 1))
    initial_tours = []

    while unrouted_customers:
        unrouted_distances = [dist_matrix[0][i] for i in unrouted_customers]
        furthest_idx = unrouted_distances.index(max(unrouted_distances))
        tour = [0, unrouted_customers[furthest_idx], 0]
        del unrouted_customers[furthest_idx]

        while True:
            feasible_customers, c1, insert_positions = [], [], []
            for u in unrouted_customers:
                min_cost, position = float("inf"), -1
                for p in range(len(tour) - 1):
                    new_tour = tour[:p + 1] + [u] + tour[p + 1:]
                    if check_time(new_tour, dist_matrix, servicetime, readytime, duetime) and sum(
                            demand[new_tour]) <= capacity:
                        c11 = dist_matrix[tour[p]][u] + dist_matrix[u][tour[p + 1]] - muy * dist_matrix[tour[p]][
                            tour[p + 1]]
                        c12 = start_times(new_tour, dist_matrix, servicetime, readytime)[p + 2] - \
                              start_times(tour, dist_matrix, servicetime, readytime)[p + 1]
                        cost = a1 * c11 + a2 * c12
                        if cost < min_cost:
                            min_cost, position = cost, p
                if position != -1:
                    feasible_customers.append(u)
                    insert_positions.append(position)
                    c1.append(min_cost)

            if feasible_customers:
                c2 = [lam * dist_matrix[0][fc] - c1[i] for i, fc in enumerate(feasible_customers)]
                optimal_idx = c2.index(max(c2))
                tour.insert(insert_positions[optimal_idx] + 1, feasible_customers[optimal_idx])
                unrouted_customers.remove(feasible_customers[optimal_idx])
            else:
                break

        initial_tours.append(tour)

    return initial_tours


def vns_iteration(file: str, initial_tours: List[List[int]], dist_matrix: np.ndarray, servicetime: np.ndarray,
                  readytime: np.ndarray, duetime: np.ndarray, demand: np.ndarray, capacity: int, iteration: int) -> \
        Tuple[float, int, float, List[List[int]]]:
    current_tours = copy.deepcopy(initial_tours)
    shaking_neighbors = list(range(11))
    start_time = time.time()
    no_improvement, neighbor_idx, stop = 0, 0, False

    while not stop:
        shaken_tours = mutate_routes(current_tours, dist_matrix, servicetime, readytime, duetime, demand,
                                     capacity, shaking_neighbors[neighbor_idx])
        vns(shaken_tours, dist_matrix, dist_matrix, servicetime, readytime, duetime, demand, capacity)
        if calc_distance(shaken_tours, dist_matrix) < calc_distance(current_tours, dist_matrix):
            current_tours = copy.deepcopy(shaken_tours)
            neighbor_idx, no_improvement = 0, 0
        else:
            no_improvement += 1
            if no_improvement > len(shaking_neighbors):
                stop = True
            else:
                neighbor_idx = (neighbor_idx + 1) % len(shaking_neighbors)
        current_tours = [tour for tour in current_tours if len(tour) > 2]

    end_time = time.time()
    dist_result = calc_distance(current_tours, dist_matrix)
    vehicle_count = len(current_tours)
    exec_time = end_time - start_time
    logger.info(colorful_log(
        f'File: {file} | Iteration: {iteration + 1} | Distance = {dist_result} | Vehicles = {vehicle_count} | Time = {exec_time}',
        'green'))
    return dist_result, vehicle_count, exec_time, current_tours


def run_vns(file_names: List[str], cust_size: int, mu: int, lam: int, a1: int, a2: int, iterations: int = 10):
    summary_table = PrettyTable()
    summary_table.field_names = ["File", "Best Distance", "Avg Distance", "Std Distance", "Best Vehicles",
                                 "Avg Vehicles", "Std Vehicles", "Avg Time"]

    results = []

    for file in file_names:
        file_path = os.path.join(dir_path, 'src', 'dataset', f'{file}.csv')
        customer, xcoor, ycoor, demand, readytime, duetime, servicetime, capacity, dist_matrix = load_data(file_path,
                                                                                                           cust_size)

        logger.info(colorful_log(f'Starting file: {file} with {cust_size} customers', 'blue'))

        initial_tours = solve_initial_solution(cust_size, dist_matrix, servicetime, readytime, duetime, demand,
                                               capacity, mu, lam, a1, a2)
        logger.info(colorful_log(
            f'Initial solution | Distance = {calc_distance(initial_tours, dist_matrix)} | Vehicles = {len(initial_tours)}',
            'red'))

        random.seed(42)

        iter_results = []
        for i in range(iterations):
            result = vns_iteration(file, initial_tours, dist_matrix, servicetime, readytime, duetime, demand, capacity,
                                   i)
            iter_results.append(result)

        dist_results = [result[0] for result in iter_results]
        vehicle_counts = [result[1] for result in iter_results]
        exec_times = [result[2] for result in iter_results]
        current_tours = iter_results[0][3]

        summary_table.add_row([
            file,
            min(dist_results), np.mean(dist_results), np.std(dist_results),
            min(vehicle_counts), np.mean(vehicle_counts), np.std(vehicle_counts),
            np.mean(exec_times)
        ])
        graph(current_tours, xcoor, ycoor, dist_matrix, file, cust_size)
        results.append([file, min(dist_results), np.mean(dist_results), np.std(dist_results),
                        min(vehicle_counts), np.mean(vehicle_counts), np.std(vehicle_counts), np.mean(exec_times)])

    print(summary_table)
    save_results_to_csv(results, f'./results/solomon/slm_results_{cust_size}.csv')


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=["file", "best_distance", "avg_distance", "std_distance", "best_vehicles",
                                        "avg_vehicles", "std_vehicles", "avg_time"])
    df.to_csv(output_file, index=False)
    logger.info(colorful_log(f'Results saved to {output_file}', 'green'))


def main():
    parser = ArgumentParser(description="Vehicle Routing Problem Solver")
    parser.add_argument('--files', nargs='+',
                        default=["R101", "C101", "RC101"],
                        help='List of files to process')
    parser.add_argument('--customers', type=int, default=50, help='Number of customers')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for Variable Neighborhood Search')
    args = parser.parse_args()

    run_vns(args.files, args.customers, mu=1, lam=2, a1=0, a2=1, iterations=args.iterations)


if __name__ == "__main__":
    main()

# python3 ./src/solomon.py --files R101 C101 RC101 R102 C102 RC102 R103 C103 RC103 R104 C104 RC104 R105 C105 RC105 R106 C106 RC106 R107 C107 RC107 R108 C108 RC108 --customers 50 --iterations 10
