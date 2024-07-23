import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance_matrix
from ortools.linear_solver import pywraplp
from prettytable import PrettyTable
from argparse import ArgumentParser
import logging

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


def load_data(file_path: str, cust_size: int):
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.head(cust_size + 1)

    n = len(df) - 1
    capacity = df['CAPACITY'].iloc[0]
    customers = list(range(1, n + 1))
    all_customers = [0] + customers + [n + 1]

    coordinates = df.iloc[:, 1:3]
    coordinates.loc[n + 1, :] = coordinates.loc[0, :]

    dist_matrix = pd.DataFrame(distance_matrix(coordinates.values, coordinates.values), index=coordinates.index,
                               columns=coordinates.index)

    ready_times = [df['READYTIME'].iloc[i] for i in range(n + 1)] + [df['READYTIME'].iloc[0]]
    due_times = [df['DUETIME'].iloc[i] for i in range(n + 1)] + [df['DUETIME'].iloc[0]]
    service_times = [df['SERVICETIME'].iloc[i] for i in range(n + 1)] + [df['SERVICETIME'].iloc[0]]
    demands = [df['DEMAND'].iloc[i] for i in range(n + 1)] + [0]

    return dist_matrix, capacity, customers, all_customers, ready_times, due_times, service_times, demands


def solve_vrptw(file: str, cust_size: int, large_constant: int):
    logger.info(colorful_log(f'Starting to process file: {file} with {cust_size} customers', 'blue'))
    file_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'src', 'dataset', f'{file}.csv')
    dist_matrix, capacity, customers, all_customers, ready_times, due_times, service_times, demands = load_data(
        file_path, cust_size)

    start_time = time.time()
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.EnableOutput()

    if not solver:
        logger.error(colorful_log('SCIP solver is not available.', 'red'))
        return file, cust_size, 'NA', 'NA', 'NA', 'NA'

    vehicles = list(range(1, 26))

    distance = {(i, j): dist_matrix.at[i, j] for i in all_customers for j in all_customers}
    travel_time = {(i, j): dist_matrix.at[i, j] for i in all_customers for j in all_customers}

    X = [(i, j, k) for i in all_customers for j in all_customers for k in vehicles if i != j]
    S = [(i, k) for i in all_customers for k in vehicles]

    x = {}
    for i, j, k in X:
        x[i, j, k] = solver.BoolVar(f'x[{i},{j},{k}]')

    s = {}
    for i, k in S:
        s[i, k] = solver.NumVar(0.0, solver.infinity(), f's[{i},{k}]')

    logger.info(colorful_log('Adding constraints to the model...', 'yellow'))

    for i in customers:
        solver.Add(sum(x[i, j, k] for j in all_customers for k in vehicles if j != i) == 1)

    for k in vehicles:
        solver.Add(sum(demands[i] * x[i, j, k] for i in customers for j in all_customers if i != j) <= capacity)

    for k in vehicles:
        solver.Add(sum(x[0, j, k] for j in all_customers if j != 0) == 1)

    for p in customers:
        for k in vehicles:
            solver.Add(sum(x[i, p, k] for i in all_customers if i != p) - sum(
                x[p, j, k] for j in all_customers if p != j) == 0)

    for k in vehicles:
        solver.Add(sum(x[i, cust_size + 1, k] for i in all_customers if i != cust_size + 1) == 1)

    for i, j, k in X:
        if i != j:
            solver.Add(
                s[i, k] + service_times[i] + travel_time[i, j] - large_constant * (1 - x[i, j, k]) - s[j, k] <= 0)

    for k in vehicles:
        solver.Add(s[0, k] == 0)

    for i, k in S:
        if i != 0:
            solver.Add(s[i, k] >= ready_times[i])
            solver.Add(s[i, k] <= due_times[i])

    objective = solver.Sum(distance[i, j] * x[i, j, k] for i, j, k in X)
    solver.Minimize(objective)

    logger.info(colorful_log('Starting the solver...', 'yellow'))
    solve_start_time = time.time()
    status = solver.Solve()
    solve_end_time = time.time()

    running_time = round(solve_end_time - solve_start_time, 2)
    elapsed_time = round(solve_end_time - start_time, 2)

    if status == pywraplp.Solver.OPTIMAL:
        route = [x[0, i, k] for i in customers for k in vehicles if x[0, i, k].solution_value() == 1]
        no_vehicles = len(route)
        objective_value = round(solver.Objective().Value(), 2)
        logger.info(colorful_log(
            f'File: {file} with {cust_size} customers, {no_vehicles} vehicles, objective value: {objective_value}, elapsed time: {elapsed_time}, running time: {running_time}',
            'green'))
        logger.info(colorful_log(f'Finished processing file: {file}', 'blue'))
        return file, cust_size, no_vehicles, objective_value, elapsed_time, running_time
    else:
        logger.info(colorful_log(
            f'File: {file} with {cust_size} NA NA elapsed time: {elapsed_time}, running time: {running_time}', 'red'))
        logger.info(colorful_log(f'Finished processing file: {file}', 'blue'))
        return file, cust_size, 'NA', 'NA', elapsed_time, running_time


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=["file", "customers", "vehicles", "objective", "elapsed_time", "running_time"])
    df.to_csv(output_file, index=False)
    logger.info(colorful_log(f'Results saved to {output_file}', 'green'))


def main():
    parser = ArgumentParser(description="VRPTW Exact Solver")
    parser.add_argument('--files', nargs='+', default=["R101", "C101", "RC101"], help='List of files to process')
    parser.add_argument('--customers', type=int, default=10, help='Number of customers')
    parser.add_argument('--large_constant', type=int, default=10000, help='Large constant for MTZ constraints')
    args = parser.parse_args()

    summary_table = PrettyTable()
    summary_table.field_names = ["File", "Customers", "Vehicles", "Objective", "Elapsed Time", "Running Time"]

    results = []
    for file in args.files:
        result = solve_vrptw(file, args.customers, args.large_constant)
        results.append(result)

    for result in results:
        summary_table.add_row(result)

    print(summary_table)
    save_results_to_csv(results, f'./results/exact_results_{args.customers}.csv')


if __name__ == "__main__":
    main()
