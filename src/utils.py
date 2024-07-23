from typing import List
import numpy as np


def calc_distance(tours: List[List[int]], dist_matrix: np.ndarray) -> float:
    return sum(dist_matrix[tour[i], tour[i + 1]] for tour in tours for i in range(len(tour) - 1))


def check_time(tour: List[int], travel_time: np.ndarray, service_time: List[float], ready_time: List[float],
               due_time: List[float]) -> bool:
    current_time = 0
    return all((current_time := max(current_time, ready_time[tour[i - 1]]) + service_time[tour[i - 1]] + travel_time[
        tour[i - 1], tour[i]]) <= due_time[tour[i]] for i in range(1, len(tour)))


def start_times(tour: List[int], travel_time: np.ndarray, service_time: List[float], ready_time: List[float]) -> List[
    float]:
    times = [0]
    current_time = 0
    for i in range(1, len(tour)):
        current_time = max(current_time, ready_time[tour[i - 1]]) + service_time[tour[i - 1]] + travel_time[
            tour[i - 1], tour[i]]
        times.append(max(current_time, ready_time[tour[i]]))
    return times


def calc_total_time(tours: List[List[int]], travel_time: np.ndarray, service_time: List[float],
                    ready_time: List[float]) -> float:
    return sum(start_times(tour, travel_time, service_time, ready_time)[-1] for tour in tours)


def check_sequence(sequence, tour):
    i = tour.index(sequence[0])
    if tour[i + 1] == sequence[1]:
        return True
    else:
        return False
