import numpy as np
from typing import List, Tuple
from utils import calc_distance, check_time


def opt2_move(tour: List[int], dist_matrix: np.ndarray, travel_time: np.ndarray, service_time: List[float],
              ready_time: List[float], due_time: List[float]) -> Tuple[int, int, float]:
    best_improvement, best_start, best_end = 0, -1, -1
    n = len(tour)
    if n >= 5:
        for start in range(n - 3):
            for end in range(start + 2, n - 1):
                new_tour = tour[:start + 1] + tour[start + 1:end + 1][::-1] + tour[end + 1:]
                if check_time(new_tour, travel_time, service_time, ready_time, due_time):
                    improvement = (dist_matrix[tour[start], tour[end]] + dist_matrix[tour[start + 1], tour[end + 1]]
                                   - dist_matrix[tour[start], tour[start + 1]] - dist_matrix[tour[end], tour[end + 1]])
                    if improvement < best_improvement:
                        best_improvement, best_start, best_end = improvement, start, end
    return best_start, best_end, best_improvement


def opt2_search(tours: List[List[int]], dist_matrix: np.ndarray, travel_time: np.ndarray, service_time: List[float],
                ready_time: List[float], due_time: List[float]) -> Tuple[List[int], List[int], List[int], float]:
    total_improvement = 0
    improved_tours, improved_starts, improved_ends = [], [], []
    for idx, tour in enumerate(tours):
        start, end, improvement = opt2_move(tour, dist_matrix, travel_time, service_time, ready_time, due_time)
        if start != -1:
            total_improvement += improvement
            improved_tours.append(idx)
            improved_starts.append(start)
            improved_ends.append(end)
    return improved_tours, improved_starts, improved_ends, total_improvement


def or_opt_move(tour: List[int], dist_matrix: np.ndarray, travel_time: np.ndarray, service_time: List[float],
                ready_time: List[float], due_time: List[float], k: int) -> Tuple[int, int, int, float]:
    best_improvement = 0
    best_start, best_end, best_insert = -1, -1, -1
    tour_length = len(tour)

    if tour_length >= k + 3:
        for start in range(tour_length - k - 1):
            end = start + k
            for insert in range(tour_length - 1):
                if insert < start or end < insert:
                    if insert < start:
                        new_tour = tour[:insert + 1] + tour[start + 1:end + 1] + tour[insert + 1:start + 1] + tour[
                                                                                                              end + 1:]
                    else:
                        new_tour = tour[:start + 1] + tour[end + 1:insert + 1] + tour[start + 1:end + 1] + tour[
                                                                                                           insert + 1:]

                    if check_time(new_tour, travel_time, service_time, ready_time, due_time):
                        removal_cost = (
                                dist_matrix[tour[start], tour[start + 1]] + dist_matrix[tour[end], tour[end + 1]] +
                                dist_matrix[tour[insert], tour[insert + 1]])
                        insertion_cost = (dist_matrix[tour[start], tour[end + 1]] + dist_matrix[
                            tour[insert], tour[start + 1]] +
                                          dist_matrix[tour[end], tour[insert + 1]])
                        improvement = removal_cost - insertion_cost

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_start, best_end, best_insert = start, end, insert

    return best_start, best_end, best_insert, best_improvement


def or_opt_search(tours: List[List[int]], dist_matrix: np.ndarray, travel_time: np.ndarray, service_time: List[float],
                  ready_time: List[float], due_time: List[float], k: int) -> Tuple[
    List[int], List[int], List[int], List[int], float]:
    total_improvement = 0
    improved_tours, start_indices, end_indices, insert_positions = [], [], [], []

    for tour_idx, tour in enumerate(tours):
        start, end, insert, improvement = or_opt_move(tour, dist_matrix, travel_time, service_time, ready_time,
                                                      due_time, k)
        if start != -1:
            total_improvement += improvement
            improved_tours.append(tour_idx)
            start_indices.append(start)
            end_indices.append(end)
            insert_positions.append(insert)

    return improved_tours, start_indices, end_indices, insert_positions, total_improvement


def relocate(tour_a: List[int], tour_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
             ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, float]:
    min_gain = float('inf')
    best_cust, best_pos = -1, -1

    for i in range(1, len(tour_a) - 1):
        if demand[tour_a[i]] + sum(demand[stop] for stop in tour_b) <= cap:
            for j in range(len(tour_b) - 1):
                new_tour_b = tour_b[:j + 1] + [tour_a[i]] + tour_b[j + 1:]
                if check_time(new_tour_b, travel, service, ready, due):
                    removal_cost = dist[tour_a[i - 1], tour_a[i]] + dist[tour_a[i], tour_a[i + 1]] - dist[
                        tour_a[i - 1], tour_a[i + 1]]
                    insertion_cost = dist[tour_b[j], tour_a[i]] + dist[tour_a[i], tour_b[j + 1]] - dist[
                        tour_b[j], tour_b[j + 1]]
                    gain = insertion_cost - removal_cost
                    if gain < min_gain:
                        min_gain, best_cust, best_pos = gain, i, j

    return best_cust, best_pos, min_gain


def relocate_search(tours: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                    ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[
    int, int, int, int, float]:
    min_gain = float('inf')
    best_a, best_b, best_cust, best_pos = -1, -1, -1, -1

    for idx_a, tour_a in enumerate(tours):
        for idx_b, tour_b in enumerate(tours):
            if idx_a != idx_b:
                cust, pos, gain = relocate(tour_a, tour_b, dist, travel, service, ready, due, demand, cap)
                if gain < min_gain:
                    min_gain = gain
                    best_a, best_b = idx_a, idx_b
                    best_cust, best_pos = cust, pos

    return best_a, best_b, best_cust, best_pos, min_gain


def exchange(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
             ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, float]:
    max_gain = float('inf')
    best_i, best_j = -1, -1

    for i in range(1, len(route_a) - 1):
        for j in range(1, len(route_b) - 1):
            demand_a_new = demand[route_b[j]] + sum(demand[route_a]) - demand[route_a[i]]
            demand_b_new = demand[route_a[i]] + sum(demand[route_b]) - demand[route_b[j]]

            if demand_a_new <= cap and demand_b_new <= cap:
                new_route_a = route_a[:i] + [route_b[j]] + route_a[i + 1:]
                new_route_b = route_b[:j] + [route_a[i]] + route_b[j + 1:]

                if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel, service,
                                                                                       ready, due):
                    cost_a = dist[route_a[i - 1], route_b[j]] + dist[route_b[j], route_a[i + 1]] - dist[
                        route_a[i - 1], route_a[i]] - dist[route_a[i], route_a[i + 1]]
                    cost_b = dist[route_b[j - 1], route_a[i]] + dist[route_a[i], route_b[j + 1]] - dist[
                        route_b[j - 1], route_b[j]] - dist[route_b[j], route_b[j + 1]]

                    total_gain = cost_a + cost_b

                    if total_gain < max_gain:
                        max_gain, best_i, best_j = total_gain, i, j

    return best_i, best_j, max_gain


def exchange_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                    ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[
    int, int, int, int, float]:
    max_gain = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_j = -1, -1

    for idx_a in range(len(routes) - 1):
        for idx_b in range(idx_a + 1, len(routes)):
            i, j, gain = exchange(routes[idx_a], routes[idx_b], dist, travel, service, ready, due, demand, cap)

            if gain < max_gain:
                max_gain = gain
                best_route_a, best_route_b = idx_a, idx_b
                best_i, best_j = i, j

    return best_route_a, best_route_b, best_i, best_j, max_gain


def cross(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
          ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, int, int, float]:
    min_cost = float('inf')
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for i in range(1, len(route_a) - 2):
        for k in range(i + 1, len(route_a) - 1):
            for j in range(1, len(route_b) - 2):
                for l in range(j + 1, len(route_b) - 1):
                    new_demand_a = sum(demand[route_b[j:l + 1]]) + sum(demand[route_a]) - sum(demand[route_a[i:k + 1]])
                    new_demand_b = sum(demand[route_a[i:k + 1]]) + sum(demand[route_b]) - sum(demand[route_b[j:l + 1]])

                    if new_demand_a <= cap and new_demand_b <= cap:
                        new_route_a = route_a[:i] + route_b[j:l + 1] + route_a[k + 1:]
                        new_route_b = route_b[:j] + route_a[i:k + 1] + route_b[l + 1:]

                        if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel,
                                                                                               service, ready, due):
                            cost_before = (dist[route_a[i - 1], route_a[i]] + dist[route_a[k], route_a[k + 1]] +
                                           dist[route_b[j - 1], route_b[j]] + dist[route_b[l], route_b[l + 1]])
                            cost_after = (dist[route_a[i - 1], route_b[j]] + dist[route_b[l], route_a[k + 1]] +
                                          dist[route_b[j - 1], route_a[i]] + dist[route_a[k], route_b[l + 1]])
                            cost_diff = cost_after - cost_before

                            if cost_diff < min_cost:
                                min_cost = cost_diff
                                best_i, best_k, best_j, best_l = i, k, j, l
    return best_i, best_k, best_j, best_l, min_cost


def cross_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                 ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[
    int, int, int, int, int, int, float]:
    min_cost = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for idx_a in range(len(routes) - 1):
        for idx_b in range(idx_a + 1, len(routes)):
            i, k, j, l, cost = cross(routes[idx_a], routes[idx_b], dist, travel, service, ready, due, demand, cap)

            if cost < min_cost:
                min_cost = cost
                best_route_a, best_route_b = idx_a, idx_b
                best_i, best_k, best_j, best_l = i, k, j, l
    return best_route_a, best_route_b, best_i, best_k, best_j, best_l, min_cost


def icross(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
           ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, int, int, float]:
    min_cost = float('inf')
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for i in range(1, len(route_a) - 2):
        for k in range(i + 1, len(route_a) - 1):
            for j in range(1, len(route_b) - 2):
                for l in range(j + 1, len(route_b) - 1):
                    new_demand_a = sum(demand[route_b[j:l + 1]]) + sum(demand[route_a]) - sum(demand[route_a[i:k + 1]])
                    new_demand_b = sum(demand[route_a[i:k + 1]]) + sum(demand[route_b]) - sum(demand[route_b[j:l + 1]])

                    if new_demand_a <= cap and new_demand_b <= cap:
                        new_route_a = route_a[:i] + route_b[j:l + 1][::-1] + route_a[k + 1:]
                        new_route_b = route_b[:j] + route_a[i:k + 1][::-1] + route_b[l + 1:]

                        if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel,
                                                                                               service, ready, due):
                            cost_before = (dist[route_a[i - 1], route_a[i]] + dist[route_a[k], route_a[k + 1]] +
                                           dist[route_b[j - 1], route_b[j]] + dist[route_b[l], route_b[l + 1]])
                            cost_after = (dist[route_a[i - 1], route_b[l]] + dist[route_b[j], route_a[k + 1]] +
                                          dist[route_b[j - 1], route_a[k]] + dist[route_a[i], route_b[l + 1]])
                            reversed_cost_a = calc_distance([route_a[i:k + 1][::-1]], dist) - calc_distance(
                                [route_a[i:k + 1]], dist)
                            reversed_cost_b = calc_distance([route_b[j:l + 1][::-1]], dist) - calc_distance(
                                [route_b[j:l + 1]], dist)
                            total_cost = cost_after - cost_before + reversed_cost_a + reversed_cost_b

                            if total_cost < min_cost:
                                min_cost = total_cost
                                best_i, best_k, best_j, best_l = i, k, j, l

    return best_i, best_k, best_j, best_l, min_cost


def icross_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                  ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[
    int, int, int, int, int, int, float]:
    min_cost = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for idx_a in range(len(routes) - 1):
        for idx_b in range(idx_a + 1, len(routes)):
            i, k, j, l, cost = icross(routes[idx_a], routes[idx_b], dist, travel, service, ready, due, demand, cap)

            if cost < min_cost:
                min_cost = cost
                best_route_a, best_route_b = idx_a, idx_b
                best_i, best_k, best_j, best_l = i, k, j, l

    return best_route_a, best_route_b, best_i, best_k, best_j, best_l, min_cost


def geni(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
         ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, int, float]:
    min_cost = float('inf')
    best_i, best_j, best_k = -1, -1, -1

    if len(route_b) >= 4:
        total_demand_b = sum(demand[stop] for stop in route_b)
        for i in range(1, len(route_a) - 1):
            for j in range(len(route_b) - 3):
                for k in range(j + 2, len(route_b) - 1):
                    new_demand_b = demand[route_a[i]] + total_demand_b
                    if new_demand_b <= cap:
                        new_route_a = route_a[:i] + route_a[i + 1:]
                        new_route_b = route_b[:j + 1] + [route_a[i]] + [route_b[k]] + route_b[j + 1:k] + route_b[k + 1:]
                        if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel,
                                                                                               service, ready, due):
                            cost_a = dist[route_a[i - 1], route_a[i + 1]] - dist[route_a[i - 1], route_a[i]] - dist[
                                route_a[i], route_a[i + 1]]
                            cost_b = (dist[route_b[j], route_a[i]] + dist[route_a[i], route_b[k]] +
                                      dist[route_b[k], route_b[j + 1]] + dist[route_b[k - 1], route_b[k + 1]] -
                                      dist[route_b[j], route_b[j + 1]] - dist[route_b[k - 1], route_b[k]] -
                                      dist[route_b[k], route_b[k + 1]])
                            total_cost = cost_a + cost_b
                            if total_cost < min_cost:
                                min_cost, best_i, best_j, best_k = total_cost, i, j, k

    return best_i, best_j, best_k, min_cost


def geni_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float], ready: List[float],
                due: List[float], demand: List[int], cap: int) -> Tuple[int, int, int, int, int, float]:
    min_cost = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_j, best_k = -1, -1, -1

    for idx_a, route_a in enumerate(routes):
        for idx_b, route_b in enumerate(routes):
            if idx_a != idx_b:
                i, j, k, cost = geni(route_a, route_b, dist, travel, service, ready, due, demand, cap)
                if cost < min_cost:
                    min_cost = cost
                    best_route_a, best_route_b = idx_a, idx_b
                    best_i, best_j, best_k = i, j, k

    return best_route_a, best_route_b, best_i, best_j, best_k, min_cost


def opt2star(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
             ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[int, int, float]:
    min_cost = float('inf')
    best_i, best_j = -1, -1

    for i in range(len(route_a) - 1):
        for j in range(len(route_b) - 1):
            new_route_a = route_a[:i + 1] + route_b[j + 1:]
            new_route_b = route_b[:j + 1] + route_a[i + 1:]
            new_demand_a = sum(demand[stop] for stop in new_route_a)
            new_demand_b = sum(demand[stop] for stop in new_route_b)

            if new_demand_a <= cap and new_demand_b <= cap:
                if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel, service,
                                                                                       ready, due):
                    cost = (dist[route_a[i], route_b[j + 1]] + dist[route_b[j], route_a[i + 1]] -
                            dist[route_a[i], route_a[i + 1]] - dist[route_b[j], route_b[j + 1]])

                    if cost < min_cost:
                        min_cost, best_i, best_j = cost, i, j

    return best_i, best_j, min_cost


def opt2star_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                    ready: List[float], due: List[float], demand: List[int], cap: int) -> Tuple[
    int, int, int, int, float]:
    min_cost = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_j = -1, -1

    for idx_a in range(len(routes) - 1):
        for idx_b in range(idx_a + 1, len(routes)):
            i, j, cost = opt2star(routes[idx_a], routes[idx_b], dist, travel, service, ready, due, demand, cap)
            if cost < min_cost:
                min_cost = cost
                best_route_a, best_route_b = idx_a, idx_b
                best_i, best_j = i, j

    return best_route_a, best_route_b, best_i, best_j, min_cost


def interchange(route_a: List[int], route_b: List[int], dist: np.ndarray, travel: np.ndarray, service: List[float],
                ready: List[float], due: List[float], demand: List[int], cap: int, lam: int) -> Tuple[
    int, int, int, int, float]:
    min_cost = float('inf')
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for i in range(1, len(route_a) - lam):
        for k in range(i, i + lam):
            for j in range(1, len(route_b) - lam):
                for l in range(j, j + lam):
                    new_demand_a = sum(demand[route_b[j:l + 1]]) + sum(demand[route_a]) - sum(demand[route_a[i:k + 1]])
                    new_demand_b = sum(demand[route_a[i:k + 1]]) + sum(demand[route_b]) - sum(demand[route_b[j:l + 1]])

                    if new_demand_a <= cap and new_demand_b <= cap:
                        new_route_a = route_a[:i] + route_b[j:l + 1] + route_a[k + 1:]
                        new_route_b = route_b[:j] + route_a[i:k + 1] + route_b[l + 1:]

                        if check_time(new_route_a, travel, service, ready, due) and check_time(new_route_b, travel,
                                                                                               service, ready, due):
                            cost_before = calc_distance([route_a], dist) + calc_distance([route_b], dist)
                            cost_after = calc_distance([new_route_a], dist) + calc_distance([new_route_b], dist)
                            total_cost = cost_after - cost_before

                            if total_cost < min_cost:
                                min_cost, best_i, best_k, best_j, best_l = total_cost, i, k, j, l

    return best_i, best_k, best_j, best_l, min_cost


def interchange_search(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float],
                       ready: List[float], due: List[float], demand: List[int], cap: int, lam: int) -> Tuple[
    int, int, int, int, int, int, float]:
    min_cost = float('inf')
    best_route_a, best_route_b = -1, -1
    best_i, best_k, best_j, best_l = -1, -1, -1, -1

    for idx_a in range(len(routes) - 1):
        for idx_b in range(idx_a + 1, len(routes)):
            i, k, j, l, cost = interchange(routes[idx_a], routes[idx_b], dist, travel, service, ready, due, demand, cap,
                                           lam)
            if cost < min_cost:
                min_cost = cost
                best_route_a, best_route_b = idx_a, idx_b
                best_i, best_k, best_j, best_l = i, k, j, l

    return best_route_a, best_route_b, best_i, best_k, best_j, best_l, min_cost
