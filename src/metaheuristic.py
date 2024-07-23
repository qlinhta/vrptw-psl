import copy
import logging
import random
import time
from typing import List
from random import randint
import numpy as np
from search import opt2_search, or_opt_search, opt2star_search, relocate_search, exchange_search, cross_search, \
    icross_search, geni_search, interchange_search
from utils import calc_distance, check_time, start_times


def vns(routes: List[List[int]], dist: np.ndarray, travel: np.ndarray, service: List[float], ready: List[float],
        due: List[float], demand: List[int], cap: int) -> List[List[int]]:
    iteration = 0
    neighbor_order = list(range(11))
    random.shuffle(neighbor_order)
    neighbor_idx = 0
    no_improvement = 0
    stop = False

    search_functions = [
        lambda rt: opt2_search(rt, dist, travel, service, ready, due),
        lambda rt: or_opt_search(rt, dist, travel, service, ready, due, 1),
        lambda rt: or_opt_search(rt, dist, travel, service, ready, due, 2),
        lambda rt: or_opt_search(rt, dist, travel, service, ready, due, 3),
        lambda rt: opt2star_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: relocate_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: exchange_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: cross_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: icross_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: geni_search(rt, dist, travel, service, ready, due, demand, cap),
        lambda rt: interchange_search(rt, dist, travel, service, ready, due, demand, cap, 2)
    ]

    def apply_move(routes, move, indices):
        moves = {
            0: "2-opt",
            1: "Or-opt-1",
            2: "Or-opt-2",
            3: "Or-opt-3",
            4: "2-opt*",
            5: "Relocate",
            6: "Exchange",
            7: "Cross",
            8: "ICross",
            9: "Geni",
            10: "Interchange"
        }
        # logging.info(f'Applying {moves[move]} with indices {indices}')
        if move == 0:
            for t, i1, i2 in zip(*indices):
                routes[t] = routes[t][:i1 + 1] + routes[t][i1 + 1:i2 + 1][::-1] + routes[t][i2 + 1:]
        elif move in {1, 2, 3}:
            for t, i1, i2, i3 in zip(*indices):
                if i3 < i1:
                    routes[t] = routes[t][:i3 + 1] + routes[t][i1 + 1:i2 + 1] + routes[t][i3 + 1:i1 + 1] + routes[t][
                                                                                                           i2 + 1:]
                else:
                    routes[t] = routes[t][:i1 + 1] + routes[t][i2 + 1:i3 + 1] + routes[t][i1 + 1:i2 + 1] + routes[t][
                                                                                                           i3 + 1:]
        elif move == 4:
            routes[indices[0]], routes[indices[1]] = routes[indices[0]][:indices[2] + 1] + routes[indices[1]][
                                                                                           indices[3] + 1:], routes[
                                                                                                                 indices[
                                                                                                                     1]][
                                                                                                             :indices[
                                                                                                                  3] + 1] + \
                                                     routes[indices[0]][indices[2] + 1:]
        elif move == 5:
            routes[indices[1]].insert(indices[3] + 1, routes[indices[0]][indices[2]])
            del routes[indices[0]][indices[2]]
        elif move == 6:
            routes[indices[0]], routes[indices[1]] = routes[indices[0]][:indices[2]] + [
                routes[indices[1]][indices[3]]] + routes[indices[0]][indices[2] + 1:], routes[indices[1]][
                                                                                       :indices[3]] + [
                                                         routes[indices[0]][indices[2]]] + routes[indices[1]][
                                                                                           indices[3] + 1:]
        elif move in {7, 10}:
            routes[indices[0]], routes[indices[1]] = routes[indices[0]][:indices[2]] + routes[indices[1]][
                                                                                       indices[4]:indices[5] + 1] + \
                                                     routes[indices[0]][indices[3] + 1:], routes[indices[1]][
                                                                                          :indices[4]] + routes[indices[
                0]][indices[2]:indices[3] + 1] + routes[indices[1]][indices[5] + 1:]
        elif move == 8:
            routes[indices[0]], routes[indices[1]] = routes[indices[0]][:indices[2]] + routes[indices[1]][
                                                                                       indices[4]:indices[5] + 1][
                                                                                       ::-1] + routes[indices[0]][
                                                                                               indices[3] + 1:], routes[
                                                                                                                     indices[
                                                                                                                         1]][
                                                                                                                 :
                                                                                                                 indices[
                                                                                                                     4]] + \
                                                     routes[indices[0]][indices[2]:indices[3] + 1][::-1] + routes[
                                                                                                               indices[
                                                                                                                   1]][
                                                                                                           indices[
                                                                                                               5] + 1:]
        elif move == 9:
            routes[indices[0]], routes[indices[1]] = routes[indices[0]][:indices[2]] + routes[indices[0]][
                                                                                       indices[2] + 1:], routes[indices[
                1]][:indices[3] + 1] + [routes[indices[0]][indices[2]]] + [routes[indices[1]][indices[4]]] + routes[
                                                                                                                 indices[
                                                                                                                     1]][
                                                                                                             indices[
                                                                                                                 3] + 1:
                                                                                                             indices[
                                                                                                                 4]] + \
                                                     routes[indices[1]][indices[4] + 1:]

    while not stop:
        iteration += 1
        # logging.info(f'Iteration: {iteration}, Neighbor Index: {neighbor_idx}, No Improvement: {no_improvement}')
        move = neighbor_order[neighbor_idx]
        result = search_functions[move](routes)

        if round(result[-1], 5) < 0:
            # logging.info(f'Improvement found with move {move}: {result[-1]}')
            apply_move(routes, move, result[:-1])
            neighbor_idx, no_improvement = 0, 0
        else:
            no_improvement += 1
            if no_improvement > len(neighbor_order):
                stop = True
            else:
                neighbor_idx = (neighbor_idx + 1) % len(neighbor_order)

    # logging.info(f'Final iteration: {iteration}, Total routes: {len(routes)}, Total distance: {calc_distance(routes, dist)}')
    return routes


def mutate_routes(input_tour, travel_time, service_time, ready_time, due_time, demand, capacity, neighbor_type):
    n = len(input_tour) - 1
    sub_tour = copy.deepcopy(input_tour)
    shaking_start = time.time()

    if neighbor_type == 0:  # 2-opt
        while True:
            while True:
                tour = random.randint(0, n)
                if len(sub_tour[tour]) >= 5:
                    break

            pos1 = random.randint(0, len(sub_tour[tour]) - 4)
            pos2 = random.randint(pos1 + 2, len(sub_tour[tour]) - 2)

            new_tour = sub_tour[tour][:pos1 + 1] + sub_tour[tour][pos1 + 1:pos2 + 1][::-1] + sub_tour[tour][pos2 + 1:]
            time_check = check_time(new_tour, travel_time, service_time, ready_time, due_time)

            if time_check:
                sub_tour[tour] = new_tour
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type in {1, 2, 3}:  # Or-opt
        k = neighbor_type
        while True:
            while True:
                tour = random.randint(0, n)
                if len(sub_tour[tour]) >= k + 3:
                    break

            pos1 = random.randint(0, len(sub_tour[tour]) - k - 2)
            pos2 = pos1 + k
            pos3 = random.randint(0, len(sub_tour[tour]) - 2)

            while pos1 <= pos3 <= pos2:
                pos3 = random.randint(0, len(sub_tour[tour]) - 2)

            if pos3 < pos1:
                new_tour = (sub_tour[tour][:pos3 + 1] + sub_tour[tour][pos1 + 1:pos2 + 1] +
                            sub_tour[tour][pos3 + 1:pos1 + 1] + sub_tour[tour][pos2 + 1:])
            else:
                new_tour = (sub_tour[tour][:pos1 + 1] + sub_tour[tour][pos2 + 1:pos3 + 1] +
                            sub_tour[tour][pos1 + 1:pos2 + 1] + sub_tour[tour][pos3 + 1:])

            time_check = check_time(new_tour, travel_time, service_time, ready_time, due_time)

            if time_check:
                sub_tour[tour] = new_tour
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 4 and n > 0:  # 2-opt*
        while True:
            tour1 = random.randint(0, n - 1)
            tour2 = random.randint(tour1 + 1, n)
            pos1 = random.randint(1, len(sub_tour[tour1]) - 2)
            pos2 = random.randint(1, len(sub_tour[tour2]) - 2)

            new_tour1 = sub_tour[tour1][:pos1 + 1] + sub_tour[tour2][pos2 + 1:]
            new_tour2 = sub_tour[tour2][:pos2 + 1] + sub_tour[tour1][pos1 + 1:]

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 5 and n > 0:  # Relocation
        while True:
            tour1 = random.randint(0, n)
            tour2 = random.randint(0, n)
            while tour1 == tour2:
                tour2 = random.randint(0, n)

            customer = random.randint(1, len(sub_tour[tour1]) - 2)
            insert_pos = random.randint(0, len(sub_tour[tour2]) - 2)

            new_tour2 = sub_tour[tour2][:insert_pos + 1] + [sub_tour[tour1][customer]] + sub_tour[tour2][
                                                                                         insert_pos + 1:]
            if (check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    demand[sub_tour[tour1][customer]] + sum(demand[sub_tour[tour2]]) <= capacity):
                sub_tour[tour2].insert(insert_pos + 1, sub_tour[tour1][customer])
                del sub_tour[tour1][customer]
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 6 and n > 0:  # Exchange
        while True:
            tour1 = random.randint(0, n - 1)
            tour2 = random.randint(tour1 + 1, n)
            pos1 = random.randint(1, len(sub_tour[tour1]) - 2)
            pos2 = random.randint(1, len(sub_tour[tour2]) - 2)

            new_tour1 = sub_tour[tour1][:pos1] + [sub_tour[tour2][pos2]] + sub_tour[tour1][pos1 + 1:]
            new_tour2 = sub_tour[tour2][:pos2] + [sub_tour[tour1][pos1]] + sub_tour[tour2][pos2 + 1:]

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 7 and n > 0:  # CROSS
        while True:
            while True:
                tour1 = random.randint(0, n - 1)
                tour2 = random.randint(tour1 + 1, n)
                if len(sub_tour[tour1]) >= 4 and len(sub_tour[tour2]) >= 4:
                    break

            node11 = random.randint(1, len(sub_tour[tour1]) - 3)
            node12 = random.randint(node11, len(sub_tour[tour1]) - 2)
            node21 = random.randint(1, len(sub_tour[tour2]) - 3)
            node22 = random.randint(node21, len(sub_tour[tour2]) - 2)

            new_tour1 = sub_tour[tour1][:node11] + sub_tour[tour2][node21:node22 + 1] + sub_tour[tour1][node12 + 1:]
            new_tour2 = sub_tour[tour2][:node21] + sub_tour[tour1][node11:node12 + 1] + sub_tour[tour2][node22 + 1:]

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 8 and n > 0:  # ICROSS
        while True:
            while True:
                tour1 = random.randint(0, n - 1)
                tour2 = random.randint(tour1 + 1, n)
                if len(sub_tour[tour1]) >= 4 and len(sub_tour[tour2]) >= 4:
                    break

            node11 = random.randint(1, len(sub_tour[tour1]) - 3)
            node12 = random.randint(node11, len(sub_tour[tour1]) - 2)
            node21 = random.randint(1, len(sub_tour[tour2]) - 3)
            node22 = random.randint(node21, len(sub_tour[tour2]) - 2)

            new_tour1 = (sub_tour[tour1][:node11] + sub_tour[tour2][node21:node22 + 1][::-1] +
                         sub_tour[tour1][node12 + 1:])
            new_tour2 = (sub_tour[tour2][:node21] + sub_tour[tour1][node11:node12 + 1][::-1] +
                         sub_tour[tour2][node22 + 1:])

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 9 and n > 0:  # GENI
        while True:
            while True:
                tour1 = random.randint(0, n)
                tour2 = random.randint(0, n)
                if len(sub_tour[tour2]) >= 4 and tour1 != tour2:
                    break

            node1 = random.randint(1, len(sub_tour[tour1]) - 2)
            node21 = random.randint(0, len(sub_tour[tour2]) - 4)
            node22 = random.randint(node21 + 2, len(sub_tour[tour2]) - 2)

            new_tour1 = sub_tour[tour1][:node1] + sub_tour[tour1][node1 + 1:]
            new_tour2 = (sub_tour[tour2][:node21 + 1] + [sub_tour[tour1][node1]] + [sub_tour[tour2][node22]] +
                         sub_tour[tour2][node21 + 1:node22] + sub_tour[tour2][node22 + 1:])

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    elif neighbor_type == 10 and n > 0:  # λ-interchange
        while True:
            while True:
                tour1 = random.randint(0, n - 1)
                tour2 = random.randint(tour1 + 1, n)
                if len(sub_tour[tour1]) >= 4 and len(sub_tour[tour2]) >= 4:
                    break

            node11 = random.randint(1, len(sub_tour[tour1]) - 3)
            node12 = random.randint(node11, len(sub_tour[tour1]) - 2)
            node21 = random.randint(1, len(sub_tour[tour2]) - 3)
            node22 = random.randint(node21, len(sub_tour[tour2]) - 2)

            new_tour1 = sub_tour[tour1][:node11] + sub_tour[tour2][node21:node22 + 1] + sub_tour[tour1][node12 + 1:]
            new_tour2 = sub_tour[tour2][:node21] + sub_tour[tour1][node11:node12 + 1] + sub_tour[tour2][node22 + 1:]

            if (check_time(new_tour1, travel_time, service_time, ready_time, due_time) and
                    check_time(new_tour2, travel_time, service_time, ready_time, due_time) and
                    sum(demand[new_tour1]) <= capacity and sum(demand[new_tour2]) <= capacity):
                sub_tour[tour1] = new_tour1
                sub_tour[tour2] = new_tour2
                break
            elif time.time() - shaking_start > 0.5:
                break

    return sub_tour
