#!/usr/bin/env python3

# A* to determine shortest path - multiply edge weight by Euclidean distance

import heapq
import pickle
from collections import deque
import time

def a_star_heuristic(location, end, weight):
    """
    Heuristic to multiply edge weight by distance
    then don't need to normalize edge weights
    Returns Edge weight multiplied by Euclidean distance, using as function h in A*
    location : 1x6 tuple-pose
    end: 1x6 tuple-pose that is end point
    weight: edge weight for particular node
    """
    return (((location[0] - end[0])**2 + (location[1] - end[1])**2)**(1/2)) * weight

def a_star_map(map):
    """
    map: dictionary with graph
    {1x6 tuple-pose : {1x6 tuple-pose : integer-weight}}
    2 special keys: 'start', 'end'
    """

    g_score, f_score, prev = {}, {}, {}

    heap = []

    for key in map:

        # deals with special keys
        if key in {'start', 'end'}:
            continue

        if key not in g_score:
            g_score[key] = float('inf')
            f_score[key] = float('inf')
            prev[key] = -1

        for v in map[key]:
            if v not in g_score:
                f_score[key] = float('inf')
                g_score[v] = float('inf')
                prev[v] = -1

    # -1 in prev is a stand in for undefined

    start, end = map['start'], map['end']

    g_score[start] = 0
    f_score[start] = a_star_heuristic(start, end, g_score[start])
    prev[start] = 0

    heapq.heappush(heap, (f_score[start], start))

    while heap:
        # Only need weight in heap to give priority to easier paths 
        _, location = heapq.heappop(heap)

        if location == end:
            break
        
        if location not in map:
            continue

        for node in map[location]:
            node_weight = map[location][node]
            new_weight = g_score[location] + node_weight

            if new_weight < g_score[node]:
                prev[node] = location
                g_score[node] = new_weight
                f_score[node] = new_weight + a_star_heuristic(node, end, g_score[node])
                heapq.heappush(heap, (f_score[node], node))

    return (prev, start, end)

def a_star_path(paths, target, source):
    """
    returns the shortest path from end to start
    Because builds the graph from start to finish and terminates when
    it reaches the end, the path from start to end has to be built backwards

    paths : dictionary of poses 
    target : goal pose
    source : starting pose

    Returns list of list
    """

    # Only keep x, y, yaw from the pose
    path_to_end = deque([[target[0], target[1], target[-1]]])

    curr = target

    while curr != source:
        nxt = paths[curr]
        path_to_end.appendleft([nxt[0], nxt[1], nxt[-1]])
        curr = nxt

    return list(path_to_end)

def a_star2_planner(map):
    '''
    allows single function to be called from file to return shortest path

    map : weighted graph
    '''
    path, start, end = a_star_map(map)
    return a_star_path(path, end, start)