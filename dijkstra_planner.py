#!/usr/bin/env python3

# Dijkstra to determine shortest path

import heapq
import pickle # Used for testing
from collections import deque
import time # Used for testing

def dij_map(map):
    """
    Creates a dictionary of poses that terminates at the end pose and provides
    the shortest path to the start node
    The output is used in conjuction with a helper function to provide the 
    shortest path in list form

    map: dictionary with graph
    {1x6 tuple-pose : {1x6 tuple-pose : integer-weight}}
    2 special keys: 'start', 'end'

    Returns 1x3 tuple
    prev : dictionary of poses
    start : 1x6 tuple which gives the starting pose
    end : 1x6 tuple which gives the ending pose
    """
    
    dist, prev = {}, {}

    heap = []
    
    for key in map:

        # deals with special keys
        if key in {'start', 'end'}:
            continue

        if key not in dist:
            dist[key] = float('inf')
            prev[key] = -1

        for v in map[key]:
            if v not in dist:
                dist[v] = float('inf')
                prev[v] = -1

    # -1 in prev is a stand in for undefined

    start, end = map['start'], map['end']

    dist[start] = 0
    prev[start] = 0

    heapq.heappush(heap, (0, start))

    while heap:
        # Only need weight in heap to give priority to easier paths 
        _, location = heapq.heappop(heap)

        if location == end:
            break
        
        if location not in map:
            continue

        for node in map[location]:
            node_weight = map[location][node]
            new_weight = dist[location] + node_weight

            if new_weight < dist[node]:
                prev[node] = location
                dist[node] = new_weight
                heapq.heappush(heap, (new_weight, node))

    return (prev, start, end)

def dij_path(paths, target, source):
    """
    returns the shortest path from end to start
    Because Dijkstra builds the graph from start to finish and terminates when
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

def dij_planner(map):
    '''
    allows single function to be called from file to return shortest path

    map : weighted graph
    '''

    path, start, end = dij_map(map)
    return dij_path(path, end, start)