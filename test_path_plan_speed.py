# Testing the 3 path planning algorithms

import pickle
import time

from dijkstra_planner import dij_planner
from a_star_heur1 import a_star1_planner
from a_star_heur2 import a_star2_planner

def unpickler(filename, num):
    with open(f"{filename}{num}.pkl", 'rb') as p:
        return pickle.load(p)
    
def normal_pkl(num):
    return unpickler("normalized", num)

def unnormal_pkl(num):
    return unpickler("unnormalized", num)
    
def test_paths():
    un2 = unnormal_pkl(2)
    un3 = unnormal_pkl(3)
    un4 = unnormal_pkl(4)
    un5 = unnormal_pkl(5)
    n2 = normal_pkl(2)
    n3 = normal_pkl(3)
    n4 = normal_pkl(4)
    n5 = normal_pkl(5)

    # start_dij2 = time.time()
    # for _ in range(10000):
    #     dij_planner(un2)
    # end_dij2 = time.time()

    # print("2")
    # print("DIJ time: ", func_time(dij_planner, un2))
    # print("A* unnormalize: ", func_time(a_star2_planner,un2))
    # print("A* normalize: ", func_time(a_star1_planner,n2))

    # print("3")
    # print("DIJ time: ", func_time(dij_planner, un3))
    # print("A* unnormalize: ", func_time(a_star2_planner,un3))
    # print("A* normalize: ", func_time(a_star1_planner,n3))

    # print("4")
    # print("DIJ time: ", func_time(dij_planner, un4))
    # print("A* unnormalize: ", func_time(a_star2_planner,un4))
    # print("A* normalize: ", func_time(a_star1_planner,n4))

    print("5")
    print("DIJ time: ", func_time(dij_planner, un5))
    print("A* unnormalize: ", func_time(a_star2_planner,un5))
    print("A* normalize: ", func_time(a_star1_planner,n5))

    # print(dij_planner(un3))
    # print(dij_planner(un4))
    # print(dij_planner(un5))

    # print("A* unnormalized")
    # print(a_star2_planner(un2))
    # print(a_star2_planner(un3))
    # print(a_star2_planner(un4))
    # print(a_star2_planner(un5))

    # print("A* normalized")
    # print(a_star1_planner(n2))
    # print(a_star1_planner(n3))
    # print(a_star1_planner(n4))
    # print(a_star1_planner(n5))

def func_time(func, data):
    start = time.time()
    for _ in range(100000):
        func(data)
    return time.time() - start

def main():
    test_paths()

if __name__ == "__main__":
    main()
