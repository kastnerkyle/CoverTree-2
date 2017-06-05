#!/usr/bin/env python
#
# File: test_covertree.py
# Date of creation: 11/20/08
# Copyright (c) 2007, Thomas Kollar <tkollar@csail.mit.edu>
# Copyright (c) 2011, Nil Geisweiller <ngeiswei@gmail.com>
# All rights reserved.
#
# This is a tester for the cover tree nearest neighbor algorithm.  For
# more information please refer to the technical report entitled "Fast
# Nearest Neighbors" by Thomas Kollar or to "Cover Trees for Nearest
# Neighbor" by John Langford, Sham Kakade and Alina Beygelzimer
#  
# If you use this code in your research, kindly refer to the technical
# report.

from covertree import CoverTree
from naiveknn import knn

# from pylab import plot, show
from numpy import subtract, dot, sqrt
from random import random, seed
import time

import operator
from scipy.spatial.distance import cdist, euclidean
import numpy as np


def distance(p, q):
    # print "distance"
    # print "p =", p
    # print "q =", q
    x = subtract(p, q)
    return sqrt(dot(x, x))

def test_covertree():
    seed(1)

    total_tests = 0
    passed_tests = 0
    
    n_points = 400

    k = 5
    
    pts = [(random(), random()) for _ in range(n_points)]

    gt = time.time

    print("Build cover tree of", n_points, "2D-points")
    
    t = gt()
    ct = CoverTree(distance)
    for p in pts:
        ct.insert(p)
    b_t = gt() - t
    print("Building time:", b_t, "seconds")

    print("==== Check that all cover tree invariants are satisfied ====")
    assert ct._check_invariants()
    
    print("==== Write test1.dot, dotty file of the built tree ====")
    with open("test1.dot", "w") as testDottyFile1:
        ct.writeDotty(testDottyFile1)

    '''
    print "==== Test saving/loading (via pickle)"
    ctFileName = "test.ct"
    print "Save cover tree under", ctFileName
    t = gt()
    ct_file = open("test.ct", "w")
    pickle.dump(ct, ct_file)
    ct_file.close()
    print "Saving time:", gt() - t
    del ct_file
    del ct
    # load ct
    print "Reload", ctFileName
    t = gt()
    ct_file = open("test.ct")
    ct = pickle.load(ct_file)
    ct_file.close()
    print "Loading time:", gt() - t
    '''
    
    print("==== Test " + str(k) + "-nearest neighbors cover tree query ====")
    query = (0.5,0.5)

    # naive nearest neighbor
    t = gt()
    naive_results = knn(k, query, pts, distance)
    # print "resultNN =", resultNN
    n_t = gt() - t
    print("Time to run a naive " + str(k) + "-nn query:", n_t, "seconds")

    # cover-tree nearest neighbor
    t = gt()
    results = ct.knn(query, k)
    # print "result =", result
    ct_t = gt() - t
    print("Time to run a cover tree " + str(k) + "-nn query:", ct_t, "seconds")
    results = list(map(operator.itemgetter(1), results))
    
    assert all([distance(r, nr) == 0 for r, nr in zip(results, naive_results)])
    print("Cover tree query is", n_t/ct_t, "faster")


    # you need pylab for that
    # plot(pts[0], pts[1], 'rx')
    # plot([query[0]], [query[1]], 'go')
    # plot([naive_results[0][0]], [naive_results[0][1]], 'y^')
    # plot([results[0][0]], [results[0][1]], 'mo')

    print("==== Write test2.dot, dotty file of the built tree after knn_insert ====")
    with open("test2.dot", "w") as testDottyFile2:
        ct.writeDotty(testDottyFile2)
        
    # printDotty prints the tree that was generated in dotty format,
    # for more info on the format, see http://graphviz.org/
    # ct.printDotty()

    # show()

    print(passed_tests, "tests out of", total_tests, "have passed")
    
def test_neighbors():
    N = 1000

    np.random.seed(42)
    data = np.random.random((N,1))
    # data = np.array([0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9])
    # N = len(data)
    # data = data.reshape((N, 1))

    T = CoverTree(distance)
    for p in data:
        T.insert(p)

    subset = data[np.random.choice(np.arange(len(data)), size=1)]
    # subset = np.array([5]).reshape((1,1))
    realdist = cdist(subset, data)

    # import pdb;pdb.set_trace()
    for i, p in enumerate(subset):
        r = 0.01
        result = T.neighbors(p, r)
        ix,ns,ds = zip(*result)
        ns = np.array(ns)
        ds = np.array(ds)
        real = realdist[i][realdist[i] <= r]

        assert (data[np.array(ix)] == ns).all()
        assert ds.max() <= r
        assert len(ds) == len(real)
        assert sorted(ds) == sorted(real)

        # print 'N({}, {}) =\n{}'.format(p, r, ns.reshape((len(ns),)))
        # print 'D ='
        # print np.array(sorted(ds))

    print('Test Neighborhood query: OK')

def test_neighbors_of_empty_tree():
    T = CoverTree(distance)
    nn = T.neighbors(1, 1)
    assert len(nn) == 0


def test_knn():
    np.random.seed(42)
    n = 10
    k = 1
    data = np.arange(n)
    T = CoverTree(euclidean, data)

    for p in data:
        nns = list(T.knn(p, 1))
        assert len(nns) == 1
        idx, nn, d = nns[0]
        print(p, nn, d)
        assert nn == p
        assert d == 0


def test_contains():
    N = 100

    np.random.seed(42)
    data = np.random.random((N,1))
    T = CoverTree(distance)
    for p in data: T.insert(p)

    points = 10 + np.random.random((int(N*0.1), 1))
    for p in points:
        assert not T.contains(p)

    for p in np.random.random((int(N*0.1), 1)):
        assert not T.contains(p)

def test_traverse():
    N = 100

    np.random.seed(42)
    data = np.random.random((N,1))
    T = CoverTree(distance)
    for p in data: T.insert(p)

    for i, p in T:
        assert data[i] == p

def test_extend():
    N = 100
    np.random.seed(42)
    D0 = np.random.random((N,1))
    D1 = np.random.random((N,1))
    T0 = CoverTree(distance, data=D0)
    T1 = CoverTree(distance, data=D1)

    T0.extend(T1)
    for i, p in T0:
        if i < N:
            D = D0
        else:
            D = D1
        assert p in D

def test_stable_indexing():
    N = 50
    np.random.seed(42)
    D0 = np.random.random((N, 1))
    D1 = np.random.random((N, 1))

    T = CoverTree(distance, data=D0)
    T.extend(D1)

    for i, p in T:
        if i < N: D = D0
        else:     D = D1
        assert D[i%N] == p

if __name__ == '__main__':
    test_covertree()
    test_neighbors()
    test_neighbors_of_empty_tree()
    test_contains()
    test_traverse()
    test_extend()
    test_stable_indexing()
