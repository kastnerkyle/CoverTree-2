# File: covertree.py
# Date of creation: 05/04/07
# Copyright (c) 2007, Thomas Kollar <tkollar@csail.mit.edu>
# Copyright (c) 2011, Nil Geisweiller <ngeiswei@gmail.com>
# All rights reserved.
#
# This is a class for the cover tree nearest neighbor algorithm.  For
# more information please refer to the technical report entitled "Fast
# Nearest Neighbors" by Thomas Kollar or to "Cover Trees for Nearest
# Neighbor" by John Langford, Sham Kakade and Alina Beygelzimer
#  
# If you use this code in your research, kindly refer to the technical
# report.

import numpy as np
import operator
import itertools
from random import choice
from heapq import nsmallest
from itertools import product
try:
    from collections import Counter
except ImportError: # Counter is not available in Python before v2.7
    from recipe_576611_1 import Counter
try:
    from joblib import Parallel, delayed
except ImportError:
    pass
import cStringIO

# method that returns true iff only one element of the container is True
def unique(container):
    return Counter(container).get(True, 0) == 1


# the Node representation of the data
class Node:
    # data is an array of values
    def __init__(self, data=None, idx=None):
        self.data = data
        self.children = {}      # dict mapping level and children
        self.parent = None
        self.idx = idx

    # addChild adds a child to a particular Node and a given level i
    def addChild(self, child, i):
        try:
            # in case i is not in self.children yet
            if(child not in self.children[i]):
                self.children[i].append(child)
        except(KeyError):
            self.children[i] = [child]
        child.parent = self

    # getChildren gets the children of a Node at a particular level
    def getChildren(self, level):
        retLst = [self]
        try:
            retLst.extend(self.children[level])
        except(KeyError):
            pass
        
        return retLst

    # like getChildren but does not return the parent
    def getOnlyChildren(self, level):
        try:
            return self.children[level]
        except(KeyError):
            pass
        
        return []


    def removeConnections(self, level):
        if(self.parent != None):
            self.parent.children[level+1].remove(self)
            self.parent = None

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)

class CoverTree:
    
    #
    # Overview: initalization method
    #
    # Input: distance function, root, maxlevel, minlevel, base, and
    #  for parallel support jobs and min_len_parallel. Here root is a
    #  point, maxlevel is the largest number that we care about
    #  (e.g. base^(maxlevel) should be our maximum number), just as
    #  base^(minlevel) should be the minimum distance between Nodes.
    #
    #  In case parallel is enbaled (jobs > 1), min_len_parallel is the
    #  minimum number of elements at a given level to have their
    #  distances to the element to insert or query evaluated.
    #
    def __init__(self, distance, root = None, maxlevel = 10, base = 2,
                 jobs = 1, min_len_parallel = 100):
        self.distance = distance
        self.root = root
        self.maxlevel = maxlevel
        self.minlevel = maxlevel # the minlevel will adjust automatically
        self.idx = 0
        self.base = base
        self.jobs = jobs
        self.min_len_parallel = min_len_parallel
        # for printDotty
        self.__printHash__ = set()


    @property
    def size(self):
        "Number of elements in the tree"
        return self.idx


    #
    # Overview: insert an element p into the tree
    #
    # Input: p
    # Output: nothing
    #
    def insert(self, p):
        if self.root == None:
            self.root = self._newNode(p)
        else:
            self._insert_iter(p)

    def _newNode(self, *args, **kws):
        kws['idx'] = self.idx
        self.idx += 1
        return Node(*args, **kws)

    #
    # Overview:insert an element p in to the cover tree
    #
    # Input: point p
    #
    # Output: nothing
    #
    def _insert_iter(self, p):
        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        i = self.maxlevel
        while True:
            # get the children of the current level
            # and the distance of the all children
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            d_p_Q = self._min_ds_(Q_p_ds)

            if d_p_Q == 0.0:    # already there, no need to insert
                return
            elif d_p_Q > self.base**i: # the found parent should be right
                break
            else: # d_p_Q <= self.base**i, keep iterating

                # find parent
                if self._min_ds_(Qi_p_ds) <= self.base**i:
                    parent = choice([q for q, d in Qi_p_ds if d <= self.base**i])
                    pi = i
                
                # construct Q_i-1
                Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= self.base**i]
                i -= 1

        # insert p
        parent.addChild(self._newNode(p), pi)
        # update self.minlevel
        self.minlevel = min(self.minlevel, pi-1)


    def neighbors(self, point, radius):
        """
        Overview: get the neighrbors of `p` within distance `r`

        Input:
         - point :: a point
         - radius :: float - the maximum (inclusive) distance
        Output:
         - [(i, n, d)] :: list of pairs (`index`, `point`, `float`) which are the point and it's distance to `p`
        """

        def containsPoint(point, radius, node, level, dist=None):
            if dist is None:
                dist = self.distance(point, node.data)
            # print level, point, dist, radius, radius + self.base**level
            return dist <= radius + self.base**level


        result = set()
        queue = [(self.maxlevel, self.root, self.distance(point, self.root.data))]

        while queue:
            level, node, dist = queue.pop(0)

            if not containsPoint(point, radius, node, level, dist=dist):
                continue

            if dist <= radius:
                result.add((node, dist))

            next_level = level-1
            if next_level < self.minlevel: continue

            for child in node.getChildren(next_level):
                if not child == node:
                    d = self.distance(point, child.data)
                else:
                    d = dist
                queue.append((next_level, child, d))


        return itertools.imap(lambda (child, dist): (child.idx, child.data, dist), result)

    #
    # Overview: get the children of cover set Qi at level i and the
    # distances of them with point p
    #
    # Input: point p to compare the distance with Qi's children, and
    # Qi_p_ds the distances of all points in Qi with p
    #
    # Output: the children of Qi and the distances of them with point
    # p
    #
    def _getChildrenDist_(self, p, Qi_p_ds, i):
        Q = sum([n.getOnlyChildren(i) for n, _ in Qi_p_ds], [])
        if 'Parallel' in dir() and self.jobs > 1 and len(Q) >= self.min_len_parallel:
            df = self.distance
            ds = Parallel(n_jobs = self.jobs)(delayed(df)(p, q.data) for q in Q)
            Q_p_ds = zip(Q, ds)
        else:
            Q_p_ds = [(q, self.distance(p, q.data)) for q in Q]
        
        return Qi_p_ds + Q_p_ds

    #
    # Overview: get a list of pairs <point, distance> with the k-min distances
    #
    # Input: Input cover set Q, distances of all nodes of Q to some point
    # Output: list of pairs 
    #
    def _kmin_p_ds_(self, k, Q_p_ds):
        return nsmallest(k, Q_p_ds, lambda x: x[1])

    # return the minimum distance of Q_p_ds
    def _min_ds_(self, Q_p_ds):
        return self._kmin_p_ds_(1, Q_p_ds)[0][1]

    # format the final result. If without_distance is True then it
    # returns only a list of data points, other it return a list of
    # pairs <point.data, distance>
    def _result_(self, res, without_distance):
        if without_distance:
            return [p.data for p, _ in res]
        else:
            return [(p.data, d) for p, d in res]
    
    #
    # Overview: write to a file the dot representation
    #
    # Input: None
    # Output: 
    #
    def writeDotty(self, outputFile):
        outputFile.write("digraph {\n")
        self._writeDotty_rec(outputFile, [self.root], self.maxlevel)
        outputFile.write("}")


    #
    # Overview:recursively build printHash (helper function for writeDotty)
    #
    # Input: C, i is the level
    #
    def _writeDotty_rec(self, outputFile, C, i):
        if(i == self.minlevel):
            return

        children = []
        for p in C:
            childs = p.getChildren(i)

            for q in childs:
                outputFile.write("\"lev:" +str(i) + " "
                                 + str(p.data) + "\"->\"lev:"
                                 + str(i-1) + " "
                                 + str(q.data) + "\"\n")

            children.extend(childs)
        
        self._writeDotty_rec(outputFile, children, i-1)

    def __str__(self):
        output = cStringIO.StringIO()
        self.writeDotty(output)
        return output.getvalue()


    # check if the tree satisfies all invariants
    def _check_invariants(self):
        return self._check_nesting() and \
            self._check_covering_tree() and \
            self._check_seperation()


    # check if my_invariant is satisfied:
    # C_i denotes the set of nodes at level i
    # for all i, my_invariant(C_i, C_{i-1})
    def _check_my_invariant(self, my_invariant):
        C = [self.root]
        for i in reversed(xrange(self.minlevel, self.maxlevel + 1)):        
            C_next = sum([p.getChildren(i) for p in C], [])
            if not my_invariant(C, C_next, i):
                print "At level", i, "the invariant", my_invariant, "is false"
                return False
            C = C_next
        return True
        
    
    # check if the invariant nesting is satisfied:
    # C_i is a subset of C_{i-1}
    def _nesting(self, C, C_next, _):
        return set(C) <= set(C_next)

    def _check_nesting(self):
        return self._check_my_invariant(self._nesting)
        
    
    # check if the invariant covering tree is satisfied
    # for all p in C_{i-1} there exists a q in C_i so that
    # d(p, q) <= base^i and exactly one such q is a parent of p
    def _covering_tree(self, C, C_next, i):
        return all(unique(self.distance(p.data, q.data) <= self.base**i
                          and p in q.getChildren(i)
                          for q in C)
                   for p in C_next)

    def _check_covering_tree(self):
        return self._check_my_invariant(self._covering_tree)

    # check if the invariant seperation is satisfied
    # for all p, q in C_i, d(p, q) > base^i
    def _seperation(self, C, _, i):
        return all(self.distance(p.data, q.data) > self.base**i
                   for p, q in product(C, C) if p != q)

    def _check_seperation(self):
        return self._check_my_invariant(self._seperation)
