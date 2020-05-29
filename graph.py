import os
import sys, time, string, random
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from itertools import chain, repeat, product
from util import randint_excluding

class DirectedMultiGraph(object):
  '''
  Directed graph instance with fixed number of nodes that preserves the edge
  formations in chronological order with adjacency stored
  both as two lists of Counter (to and fron) and as edges list.
  '''

  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    self.adjacency_to = [Counter() for _ in range(num_nodes)]
    self.adjacency_from = [Counter() for _ in range(num_nodes)]
    self.in_degs = np.zeros(num_nodes)
    self.out_degs = np.zeros(num_nodes)
    self.edges_list = []

  def add_edge(self, actor, target):
    self.out_degs[actor] += 1
    self.in_degs[target] += 1
    self.adjacency_to[actor][target] += 1
    self.adjacency_from[target][actor] += 1
    self.edges_list.append((actor, target))

  def grow_erdos_renyi(self, num_edges):
    '''
    Grow an Erdos-Renyi G(n,m) graph, where n is fixed to self.num_nodes and m
    is num_edges.
    '''
    actors = np.random.randint(0, self.num_nodes, size=num_edges)
    targets = (np.random.randint(1, self.num_nodes, size=num_edges) + actors) % self.num_nodes
    for actor, target in zip(actors, targets):
      self.add_edge(actor, target)

  def neg_samp_by_locality(self, actor, target, num_neg=24, max_num_local_sample=[8,8]):
    hop1, hop2 = set(), set()
    for n in chain(self.adjacency_to[actor], self.adjacency_from[actor]):
      if n in hop1:
        continue
      for nn in chain(self.adjacency_to[n], self.adjacency_from[n]):
        hop2.add(nn)
      hop1.add(n)
    for n in hop1:
      hop2.discard(n)
    hop1.discard(actor)
    hop2.discard(actor)
    result, lnsw = [target], []
    n1, s1 = len(hop1), min(max_num_local_sample[0], len(hop1))
    n2, s2 = len(hop2), min(max_num_local_sample[1], len(hop2))
    n3, s3 = self.num_nodes - 1 - s1 - s2, num_neg + 1 - s1 - s2
    target_group = 3
    if target in hop1:
      lnsw.append(np.log(n1/s1))
      hop1.remove(target)
      target_group = 1
    elif target in hop2:
      lnsw.append(np.log(n2/s2))
      hop2.remove(target)
      target_group = 2
    else:
      lnsw.append(np.log(n3/s3))
    if hop1:
      result.extend(np.random.choice(list(hop1), replace=False, size=s1-int(target_group==1)))
      lnsw.extend(repeat(np.log(n1/s1), s1-int(target_group==1)))
    if hop2:
      result.extend(np.random.choice(list(hop2), replace=False, size=s2-int(target_group==2)))
      lnsw.extend(repeat(np.log(n2/s2), s2-int(target_group==2)))
    if s3 > 0:
      result.extend(randint_excluding(0, self.num_nodes, list(chain(hop1, hop2, [actor, target])),
                                      size=s3-int(target_group==3)))
      lnsw.extend(repeat(np.log(n3/s3), s3-int(target_group==3)))

    return result, lnsw

  def extract_feature(self, actor, candidates):
    num_i_to_to_j_ctr = Counter()
    is_fof_j_ctr = Counter()

    for n in self.adjacency_to[actor]:
      for nn in self.adjacency_to[n]:
        num_i_to_to_j_ctr[nn] += 1

    for n in chain(self.adjacency_to[actor], self.adjacency_from[actor]):
      for nn in chain(self.adjacency_to[n], self.adjacency_from[n]):
        is_fof_j_ctr[nn] = 1

    in_degs = np.array([self.in_degs[c] for c in candidates])
    num_i_to_j = np.array([self.adjacency_to[actor][c] for c in candidates])
    num_i_from_j = np.array([self.adjacency_from[actor][c] for c in candidates])
    num_i_to_to_j = np.array([num_i_to_to_j_ctr[c] for c in candidates])
    is_fof_j = np.array([is_fof_j_ctr[c] for c in candidates])

    log_in_degrees = np.log(in_degs + (in_degs < 0.5).astype(int))
    log_i_to_j = np.log(num_i_to_j + (num_i_to_j < 0.5).astype(int))
    log_i_from_j = np.log(num_i_from_j + (num_i_from_j < 0.5).astype(int))
    log_i_to_to_j = np.log(num_i_to_to_j + (num_i_to_to_j < 0.5).astype(int))

    return np.array([log_in_degrees, log_i_to_j, log_i_from_j, log_i_to_to_j,\
                     (in_degs > 0.5).astype(int),      (num_i_to_j > 0.5).astype(int),\
                     (num_i_from_j > 0.5).astype(int), (num_i_to_to_j > 0.5).astype(int),\
                     is_fof_j.astype(int)])

  def extract_feature_all_nodes(self, actor):
    num_i_to_j = np.zeros(self.num_nodes)
    num_i_from_j = np.zeros(self.num_nodes)
    num_i_to_to_j = np.zeros(self.num_nodes)
    is_fof_j = np.zeros(self.num_nodes)

    for n in self.adjacency_to[actor]:
      num_i_to_j[n] = self.adjacency_to[actor][n]
    for n in self.adjacency_from[actor]:
      num_i_from_j[n] = self.adjacency_from[actor][n]

    for n in self.adjacency_to[actor]:
      for nn in self.adjacency_to[n]:
        num_i_to_to_j[nn] += 1

    for n in chain(self.adjacency_to[actor], self.adjacency_from[actor]):
      for nn in chain(self.adjacency_to[n], self.adjacency_from[n]):
        is_fof_j[nn] = 1

    log_in_degrees = np.log(self.in_degs + (self.in_degs < 0.5).astype(int))
    log_i_to_j = np.log(num_i_to_j + (num_i_to_j < 0.5).astype(int))
    log_i_from_j = np.log(num_i_from_j + (num_i_from_j < 0.5).astype(int))
    log_i_to_to_j = np.log(num_i_to_to_j + (num_i_to_to_j < 0.5).astype(int))

    return np.array([log_in_degrees, log_i_to_j, log_i_from_j, log_i_to_to_j,\
                    (self.in_degs > 0.5).astype(int), (num_i_to_j > 0.5).astype(int),\
                    (num_i_from_j > 0.5).astype(int), (num_i_to_to_j > 0.5).astype(int),\
                    is_fof_j.astype(int)]) # <---- used as a flag for generating mixed model
