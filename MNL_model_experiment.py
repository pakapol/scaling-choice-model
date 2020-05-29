from graph import DirectedMultiGraph
from util import parse_info, randint_excluding
from model import MNLogit
from multiprocessing import Pool
import numpy as np
import random, os, time
from itertools import product

PATH = 'MNL-graph'

def extract_feature(**kwargs):

  n = kwargs['n']
  s = kwargs['s']
  seed = kwargs['feature_seed']
  sampling = kwargs['sampling']
  edge_sampling = kwargs['edge_sampling']

  np.random.seed(seed)

  info = parse_info(PATH)
  er_edges     = np.load(os.path.join(PATH, info['er_edges_path']),     mmap_mode='r')
  choice_edges = np.load(os.path.join(PATH, info['choice_edges_path']), mmap_mode='r')

  G = DirectedMultiGraph(info['num_nodes'])
  for actor, target in er_edges:
    G.add_edge(actor, target)

  features = []
  lnsws = []

  to_sample = set(range(n)) if edge_sampling=='first-n' else \
              set(random.sample(range(len(choice_edges)), n))

  n1 = n2 = n3 = 1

  for i, (actor, target) in enumerate(choice_edges):
    if i in to_sample:
      candidates, lnsw = None, None
      if sampling == 'stratified':
        candidates, lnsw = G.neg_samp_by_locality(actor, target, num_neg=s, max_num_local_sample=[s//3,s//3])
      elif sampling == 'importance':
        s1 = int(np.floor((s-3) * n1 / (n1 + n2 + n3)) + 1)
        s2 = int(np.floor((s-3) * n2 / (n1 + n2 + n3)) + 1)
        candidates, lnsw = G.neg_samp_by_locality(actor, target, num_neg=s, max_num_local_sample=[s1,s2])
      else:
        candidates = [target]
        candidates.extend(randint_excluding(0, G.num_nodes, [actor, target], size=s))
        lnsw = [0.0] * len(candidates)

      feature = G.extract_feature(actor, candidates)
      if feature[5][0] + feature[6][0] > 0.5:
        n1 += 1
      elif feature[8][0] > 0.5:
        n2 += 1
      else:
        n3 += 1

      features.append(feature[:-1].T)
      lnsws.append(lnsw)

    G.add_edge(actor, target)

  return np.array(features), np.array(lnsws)

def experiment(kwargs):
  X, lnsw = extract_feature(**kwargs)
  y = np.zeros(X.shape[0]).astype(int)
  m = MNLogit()
  m.data(X, y, sws=lnsw)
  m.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
  return m.get_model_info()

def write_csv(filename, kwargs_list, result):
  with open('synthetic-vary-n-and-s.csv','w') as f:
    header = ['num_nodes','graph_id','sampling','data_size','num_neg','i']
    header.extend(['w_{}'.format(i) for i in range(8)])
    header.extend(['se_{}'.format(i) for i in range(8)])
    f.write(','.join(header) + '\n')
    for info, kwargs in zip(result, kwargs_list):
      row = [5000, kwargs['graph_id'], kwargs['sampling'], kwargs['n'], kwargs['s'], kwargs['i']]
      row.extend(info['weights'])
      row.extend(info['se'])
      f.write(','.join(map(str, row)) + '\n')

def fig_3a_3b():
  result = None
  kwargs_list = [{'n':n, 's':s, 'sampling':sampling, 'i':i, 'graph_id':parse_info(PATH)['seed'], \
                  'feature_seed': random.randint(0, 2**31-1),'edge_sampling':'first-n'} \
                  for n,s,sampling,i in product([24,96], range(500,5001,500), ['uniform','stratified'], range(50))]
  with Pool(48) as p:
    result = p.map(experiment, kwargs_list)
  write_csv('synthetic-vary-n-and-s-very-small.csv', kwargs_list, result)

def fig_3c():
  result = None
  kwargs_list = [{'n':10000, 's':s, 'sampling':sampling, 'i':i, 'graph_id':parse_info(PATH)['seed'], \
                  'feature_seed': random.randint(0, 2**31-1),'edge_sampling':'first-n'} \
                  for s,sampling,i in product([3,6,12,24,48,96,192,384,768], ['uniform','stratified','importance'], range(50))]
  with Pool(48) as p:
    result = p.map(experiment, kwargs_list)
  write_csv('synthetic-fix-n-10k.csv',kwargs_list, result)

def fig_3d():
  result = None
  kwargs_list = [{'n':480000//s, 's':s, 'sampling':sampling, 'i':i, 'graph_id':parse_info(PATH)['seed'], \
                  'feature_seed': random.randint(0, 2**31-1),'edge_sampling':'random-uniform'} \
                   for s,sampling,i in product([3,6,12,24,48,96,192,384,768], ['uniform','stratified','importance'], range(50))]
  with Pool(48) as p:
    result = p.map(experiment, kwargs_list)
  write_csv('synthetic-fix-ns-480k-randedge.csv',kwargs_list, result)

fig_3a_3b()
fig_3c()
fig_3d()
