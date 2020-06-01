from graph import DirectedMultiGraph
from util import parse_info, randint_excluding
from model import MNLogit
from multiprocessing import Pool
import numpy as np
import random, os, time
from itertools import product

PATH = 'Mixed-MNL-graph'

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
        candidates, lnsw = G.neg_samp_by_locality(actor, target, num_neg=s, max_num_local_sample=[s//4,s//4])
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

      features.append(feature.T)
      lnsws.append(lnsw)

    G.add_edge(actor, target)

  return np.array(features), np.array(lnsws)

def experiment_1cl(kwargs):
  X, lnsw = extract_feature(**kwargs)
  y = np.zeros(X.shape[0]).astype(int)
  m = MNLogit()
  m.data(X[..., :-1], y, sws=lnsw)
  m.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
  return m.get_model_info()


def experiment_2cl(kwargs):
  X, lnsw = extract_feature(**kwargs)
  y = np.zeros(X.shape[0]).astype(int)

  ffof = X[:,0,8]
  ind_1 = ffof > 0.5 # local (ffof)
  ind_2 = ffof < 0.5 # non-local

  m1 = MNLogit()
  sws_filter_1 = X[ind_1,:,8] < 0.5
  m1.data(X[ind_1,:,:-1], y[ind_1], sws=lnsw[ind_1] - 10000 * sws_filter_1)
  m1.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
  info1 = m1.get_model_info()

  m2 = MNLogit()
  sws_filter_2 = X[ind_2,:,8] > 0.5
  m2.data(X[ind_2][...,[0,4]], y[ind_2], sws=lnsw[ind_2] - 10000 * sws_filter_2)
  m2.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
  info2 = m2.get_model_info()

  merged_info = {k+"_1":info1[k] for k in info1}
  merged_info.update({k+"_2":info2[k] for k in info1})
  return merged_info

def write_csv(filename, kwargs_list, result):
  with open(filename, 'w') as f:
    header = ['num_nodes','graph_id','sampling','data_size','num_neg','i']
    header.extend(['w_{}'.format(i) for i in range(8)])
    header.extend(['se_{}'.format(i) for i in range(8)])
    f.write(','.join(header) + '\n')
    for info, kwargs in zip(result, kwargs_list):
      row = [5000, kwargs['graph_id'], kwargs['sampling'], kwargs['n'], kwargs['s'], kwargs['i']]
      row.extend(info['weights'])
      row.extend(info['se'])
      f.write(','.join(map(str, row)) + '\n')

def fig_4a_4b(dest='synthetic-ml-on-1cl.csv'):
  result = None
  kwargs_list = [{'n':80000, 's':s, 'sampling':sampling, 'i':i, 'graph_id':parse_info(PATH)['graph_seed'], \
                  'feature_seed': random.randint(0, 2**31-1),'edge_sampling':'first-n'} \
                  for s,sampling,i in product([16,32,64,128,256,512,1024], ['uniform','stratified'], range(20))]
  with Pool(48) as p:
    result = p.map(experiment_1cl, kwargs_list)
  write_csv(dest, kwargs_list, result)


def fig_4c_4d(dest='synthetic-ml-on-2cl-big-n-small-s-truncated-feature.csv'):
  result = None
  kwargs_list = [{'n':80000, 's':s, 'sampling':sampling, 'i':i, 'graph_id':parse_info(PATH)['graph_seed'], \
                  'feature_seed': random.randint(0, 2**31-1),'edge_sampling':'first-n'} \
                  for s,sampling,i in product([16,32,64,128,256,512,1024], ['uniform','stratified'], range(20))]
  with Pool(48) as p:
    result = p.map(experiment_2cl, kwargs_list)

  with open(dest, 'w') as f:
    header = ['num_nodes','graph_id','sampling','data_size','num_neg','i']
    header.extend(['w_local_{}'.format(i) for i in range(8)])
    header.extend(['se_local_{}'.format(i) for i in range(8)])
    header.extend(['w_non_local_{}'.format(i) for i in range(2)])
    header.extend(['se_non_local_{}'.format(i) for i in range(2)])
    f.write(','.join(header) + '\n')
    for info, kwargs in zip(result, kwargs_list):
      row = [5000, kwargs['graph_id'], kwargs['sampling'], kwargs['n'], kwargs['s'], kwargs['i']]
      row.extend(info['weights_1'])
      row.extend(info['se_1'])
      row.extend(info['weights_2'])
      row.extend(info['se_2'])
      f.write(','.join(map(str, row)) + '\n')

fig_4a_4b()
fig_4c_4d()
