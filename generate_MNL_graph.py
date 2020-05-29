from graph import DirectedMultiGraph
from util import write_info
import numpy as np
import os, random, time

PATH = 'MNL-graph'

def synthesize_graph(**kwargs):
  t0 = time.time()
  num_nodes = kwargs['num_nodes'] # = 5000
  num_er_edges = kwargs['num_er_edges'] # = 25000
  num_choice_edges = kwargs['num_choice_edges'] # = 160000
  np.random.seed(kwargs['seed'])

  G = DirectedMultiGraph(num_nodes)
  G.grow_erdos_renyi(num_er_edges)

  # Generate multinomial choice model with the following weights
  w0 = np.array([0.5,2,2,2,1,4,4,4,0])
  for i in range(num_choice_edges):
    # Randomly pick an actor
    actor = np.random.randint(0, G.num_nodes)
    # MNL experiment
    X = G.extract_feature_all_nodes(actor)
    U = w0.dot(X) + np.random.gumbel(size=G.num_nodes)
    U[actor] = -np.inf
    target = np.argmax(U)
    G.add_edge(actor, target)

  kwargs['time_elapsed'] = time.time() - t0
  np.save(os.path.join(PATH, kwargs['er_edges_path']), G.edges_list[:num_er_edges])
  np.save(os.path.join(PATH, kwargs['choice_edges_path']), G.edges_list[num_er_edges:])
  write_info(PATH, kwargs)

kwargs = {'num_nodes':       5000,
         'num_er_edges':     25000,
         'num_choice_edges': 160000,
         'logging_interval': 1e3,
         'graph_seed': random.randint(0, 2**31-1),
         'er_edges_path': 'er_edges.npy',
         'choice_edges_path': 'choice_edges.npy'}

synthesize_graph(**kwargs)
