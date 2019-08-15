import json, os, sys
from multiprocessing import Pool

def criteria(q):
  return True

def itoa(l):
  return [str(x) for x in l]

def process(filename):
  print("Processing {}".format(filename))
  with open(filename) as f, open(os.path.join('temp',filename[-26:]), 'w') as g:
    for line in f:
      q = json.loads(line)
      if criteria(q):
        g.write(' '.join(itoa(q[:-1])) + '\n')

with Pool(48) as p:
  p.map(process, sorted(sys.argv[1:]))
        
