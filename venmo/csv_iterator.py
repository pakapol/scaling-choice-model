import numpy as np

def csv_iterator(filename, sep=',', num_rows=1):
  with open(filename, 'r') as f:
    result = []
    for line in f:
      result.append([int(x) for x in line.split(sep)])
      if len(result) == num_rows:
        yield np.array(result)
        result = []
    if len(result) > 0:
      yield np.array(result)
