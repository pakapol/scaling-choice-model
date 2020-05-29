import random, os

def randint_excluding(low, high, excluding, size=1):
  if size == high - low - len(excluding):
    return [x for x in range(low, high) if x not in excluding]
  exclusion = {num for num in excluding if low <= num < high}
  result = []
  if size == 0:
    return result
  while len(result) < size:
    rand_size = 1 + int(((size - len(result)) * (high-low) / (high-low-len(exclusion)+1)) // 1)
    buf = random.sample(range(low, high), min(rand_size, high-low))
    for num in buf:
      if len(exclusion) >= high-low:
        raise RuntimeError("randint_excluding fails: Size too large; exhausted all integers available in the range before finishing.")
      if num not in exclusion:
        result.append(num)
        if len(result) == size:
          return result
        exclusion.add(num)
  return result

def write_info(PATH, info):
  with open(os.path.join(PATH, 'info'), 'w') as f:
    for k, v in info.items():
      f.write("{}\t{}\n".format(k, v))

def parse_info(PATH):
  result = {}
  with open(os.path.join(PATH, 'info')) as f:
    for line in f:
      k, v = line[:-1].split('\t')
      try:
        v = int(v)
      except:
        try:
          v = float(v)
        except:
          pass
      result[k] = v
  return result
