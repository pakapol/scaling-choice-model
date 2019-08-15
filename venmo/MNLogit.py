import numpy as np
import torch
import time

class MNLogit(object):

  def __init__(self):
    torch.set_num_threads(48)

  def data(self, data_iter, data_iter_wrapper):
    self.data_iter = data_iter
    self.data_iter_wrapper = data_iter_wrapper

  def fit(self, max_num_iter=30, clip=1, clip_norm_ord=2, itol=1e-7, verbose=True, show_function=None):
    '''
    Fit multinomial logit choice model with the dataset

    Optimize parameters:
    - max_num_iter: The maximal number of iteration
    - batch_size: Maximal number of entries in each step (for limited memory)
    - thresh: threshold -- stop when gradient norm is smaller than thresh
    - clip: clipping the step size so it has the norm <= clip
    - clip_norm_ord: the order of the norm used for clipping step size
    - verbose: Print the process
    '''
    t0 = time.time()
    
    for i in range(max_num_iter):
      
      info = self._step(clip, clip_norm_ord)
      if show_function is not None:
        show_function(self.w.numpy(), info['se'])
      if verbose:
        print(info)
      print("---------- End of Iteration {} ----------".format(i+1))
      print("Time Elapsed = {} secs".format(time.time() - t0))
      
      
      if info['inc_norm'] < itol:
        return
    
  def data(self, Xs, ys, sws=None):
    '''
    Ingest data into the model

    Dataset:
    - Xs: An np.ndarray of shape (N * C * D) -- The feature tensor
      N entries, C candidates in each entry, D features for each candidate.
    - ys: An np.ndarray of shape (N) -- The vector of labels
      One label for each entry. The label is the index of the selected candidate
    - ws: An np.ndarray of shape (N * C) -- The log stratified sampling weight
      N entries, C candidates in each entry, 1 weight for each candidate.
      Can be None
    '''
    
    self.data_len = Xs.shape[0]
    self.num_classes = Xs.shape[1]
    self.num_features = Xs.shape[2]
    ys_oh = np.zeros((self.data_len, self.num_classes))
    ys_oh[np.arange(self.data_len), ys] += 1
    self.X = torch.Tensor(Xs).to(dtype=torch.double)
    self.y = torch.Tensor(ys).to(dtype=torch.long)
    self.y_oh = torch.Tensor(ys_oh).to(dtype=torch.double)
    self.sw = torch.Tensor(sws).to(dtype=torch.double)
    self.w = torch.Tensor(np.zeros(self.num_features)).to(dtype=torch.double)
    self.B = 1e-3 * torch.Tensor(np.eye(self.num_features)).to(dtype=torch.double)

  def eval(self, Xs, ys):
    '''
    Evaluate the dataset by classification accuracy
    '''

    result = (ys == np.argmax(Xs.dot(self.w.numpy()).reshape(-1, self.num_classes), axis=1))
    print("Accuracy: {} / {} ({:.4f})".format(np.sum(result), len(ys), np.sum(result)/len(ys)))
    
  def _loss(self, w):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, w) + self.sw)
    return -torch.sum(score[torch.arange(self.data_len, dtype=torch.long), self.y])
    
  def _grad(self, w):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, w) + self.sw)
    dscore = - self.y_oh + torch.exp(score) 
    return torch.mm(dscore.view(1,-1), self.X.view(-1, self.num_features)).view(-1) ## 1 x N*C dot N*C x D
    
    
  def _line_search(self, p, loss, dw, max_iter=40):
    c1, c2 = 0.001, 0.9
    a = 0
    t = 1
    t0 = np.linalg.norm(p.numpy(), ord=2) / np.sqrt(self.num_features)
    b = None
    for i in range(max_iter):
      if self._loss(self.w + (t/t0) * p).numpy() > (loss + c1 * (t/t0) * p.dot(dw)).numpy():
        b = t
        t = (a + b)/2
        print(t)
      elif p.dot(self._grad(self.w + (t/t0) * p)).numpy() < c2 * p.dot(dw).numpy():
        a = t
        t = 2 * a if b is None else (a + b)/2
        print(t)
      else:
         return t
    return t
  
  def _step(self, clip=1, clip_norm_ord=2):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, self.w) + self.sw)
    loss = -torch.sum(score[torch.arange(self.data_len, dtype=torch.long), self.y])
    dscore = - self.y_oh + torch.exp(score) 
    dw = torch.mm(dscore.view(1,-1), self.X.view(-1, self.num_features)).view(-1) ## 1 x N*C dot N*C x D
    p = -torch.mv(self.B, dw)
    pnorm = torch.norm(p, p=clip_norm_ord)
    if pnorm.numpy() > clip:
      # t = self._line_search(p, loss, dw)
      s = p * clip / pnorm
    else:
      s = p
    self.w += s
    y = self._grad(self.w + s) - dw
    sy = s.dot(y)
    self.B += ((sy + y.dot(torch.mv(self.B,y))) / (sy * sy)) * torch.ger(s,s) - \
              (torch.ger(torch.mv(self.B,y),s) + torch.ger(s, torch.mv(torch.t(self.B),y))) / sy
    
    return {'total_loss':loss.numpy(),
            'avg_loss':loss.numpy()/self.data_len,
            'grad_norm':torch.norm(dw).numpy(),
            'inc_norm':pnorm.numpy(),
            'se':np.sqrt(np.diag(self.B.numpy())) if torch.norm(dw).numpy() < 1 else None}
    
  def get_model_info(self):
    return {'weights':self.w.numpy(),
            'se':np.sqrt(np.diag(self.B.numpy())),
            'avg_loss':self._loss(self.w).numpy() / self.data_len}