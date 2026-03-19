"""
Basic utility functions/classes for training the autoencoder
"""

import pdb
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from .core import load_or_compute_snaps
from .config import (
                    TRAIN_FRAC, BATCH_SIZE,
                    DT, NUM_STEPS, NUM_CELLS, W0, GRID_X, GRID_Y, SNAP_FOLDER
                   )

class TrainingMonitor:
  def __init__(self, model_path, patience, model, opt, scheduler, train=True):
    self.model_path = model_path
    self.best_crit = 1E16
    self.patience = patience
    self.its_since_improvement = 0
    self.model = model
    self.opt = opt
    self.scheduler = scheduler
    self.epoch = 0
    self.train_losses = []
    self.test_crits = []
    if train:
      self.writer = SummaryWriter()

  def check_for_completion(self, train_loss, test_crit):
    self.epoch += 1
    self.its_since_improvement += 1
    self.train_losses += [train_loss]
    self.test_crits += [test_crit]
    self.writer.add_scalar('loss/train', train_loss, self.epoch)
    self.writer.add_scalar('loss/test', test_crit, self.epoch)
    if test_crit < self.best_crit:
      self.best_crit = test_crit
      self.its_since_improvement = 0
      self.save_checkpoint()
    print('  Its since improvement: {}'.format(self.its_since_improvement))
    if self.its_since_improvement > self.patience:
      return True
    else:
      return False

  def plot_training_curves(self):
    fig_curves, ax_curves = plt.subplots()
    ax_curves.semilogy(self.test_crits, label='Test loss')
    ax_curves.semilogy(self.train_losses, label='Train loss')
    ax_curves.set_xlabel('Epoch')
    ax_curves.set_ylabel('MSE loss')
    ax_curves.set_title('Training curves')
    ax_curves.legend()

  def save_checkpoint(self):
    checkpoint = {
        'epoch': self.epoch,
        'model_state_dict': self.model.state_dict(),
        'opt_state_dict': self.opt.state_dict(),
        'sched_state_dict': self.scheduler.state_dict(),
        'train_losses': self.train_losses,
        'test_crits': self.test_crits
        }
    print('  ... saving new model')
    torch.save(checkpoint, self.model_path)

  def load_from_path(self, path):
    print(' ... loading model from {}'.format(path))
    checkpoint = torch.load(path)
    self.epoch = checkpoint['epoch']
    
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.opt.load_state_dict(checkpoint['opt_state_dict'])
    self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
    self.train_losses = checkpoint['train_losses']
    self.test_crits = checkpoint['test_crits']
    self.best_crit = min(self.test_crits)


def random_split(snaps, frac, rng):
  n = snaps.shape[0]
  num1 = int(n*frac)
  perm = rng.permutation(n)
  snaps_perm = snaps[perm, :]
  snaps1 = snaps_perm[:num1, :]
  snaps2 = snaps_perm[num1:, :]
  return snaps1, snaps2

def random_split_bis(snaps, frac, rng):
  n = snaps.shape[0]
  num1 = int(n*frac)
  perm = rng.permutation(n)
  snaps_perm = snaps[perm, :]
  snaps1 = snaps_perm[:num1, :]
  snaps2 = snaps_perm[num1:, :]
  return snaps1, snaps2, perm

def get_data(np_rng, mu_samples, dtype='float64'):
  snaps_np, ref = load_data(mu_samples)
  snaps_np, ref = np.array(snaps_np, dtype=dtype), np.array(ref, dtype=dtype)
  train_np, val_np = random_split(snaps_np, TRAIN_FRAC, np_rng)
  train_t = torch.from_numpy(train_np).unsqueeze(1)
  val_t = torch.from_numpy(val_np).unsqueeze(1)
  train_data = TensorDataset(train_t)
  val_data = TensorDataset(val_t)
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  return train_t, val_t, train_data, val_data, train_loader, val_loader, ref

def load_data(mu_samples):
  # Generate or retrive HDM snapshots
  all_snaps_list = []
  for mu in mu_samples:
      snaps = load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER)
      all_snaps_list += [snaps[:, :-1]]

  snaps = np.hstack(all_snaps_list)
  ref = snaps[:, 0].copy()
  snaps -= ref[:, np.newaxis]
  return snaps.T, ref

def project_onto_manifold(auto, x, lr=1e-4, num_steps=1000):
  with torch.no_grad():
    zi = auto.enc(auto.scaler(x)).requires_grad_()
  zi.retain_grad()
  xi = auto.unscaler(auto.dec(zi))
  dec = auto.dec
  losses = []
  opt = torch.optim.Adam([zi], lr=lr)
  for i in range(num_steps):
    dec.zero_grad()
    xi = auto.unscaler(dec(zi))
    loss = ((xi - x)**2).sum()
    loss.backward()
    with torch.no_grad():
      losses += [loss.item()]
      opt.step()
      zi.grad = None

  return xi.detach()

def show_model(model, train, test, device='cpu'):
  fig, (ax1, ax2) = plt.subplots(2, 3)
  for i, ax in enumerate(ax1):
    x = train[i][0].unsqueeze(0).float().to(device)
    with torch.no_grad():
      out = model(x).to('cpu')
    ax.plot(x.squeeze().to('cpu'), linewidth=2, color='k', label='truth')
    ax.plot(out.squeeze(), linewidth=1, color='r', label='autoencoder')
    ax.set_title('Training sample')
    ax.legend()
  for i, ax in enumerate(ax2):
    x = test[i][0].unsqueeze(0).float().to(device)
    with torch.no_grad():
      out = model(x).to('cpu')
    ax.plot(x.squeeze().to('cpu'), linewidth=2, color='k', label='truth')
    ax.plot(out.squeeze(), linewidth=1, color='r', label='autoencoder')
    ax.set_title('Test sample')
  fig.tight_layout()