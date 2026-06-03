# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Function used to pad all sequences in a batch to the same size
def collate_fn(batch):
    # Sort the batch by sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    # Separate the input data and targets
    sequences, targets = zip(*batch)

    # Pad the sequences
    sequences_padded = pad_sequence(sequences, batch_first=True)

    # Stack the targets into a single tensor
    targets_padded = pad_sequence(targets,batch_first=True)

    # Get the lengths of the sequences
    lengths = torch.tensor([len(seq) for seq in sequences])

    return sequences_padded, lengths, targets_padded

class PAHMDataset(Dataset):
  """
  A PyTorch Dataset class for loading PAHM data.

  This is device agnostic.

  You can normalize the output angles, but this expects the input data to be
  already normalized from 0 to 1.
  """

  def __init__(self,
               root_dir,
               normalize_angles=True,
               min_angle=-120,max_angle=120,
               extension=None,
               verbose=False):
    """
    Args:
      root_dir (str): Path to the directory containing CSV files.
      device (str): Device to store tensors ("cuda" or "cpu").
      normalize (bool, optional): Whether to normalize angle values. Defaults to True.
    """
    self.root_dir = root_dir
    self.normalize_angles = normalize_angles
    self.data = []
    self.min_angle = min_angle
    self.max_angle = max_angle

    self.angle_norm_slope = 2/(self.max_angle-self.min_angle)
    self.angle_norm_intercept = 1 - self.angle_norm_slope*self.max_angle

    self.num_features=[]
    self.file_names = []  # This list stores the filename for each data point
    
    prepadding = np.zeros(1500)


    # Read data from CSV files
    for filename in os.listdir(root_dir):
      datafile=os.path.join(root_dir, filename)
      if verbose:
        print(f"Reading file '{datafile}'")
      data = pd.read_csv(datafile)
      pwm = np.concatenate((prepadding, data.values[:, 2]))

      # Modify pwm based on extension argument
      if extension == "zero":
        pwm = np.column_stack((pwm, np.zeros_like(pwm)))
        pwm[0,1]=0
        self.num_features=2
      elif extension == "one":
        pwm = np.column_stack((pwm, np.ones_like(pwm)))
        pwm[0,1]=0
        self.num_features=2
      elif extension == "past":
        # Shift elements, create a copy for the first element
        pwm = np.column_stack((pwm, np.roll(pwm, 1)))  
        pwm[0,1]=0
        self.num_features=2
      else: # Just use the scalar input, but return is as an m x 1 matrix
        pwm = pwm.reshape(-1,1)
        self.num_features=1

      
      if self.normalize_angles:        
        angle=np.concatenate((prepadding, self.norm(data.values[:, 3])))
      else:
        angle=np.concatenate((prepadding, data.values[:, 3]))
        
      angle = angle.reshape(-1,1)

      pwm = torch.tensor(pwm, dtype=torch.float32)
      angle = torch.tensor(angle, dtype=torch.float32)

      self.data.append((pwm,angle))
      self.file_names.append(filename)  # Store the filename for this data point


  def norm(self,data):
      data = data*self.angle_norm_slope + self.angle_norm_intercept
      return data
                             
  def __len__(self):
    return len(self.data)
  
  def features(self):
    return self.num_features
    

  def __getitem__(self, idx):
    # Preprocess data
    pwm, angle = self.data[idx]
    return pwm, angle

def get_dataloaders(root_dir,
                    train_split=0.6,
                    val_split=0.2,
                    normalize_angles=True,
                    min_angle=-120,
                    max_angle=120,
                    batch_size=1,
                    extension=None,
                    verbose=False):
  """Creates training, validation, and testing dataloaders without
     randomization."""

  dataset = PAHMDataset(root_dir=root_dir,
                        normalize_angles=normalize_angles,
                        min_angle=min_angle,
                        max_angle=max_angle,
                        extension=extension,
                        verbose=verbose)
  total_len = len(dataset)

  train_len = int(train_split * total_len)
  val_len = int(val_split * total_len)
  test_len = total_len - train_len - val_len

  # Randomly split the dataset
  generator = torch.Generator().manual_seed(425)
  indices = list(range(total_len))
  random_split(indices, lengths=[train_len, val_len, test_len], generator=generator)

  train_idx = indices[:train_len]
  val_idx   = indices[train_len:train_len+val_len]
  test_idx  = indices[train_len+val_len:]

  # Create data splits with filenames
  train_data = [dataset[i] for i in train_idx]

  if verbose:
      print("Train data uses the following files:")
      for num,i in enumerate(train_idx):
          print(f"{num+1}: {dataset.file_names[i]}")
      print()
  
  val_data = [dataset[i] for i in val_idx]

  if verbose:
      print("Validation data uses the following files:")
      for num,i in enumerate(val_idx):
          print(f"{num+1}: {dataset.file_names[i]}")
      print()

  test_data = [dataset[i] for i in test_idx]

  if verbose:
      print("Testing data uses the following files:")
      for num,i in enumerate(test_idx):
          print(f"{num+1}: {dataset.file_names[i]}")
      print()

  
  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
  val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False,collate_fn=collate_fn)
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate_fn)

  return train_dataloader, val_dataloader, test_dataloader, dataset

