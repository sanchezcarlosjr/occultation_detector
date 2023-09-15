import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class WrongInput(Exception):
    def __init__(self, message="The model needs 92 timestamps to work"):
        self.message = message
        super().__init__(self.message)

class Dataset:
   def load(self):
     pass

class NumpyLoader(Dataset):
   def __init__(self, instances):
       self.instances = instances
   def load(self):
       return np.array([instance[:,:92][1] for instance in self.instances])

class NpyFileLoader(Dataset):
   def __init__(self, *files):
      self.paths = []
      for file in files:
          if isinstance(file, str):
             self.paths.extend(glob.glob(file))
          else:
             self.paths.append(file)
   def plot(self):
      instances = self.load()
      for instance in instances:
          plt.plot(instance)
   def load(self):
      instances=[np.load(path) for path in self.paths]
      return NumpyLoader(instances).load()

class DataFrameLoader(Dataset):
   def __init__(self, *dfs):
     self.dfs = dfs
   def load(self):
     return pd.concat([df['intensity'].T for df in self.dfs])

class CSVFrameLoader(Dataset):
   def __init__(self, files):
        self.files = files
   def load(self):
        return DataFrameLoader(*[pd.read_csv(file) for file in self.files]).load()

def transform(response):
    keys = ['diameter', 'ua', 'toffset', 'T', 'b', 'D', 'z', 'R_star', 'type']
    return [dict(zip(keys, output)) for output in response]

def pipeline(model_path="checkpoints/vanilla-neuronal-network.keras"):
    model = Model(model_path)
    model.load_model()
    def predict(dataset):
        intensity = dataset.load()
        return transform(model.predict(intensity))
    return predict

class Model:
    def __init__(self, path):
        self.path = path
    def load_model(self):
        self.model = tf.keras.models.load_model(self.path)
    def predict(self, lightcurve):
        return self.model.predict(lightcurve, verbose=0)
