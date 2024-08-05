import numpy as np
import random
import sklearn.metrics as metrics
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.svm import SVC, NuSVC, SVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# 0 => svc, 1 => nusvc
MTYPES = [0, 1]
# For now, only SVC
MTYPES = [0]
KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
GAMMAS = ['scale', 'auto', 'none']
C = range(1, 50, 1)
DEG = range(1, 10, 1)
COEF= np.linspace(0, 0.2, 10)
TOL = np.linspace(1e-3, 1e-2, 10)

F_NAMES = {
  'ker': 0,
  'gamma': 1, 
  'mtype': 2,
  'c': 3,
  'deg': 4,
  'coef': 5, 
  'tol': 6,
  'tr_acc': 7,
  'val_acc': 8,
  'tr_mae': 9,
  'val_mae': 10}

class SVMScikit:
  def __init__(self) -> None:
    self.regr = SVR(C=50, epsilon=0.10, kernel='poly')
  
  def fit(
        self,
        X,
        y,
        epochs,
        batch_size,
        callbacks=[]):
    X = np.concatenate(X, axis=1)
    self.regr.fit(X, y)

    y_pred = self.regr.predict(X)
    eval = (np.abs(y - y_pred)).mean()
    print("Fitting loss: ", eval)

  def predict(self, X, verbose=0):
    X = np.concatenate(X, axis=1)
    y_pred = self.regr.predict(X)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], -1))
    return y_pred
  
  def evaluate(self, X, y):
    X = np.concatenate(X, axis=1)
    y_pred = self.regr.predict(X)

    eval = (np.abs(y - y_pred)).mean()
    print("Eval loss: ", eval)
    return eval

def one_hot_encoding_system(x_system):
    
    n_systems, n_system_features = x_system.shape
    # One-hot encoding
    one_hot_num = n_system_features + \
          len(KERNELS) + \
          len(GAMMAS)
    
    tr_x_system = np.zeros((n_systems,
                            one_hot_num))

    # Kernel
    enc = np.zeros((n_systems, len(KERNELS)))
    enc[np.arange(n_systems), x_system[:, F_NAMES['ker']].astype(int)] = 1
    col_st = 0
    col_en = len(KERNELS)
    tr_x_system[:, col_st:col_en] = enc
    col_st = col_en

    # gamma
    enc = np.zeros((n_systems, len(GAMMAS)))
    enc[np.arange(n_systems), x_system[:, F_NAMES['gamma']].astype(int)] = 1
    col_en = col_st + len(GAMMAS)
    tr_x_system[:, col_st:col_en] = enc
    col_st = col_en

    #the rest
    col_st = len(KERNELS) + len(GAMMAS) - 2
    tr_x_system[:, 
                col_st+F_NAMES['mtype']:col_st+F_NAMES['val_mae']+1] =\
                    x_system[:, F_NAMES['mtype']:F_NAMES['val_mae']+1]
    
    # Normalizing features
    cols = (tr_x_system.max(axis=0) - tr_x_system.min(axis=0)) != 0
    tr_x_system[:, cols] = (tr_x_system[:, cols] - tr_x_system[:, cols].min(axis=0)) / \
    (tr_x_system[:, cols].max(axis=0) - tr_x_system[:, cols].min(axis=0))
    
    return tr_x_system

def load_dataset(dataset_name: str = 'bike', 
                 seed: int = 42):
    data = np.loadtxt(f'./datasets/{dataset_name}.csv', 
                      delimiter=',')
    X = data[:, :data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]

    # Train+Val and Test sets splitting
    X_train, X_test, y_train, y_test = \
      train_test_split(X, y, 
                      test_size=0.2,
                      random_state=seed)
    
    # Train and Val sets splitting
    X_train, X_val, y_train, y_val = \
      train_test_split(X_train, y_train, 
                      test_size=0.2,
                      random_state=seed)
    
    sc_demom = X_train.max(axis=0)
    cond = sc_demom != 0
    X_train[:, cond] /= sc_demom[cond]
    X_val[:, cond] /= sc_demom[cond]
    X_test[:, cond] /= sc_demom[cond]

    y_max = y_train.max()
    y_train /= y_max
    y_val /= y_max
    y_test /= y_max

    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_system_features(num_systems) -> np.ndarray:
  """
  It generates system features
  INPUT:
    num_systems - int
  OUTPUT:
    x_systems - arrays, shape(num_systems, len(F_NAMES))
  """
  x_system = np.zeros((num_systems, len(F_NAMES)))

  for i in range(num_systems):
    print(f'System feature: {i + 1}')
    # model type
    x_system[i, F_NAMES['mtype']] = random.choice(MTYPES)
    # kernel
    kernel = random.choice(KERNELS)
    x_system[i, F_NAMES['ker']] = KERNELS.index(kernel)
    # C
    x_system[i, F_NAMES['c']] = random.choice(C)
    # Degree is for only poly kernel
    if kernel == 'poly':
      x_system[i, F_NAMES['deg']] = random.choice(DEG)
    else:
      x_system[i, F_NAMES['deg']] = 0
    # Gamma
    if kernel != 'linear':
      gamma = random.choice(GAMMAS[:2])
      x_system[i, F_NAMES['gamma']] = GAMMAS.index(gamma)
    else:
      x_system[i, F_NAMES['gamma']] = GAMMAS.index('none')
    # Coef
    x_system[i, F_NAMES['coef']] = random.choice(COEF)
    # tol
    x_system[i, F_NAMES['tol']] = random.choice(TOL)

  return x_system

def generate_system(x_system):
  model = None
  (kernel, gamma, mtype, C, deg, coef, tol, _, _, _, _) = tuple(x_system)
  kernel = KERNELS[int(kernel)]
  gamma = GAMMAS[int(gamma)]
  deg = int(deg)

  def feature_set(ModelClass):
    if kernel == 'poly':
      model = ModelClass(C=C, 
                  kernel=kernel,
                  degree=deg,
                  gamma=gamma,
                  coef0=coef,
                  tol=tol)
    elif kernel == 'linear':
      model = ModelClass(C=C, 
                  kernel=kernel,
                  coef0=coef,
                  tol=tol)
    else:
      model = ModelClass(C=C, 
                kernel=kernel,
                gamma=gamma,
                coef0=coef,
                tol=tol)
    return model
  # SVC
  if mtype == 0:
    model = feature_set(SVR)
  else:
     model = feature_set(NuSVR)  

  return model

def train_assessor(X, 
                   y, 
                   x_system, 
                   systems, 
                   ass_model, 
                   random=True,
                   n_updates: int = 2):
    # Train assessor model
    # Generate assessor dataset from training dataset
    # Selecting systems randomly
    n_train = X.shape[0]
    n_systems, n_system_features = x_system.shape
    
    x_ass_train = np.zeros((n_train, n_system_features))
    y_ass_train = np.zeros((n_train, ))

    system_indices = None

    def updaete(epochs=1):
        for i, model in enumerate(systems):
            cond = i == system_indices
            if sum(cond) == 0: continue
            y_pred = model.predict(X[cond])
            y_ass_train[cond] = y_pred
            x_ass_train[cond] = x_system[i]

        y_ass_delta = y - y_ass_train

        print('='*20, "Training assessor on the assessor dataset", '='*20)
        ass_model.fit(
            [X, x_ass_train],
            y_ass_delta,
            epochs=epochs,
            batch_size=64,
            callbacks=[])

    if random:
      # Randomly take system
      for i in range(5):
          system_indices = np.random.randint(0, n_systems, n_train)
          updaete(epochs=2)
    
    for i in range(n_updates):
      # select best systems
      a_preds = np.zeros((n_train, n_systems))
      input_x_system = np.zeros((n_train, n_system_features))
      for j in range(n_systems):
          input_x_system[np.arange(X.shape[0])] = x_system[j]
          a_preds[:, j] = ass_model.predict([X, input_x_system])[:, 0]

      system_indices = np.argmin(np.abs(a_preds), axis=1)

      updaete(epochs=20)