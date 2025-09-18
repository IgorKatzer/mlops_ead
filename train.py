import os
import random
import numpy as np
import random as python_random
import mlflow
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# # Definindo funções adicionais

def reset_seeds() -> None:
  """
  Resets the seeds to ensure reproducibility of results.

  This function sets the seed for various random number generation libraries
  to ensure that results are reproducible. The affected libraries are:
  - Python's built-in `random`
  - NumPy
  - TensorFlow

  The seed used is 42.

  Returns:
      None
  """
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

# # 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis

def read_data():
  url = 'raw.githubusercontent.com'
  username = 'renansantosmendes'
  repository = 'lectures-cdas-2023'
  file_name = 'fetal_health_reduced.csv'
  data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')
  X=data.drop(["fetal_health"], axis=1)
  y=data["fetal_health"]
  return X, y



# # 3 - Preparando o dado antes de iniciar o treino do modelo


def process_data(X, y):
  columns_names = list(X.columns)
  scaler = preprocessing.StandardScaler()
  X_df = scaler.fit_transform(X)
  X_df = pd.DataFrame(X_df, columns=columns_names)

  X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=42)

  y_train = y_train -1
  y_test = y_test - 1
  return X_train, X_test, y_train, y_test

# # 4 - Criando o modelo, adicionando as camadas e o compilando

def create_model(X):
  reset_seeds()
  model = Sequential()
  model.add(Input(shape=(X.shape[1], )))
  model.add(Dense(units=10, activation='relu'))
  model.add(Dense(units=10, activation='relu'))
  model.add(Dense(units=3, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  return model




# ##**Configurando o mlflow**
def config_mlflow():
  os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
  os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
  mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow')

  mlflow.keras.autolog(log_models=True,
                      log_input_examples=True,
                      log_model_signatures=True)

# # 6 - Executando o treino do modelo
def train_model(model, X_train, y_train, is_train=True):
  with mlflow.start_run(run_name='experiment_mlops_ead') as run:
    model.fit(X_train,
              y_train,
              epochs=50,
              validation_split=0.2,
              verbose=3)
  if is_train:
    run_uri = f'runs:/{run.info.run_id}/model'
    mlflow.register_model(model_uri=run_uri, name='fetal health')

if __name__ == "__main__":
  X, y = read_data()
  X_train, X_test, y_train, y_test = process_data(X, y)
  model = create_model(X)
  config_mlflow()
  train_model(model, X_train, y_train)


