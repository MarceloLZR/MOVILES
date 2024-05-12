from django.shortcuts import render
from django.http import HttpResponse
from keras.models import Model,model_from_json
from keras.models import Sequential
#from keras.layers.core import Dense
from keras.layers import Activation, Dense, Dropout
from keras import regularizers
#import pandas as pd

def respuesta(modelo,X):
  res = modelo.predict([X])
  res = res.round()[0][0]
  return res

def recibe(request,x0,x1,x2,x3,x4):
  X = [float(x0),float(x1),float(x2),float(x3),float(x4)]
  from keras.models import Model,model_from_json
  json_file = open("apipy/modelo.json","r")
  modelo_cargado_json = json_file.read()
  json_file.close()
  modelo_cargado = model_from_json(modelo_cargado_json)
  modelo_cargado.load_weights("apipy/modelo.h5")
  modelo_cargado.compile(loss='mean_squared_error', optimizer= 'adam', metrics = 'binary_accuracy')
  result = modelo_cargado.predict([X])
  print(result.round()[0][0])
  
  return HttpResponse(result.round()[0])