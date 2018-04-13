import numpy as np
import pickle


with open('reses.pickle', 'rb') as f:
   res = pickle.load(f)

print(res)

