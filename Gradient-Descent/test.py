import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

train_x = np.array([[0,1,0,1],[0,0,1,1]])
train_y = np.array([0,1,1,1])

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

