import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("Gradient-Descent\FuelConsumption.csv")
#print(df.head())
#print(df.info())
data = df[['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]\
        .rename(columns={'FUELCONSUMPTION_HWY': 'HWY',
                 'FUELCONSUMPTION_COMB_MPG': 'COMB_MPG'})
#print(data.head())
data.skew()
data['HWY_log'] = data['HWY'].apply(np.log1p)
data['HWY_log'].hist();
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
#print(data.head())
sns.pairplot(data);
data['COMB_MPG^2'] = data['COMB_MPG']**2

X = np.asanyarray(data[['HWY_log', 'COMB_MPG', 'COMB_MPG^2']])
Y = np.asanyarray(data[['CO2EMISSIONS']])
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=12)
train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis = 1)


print('Train size:', len(train_y))
print('Train test:', len(test_y))

def linear_regression():
  A = train_x.T @ train_x
  b = train_x.T @ train_y
  return np.linalg.pinv(A) @ b
lr = linear_regression()
w = lr[:3][:]
b = lr[3:][:]
print(w, b, " ")

def predict(X, w, b):
  return X @ w + b
def compute_cost(y, y_pre):
  return .5/y.shape[0]*np.linalg.norm(y - y_pre, 2)**2
import numpy as np

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dw = (1/m) * np.dot(x.T, (np.dot(x, w) + b - y))
    db = (1/m) * np.sum(np.dot(x, w) + b - y)
    return dw, db
def gradient_descent(x_train, y_train, w, alpha, numIter):
  J_history = []
  weight = [w]
  for i in range(numIter):
    grad = compute_gradient(x_train, y_train, weight[-1])
    weight_new = weight[-1] - alpha * grad
    J_history.append(compute_cost(y_train, predict(x_train, weight_new)))
    weight.append(weight_new)
  return weight[-1], J_history, weight
# initialize parameters
w_init = np.array([[0], [0], [0], [0]])
b_init = w_init[2:][:]
# some gradient descent settings
iterations = 400
alpha = 0.001
# run gradient descent
w, J_hist, p_hist = gradient_descent(train_x, train_y, w_init, 2e-1, 50)
print(w[-1])
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

test_y_ = predict(test_x, w)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))
print("Cost: %.2f" % compute_cost(test_y , test_y_))

r = 1
limit_x = data.HWY.min()-r, data.HWY.max()+r
limit_y = data.COMB_MPG.min()-r, data.COMB_MPG.max()+r
xy_plt = np.concatenate([np.linspace(*limit_x, 100)[:, None], np.linspace(*limit_y, 100)[:, None]], axis=1)

X, Y = np.meshgrid(xy_plt[:, 0], xy_plt[:, 1])
zs = regr.predict(np.concatenate([np.ravel(X)[:, None], np.ravel(Y)[:, None], np.ravel(Y**2)[:, None]], axis=1))
Z = zs[:, None].reshape(X.shape)


ax = plt.axes(projection='3d')
ax.view_init(7, 3) # change view to see more

ax.scatter3D(train_x[:, 0], train_x[:, 1], train_y, 'blue')

ax.plot_surface(X, Y, Z, color='r', alpha=0.5);

plt.show()