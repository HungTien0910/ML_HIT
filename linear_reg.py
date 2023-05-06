import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model

# Lấy dữ liệu
df = pd.read_csv("Linear_Reg\FuelConsumptionCo2.csv")

data = df[['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]\
    .rename(columns={'FUELCONSUMPTION_HWY': 'HWY',
        'FUELCONSUMPTION_COMB_MPG': 'COMB_MPG'})

plt.scatter(data.HWY, data.CO2EMISSIONS, color = 'red')
plt.xlabel('HWY')
plt.ylabel('CO2EMISSIONS')
plt.show()

plt.scatter(data.COMB_MPG, data.CO2EMISSIONS, color = 'blue')
plt.xlabel('COMB_MPG')
plt.ylabel('CO2EMISSIONS')
plt.show()

# split the data
msk = np.random.rand(len(df)) < 0.8
train = data[msk]
test = data[~msk]


#split the traing data 
train_x = np.array(train[['HWY', 'COMB_MPG']])
train_y = np.array(train['CO2EMISSIONS'])
#split the testing data
test_x = np.array(test[['HWY', 'COMB_MPG']])
test_y = np.array(test['CO2EMISSIONS'])

one = np.ones((train_x.shape[0], 1))
Xbar = np.concatenate((train_x, one), axis = 1)

#tính weight, bias
def linear_regression():
  A = np.dot(Xbar.T, Xbar)
  B = np.dot(Xbar.T, train_y)
  return np.dot(np.linalg.pinv(A), B)
lr = linear_regression()
w = lr[:2][:]
b = lr[2:][:]

def predict(X):
    return X @ w[0 : 2] + b

def compute_cost(Y, Y_):
    return .5 / Y.shape[0] * np.linalg.norm((Y - Y_)**2)

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))
print("Cost: %.2f" % compute_cost(test_y , test_y_) )


xy_plt = np.concatenate([np.linspace(0, 25, 100)[:, None], np.linspace(0, 65, 100)[:, None]], axis=1)
X, Y = np.meshgrid(xy_plt[:, 0], xy_plt[:, 1])


zs = np.array([regr.predict([[x,y]]) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax = plt.axes(projection='3d')
ax.view_init(10, 3)

ax.scatter3D(test_x[:, 0], test_x[:, 1], test_y, 'blue')
ax.set_xlabel('COMB_MPG')
ax.set_ylabel("CO2EMISSIONS")
ax.set_zlabel('HYW')
ax.plot_surface(X, Y, Z, color='g', alpha=0.5)

plt.show()
