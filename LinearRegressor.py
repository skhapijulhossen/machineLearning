import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loading Fuel Consumption Data
data = pd.read_csv(r"C:\PYTHON\Lets_Solve\DataStructures\FuelConsumption.csv")
print(data.head(10))


# Spliting Data into Train Test Set
x_train = data[["ENGINESIZE"]][:int(data.shape[0]*0.8)].values
y_train = data[["CO2EMISSIONS"]][:int(data.shape[0]*0.8)].values
x_test = data[["ENGINESIZE"]][int(data.shape[0]*0.8):].values
y_test = data[["ENGINESIZE"]][int(data.shape[0]*0.8):].values


# Regression Model From Scratch
class LinearRegressor:
    def __init__(self):
        self.intercept = 0
        self.coefficient = 0
        print("Linear Regressor Initialized!")
    

    def fit(self, x, y):
        xbar = np.mean(x)
        ybar = np.mean(y)
        self.coefficient = int(np.sum(((x-xbar) * (y- ybar)))/np.sum(((x-xbar)**2)))
        self.intercept = int(ybar - (xbar * self.coefficient))
 

    def predict(self, x):
        yhat = np.array(self.intercept + (x * self.coefficient), dtype=np.int64)
        return yhat


# Initializing Model and Training MOdel
model = LinearRegressor()
model.fit(x_train, y_train)


# Predicting Model
y_pred = model.predict(x_test)


# Plotting Model
plt.scatter(data[["ENGINESIZE"]][int(data.shape[0]*0.8):].values, data[["CO2EMISSIONS"]][int(data.shape[0]*0.8):].values)
plt.plot(data[["ENGINESIZE"]][int(data.shape[0]*0.8):].values, y_pred)
plt.show()