import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

apple = pd.read_csv("D:/Python/Dự đoán giá cổ phiếu/AAPL.csv") 
print(apple.head())

print("train_test_spliting days =",apple.shape)

sns.set()
plt.figure(figsize=(10, 4))
plt.title("Apple's Stock Price")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close Price"])
plt.show()

apple = apple[["Close Price"]]
print(apple.head())

futureDays = 30

apple["Prediction"] = apple[["Close Price"]].shift(-futureDays)
print(apple.head())
print(apple.tail())

x = np.array(apple.drop(["Prediction"], 1))[:-futureDays]
print(x)	

y = np.array(apple["Prediction"])[:-futureDays] 
print(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor().fit(xtrain, ytrain)

from sklearn.linear_model import LinearRegression
linear = LinearRegression().fit(xtrain, ytrain)

xfuture = apple.drop(["Prediction"], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)
print(xfuture)

treePrediction = tree.predict(xfuture)
print("Decision Tree Prediction =",treePrediction)

linearPrediction = linear.predict(xfuture)
print("Linear Regression Prediction =",linearPrediction)

predictions = treePrediction
valid = apple[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Apple's Stock Price Prediction Model")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close Price"])
plt.plot(valid[["Close Price", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()

predictions = linearPrediction
valid = apple[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Apple's Stock Price Prediction Model")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close Price"])
plt.plot(valid[["Close Price", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()



