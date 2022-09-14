import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/ahmet/OneDrive/Masaüstü/Python Files/Position_Salaries.csv")

x = data.iloc[:,1:2] #level
y = data.iloc[:,2:]  #salary

X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression

lin_reg1 = LinearRegression()
lin_reg1.fit(X, Y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

#scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)

sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)

#svr
from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled, y_scaled.ravel())

#visualisation

plt.subplot(1,3,1)
plt.scatter(X, Y, color="r")
plt.plot(X, lin_reg1.predict(X), color="b")
plt.title("Linear Reg")
plt.grid(True)

plt.subplot(1,3,2)
plt.scatter(X, Y, color="r")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="b")
plt.title("Polynomial Reg")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(x_scaled, y_scaled, color="r")
plt.plot(x_scaled, svr_reg.predict(x_scaled), color="b")
plt.title("Support Vector Reg")
plt.grid(True)