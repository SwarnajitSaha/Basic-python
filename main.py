import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('iphone_price.csv')

# plt.scatter(data['version'],data['Price'])
# plt.show()


model = LinearRegression()
model.fit(data[['version']], data[['Price']])
print(model.predict([[20]]))
