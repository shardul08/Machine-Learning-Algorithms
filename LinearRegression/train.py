import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("data/student-mat.csv",sep=";")
#print(data.head())

data = data[["G1","G2","G3","traveltime","studytime","failures","freetime","absences"]]

predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
'''
best = 0
for _ in range(50):
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

	linear = linear_model.LinearRegression()

	linear.fit(x_train,y_train)

	acc = linear.score(x_test,y_test)
	print(acc)
	if acc > best:
		best = acc
		with open("studentmodel.pickle","wb") as f:
			pickle.dump(linear,f)
'''
pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
print("Co: ", linear.coef_)
print("Intercept: ",linear.intercept_)

prediction = linear.predict(x_test)

for i in range(len(prediction)):
	print(prediction[i],x_test[i],y_test[i])

p = "G1"
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
