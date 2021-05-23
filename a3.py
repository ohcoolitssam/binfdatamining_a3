#created and edited by Samuel Phillips

#imports for data, classes and more
from sklearn.datasets import load_iris
from pandas import DataFrame
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#-- a3p1 starts here --
#iris data is loaded
iris = load_iris()
iData = iris.data

#data is collected from the iris dataset
X = iris.data[50:]
y = iris.target[50:]

error_rates = []

for i in range(0,20):
    #train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    p1 = clf.predict(X_test)

    vals = []

    #for loop that collects the x and y values of the correct and incorrect predictions
    for z in range(0, len(p1)):
        if p1[z] == y_test[z]:
            vals.append(0)

        elif p1[z] != y_test[z]:
            vals.append(1)
    
    error_rates.append(np.mean(vals))
    
er = DataFrame(error_rates)
ax = er.plot(kind='line', figsize=(20,4))
ax.set_xlabel('prediction indecies')
ax.set_ylabel('error rate')
#-- a3p1 ends here --

#-- a3p2 starts here --
#iris data is loaded
iris = load_iris()
iData = iris.data

#data is collected from the iris dataset
X = iris.data[50:]
y = iris.target[50:]

#train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
clf = GaussianNB()
clf.fit(X, y)
p1 = clf.predict_proba(X)

print(p1)
#-- a3p2 ends here --