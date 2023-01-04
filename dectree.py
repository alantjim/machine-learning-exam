from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

set=load_iris()
x=set.data
y=set.target

# set=pd.read_csv("slr.csv")
# x=set.iloc[1:].values
# y=set.iloc[:-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
m=DecisionTreeClassifier()
m.fit(x_train,y_train)
pr=m.predict(x_test)
print(pr)

plot_tree(m,filled=True)
plt.show()
acc=accuracy_score(y_test,pr)
print (acc)
