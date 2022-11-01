from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

dt = DecisionTreeClassifier()

X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

dt.fit(X_train,y_train)

dt.score(X_test,y_test)

dt = DecisionTreeRegressor()

X,y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y)

dt.fit(X_train,y_train)

dt.score(X_test,y_test)

dt = DecisionTreeClassifier(max_depth=3)

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y)

dt.fit(X_train,y_train)

dt.score(X_test,y_test)

dt = DecisionTreeClassifier()

X,y = load_breast_cancer(return_X_y=True)

dt.fit(X,y)

X.shape

dt.feature_importances_
