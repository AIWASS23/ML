
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
import pandas as pd

knn = KNeighborsClassifier(n_neighbors=3)
data = pd.read_table('fruit_data_with_colors.txt')
X = data[['mass','height','width','color_score']]
y = data['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X,y)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
knn.predict(X_test)
knn = KNeighborsRegressor(n_neighbors=3)
data = load_boston()
X, y = load_boston(return_X_y=True)
X.shape
y.shape
print(load_boston()['DESCR'])
X_train, X_test, y_train, y_test = train_test_split(X,y)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)