from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

X,y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes = (100,100,50,50),max_iter = 1000)
mlp.fit(X_train,y_train)
mlp.score(X_test,y_test)



