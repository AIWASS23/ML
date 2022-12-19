from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

svm = SVC(kernel = 'linear',C = 1.0)
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

X,y = load_iris(return_X_y = True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

svm = SVC(kernel = 'linear')
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

svm_kernel = SVC(kernel = 'poly',degree = 3)
svm_kernel.fit(X_train,y_train)
svm_kernel.score(X_test,y_test)

svm_rbf = SVC(kernel = 'rbf')
svm_rbf.fit(X_train,y_train)
svm_rbf.score(X_test,y_test)



