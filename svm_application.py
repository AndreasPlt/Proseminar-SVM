import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_digits()

X = cancer.data
y = cancer.target

def predict(kernel, c, deg, gamma):
    n = 10
    sum_acc = 0
    for i in range(n):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1,
                                                                                    random_state=i)
        classifier = svm.SVC(kernel=kernel, C=c, degree=deg, gamma=gamma)
        classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_test)
        acc = metrics.accuracy_score(y_predict, y_test)
        sum_acc += acc

    mean_acc = sum_acc/n
    print(f"Accuracy:\t{mean_acc}\nKernel used:\t{kernel}\nC param:\t{c}\nDegree\t{deg}\nGamma param:\t{gamma}\t")
    print("\n*********************\n")


for i in [10**5, 1, 10**-5]:
    predict("linear", i, 0, "scale")
    predict("poly", i, 3, "scale")
    predict("poly", i, 11, "scale")
    predict("rbf", i, 1, "scale")
    predict("sigmoid", i, 1, "scale")



