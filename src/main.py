from sklearn.neighbors import KNeighborsClassifier
from src.Utils import scan_directory, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt


def main():
    ROOT_DIR = '../raw/dataset/'
    dataset_train = scan_directory(ROOT_DIR + 'Train')
    dataset_test = scan_directory(ROOT_DIR + 'Test')

    X_train = preprocessing(dataset_train['text'])
    X_test = preprocessing(dataset_test['text'])
    Y_train = dataset_train['category']
    Y_test = dataset_test['category']

    NB(X_train, X_test, Y_train, Y_test)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=1, useIdf=False)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=5, useIdf=False)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=15, useIdf=False)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=1, useIdf=True)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=5, useIdf=True)
    KNN(X_train, X_test, Y_train, Y_test, n_neighbors=15, useIdf=True)
    plt.show()


def NB(X_train, X_test, Y_train, Y_test):
    cv = CountVectorizer(max_features=500)
    X_all = cv.fit_transform(X_train + X_test).toarray()
    X_train = X_all[0:56, :]
    X_test = X_all[56:70, :]
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    execute(classifier, X_test, Y_test, "Naive Bayes")


def KNN(X_train, X_test, Y_train, Y_test, n_neighbors, useIdf):
    transformer = TfidfVectorizer(max_features=500, use_idf=useIdf)
    X_all = transformer.fit_transform(X_train + X_test).toarray()
    X_train = X_all[0:56, :]
    X_test = X_all[56:70, :]
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    classifier.fit(X_train, Y_train)
    execute(classifier, X_test, Y_test, f"KNeighbors (k={n_neighbors}, IDF={useIdf})")


def execute(classifier, X_test, Y_test, algorithm_name):
    Y_pred = classifier.predict(X_test)
    print(f"{algorithm_name}:")
    print(f"Accuracy= {accuracy_score(Y_test, Y_pred)}")
    print(f"Precision= {precision_score(Y_test, Y_pred, average='weighted', zero_division=0)}")
    print(f"Recall= {recall_score(Y_test, Y_pred, average='weighted', zero_division=0)}")
    print(f"F1= {f1_score(Y_test, Y_pred, average='weighted', zero_division=0)}")
    print("-------------------------------")
    disp = plot_confusion_matrix(classifier, X_test, Y_test, cmap=plt.cm.Blues)
    disp.figure_.text(.5, .01, algorithm_name)


if __name__ == '__main__':
    main()
