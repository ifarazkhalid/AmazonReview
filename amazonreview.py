import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def main():
    # Read the file

    df = pd.read_csv('amazon_reviews.csv', header=0, usecols=[0, 1], encoding='latin-1')
    print('rows and columns:', df.shape)
    print(df.head())

    #Removing stopwords and creating a vectorizor.
    stop = set(stopwords.words('english'))
    print(stop)
    vectorizer = TfidfVectorizer(stop_words = stop, binary=True)

    X = vectorizer.fit_transform(df.Review)
    y = df.Rating

    #Creating training and test sets.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1234)

    #Modeling
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

                            #Logisitc Regression Modeling

    '''
    classifier2 = LogisticRegression(class_weight='balanced')
    classifier2.fit(X_train, y_train)
    pred2 = classifier2.predict(X_test)
    print('accuracy score: ', accuracy_score(y_test, pred2))
    print('precision score: ', precision_score(y_test, pred2))
    print('recall score: ', recall_score(y_test, pred2))
    print('f1 score: ', f1_score(y_test, pred2))
    
    '''

                            #Naive Bayes

    '''
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)

    #Testing on test set
    pred = naive_bayes.predict(X_test)
    confusion_matrix(y_test, pred)

    print('accuracy score: ', accuracy_score(y_test, pred))
    print('precision score: ', precision_score(y_test, pred))
    print('recall score: ', recall_score(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred))
    '''

                            #Random Forest

    """
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    print('accuracy score: ', accuracy_score(y_test, pred))
    print('precision score: ', precision_score(y_test, pred))
    print('recall score: ', recall_score(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred))
    
    """

                            #SVM

    '''
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    classifier = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                         decision_function_shape='ovr', degree=3, kernel='rbf', gamma=0.001,
                         max_iter=-1, probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False)

    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)

    print('accuracy score: ', accuracy_score(y_test, pred))
    print('precision score: ', precision_score(y_test, pred))
    print('recall score: ', recall_score(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred))
    '''

                            #Neural Network

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(15, 2), random_state=1)
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    print('accuracy score: ', accuracy_score(y_test, pred))
    print('precision score: ', precision_score(y_test, pred))
    print('recall score: ', recall_score(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred))







if __name__ == "__main__":
    main()