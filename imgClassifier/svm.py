from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from fetchedData import X, y

"""
ordinary SVM
"""

# create a svm Classifier
ordinaryClf = svm.SVC(kernel='linear', C=1, random_state=42)  # Linear Kernel

# split the data between training and testing data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

# training the model
ordinaryClf.fit(X_train, y_train)

# testing the model
y_pred = ordinaryClf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
SVM with 10-way cross validation.

TODO you should choose by yourself which technique to apply 
(for the splitting technique) and not the default one
"""


# create a svm Classifier
clf = svm.SVC(kernel='linear', C=1, random_state=42)  # Linear Kernel

# train the model with 10 way cross validation
scores = cross_val_score(clf, X, y, cv=10)

# a look at the statistics
print("mean:", end=" ")
print(scores.mean())

print("sd:", end=" ")
print(scores.std())


