from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from fetchedData import X, y

"""
TODO check n_estimators the higher the number the higher the accuracy of this machine learning model but the longer it 
takes to be trained
"""

"""
ordinary Random forest 
"""

# create a random forest classifier
ordinaryClf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# training the model
ordinaryClf.fit(X_train, y_train)

# testing the model
y_pred = ordinaryClf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
Random forest with 10-way cross validation
"""

# create a random forest classifier
clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)

# train the model with 10 way cross validation
scores = cross_val_score(clf, X, y, cv=10)

# a look at the statistics
print("mean", scores.mean())
print("sd", scores.std())
