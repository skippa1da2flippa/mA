from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from fetchedData import X, y


def main():
    # split the data between training and testing data (90-10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

    print(X_test)


if __name__ == "__main__":
    main()
