import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import numpy as np


###########################
     # Load Data #
###########################
print("\nLoading data.....")
data = datasets.load_digits().images.reshape((len(datasets.load_digits().images), -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, datasets.load_digits().target, test_size=0.5, shuffle=False
)

###########################
    # Train and Predict #
###########################
print("\nTraining and predicting using Decision Tree Classifier.....")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###########################
    # Train and Predict #
###########################
print("\nTraining and predicting using Random Forest Classifier.....")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
