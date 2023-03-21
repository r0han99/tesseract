# Import Packages
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import sc
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# Load Data
crimes = pd.read_csv('./Crime+Corpus/crimes_binary_v1.csv').drop('Unnamed: 0',axis=1)

# Required Feature List 
features = ['Month','Day', 'Description', 'Location Description','Community Area','Domestic','Primary Type','District','SocioEconomic-Status']
target = 'Arrest'

# Factorizing Categorical Variables
crimestfac = crimes.copy(deep=True)
for feature in features:
    if crimestfac[feature].dtype == 'object':
        crimestfac[feature] = pd.factorize(crimestfac[feature])[0]
   
factorized_crimes = crimestfac[features].copy(deep=True)
factorized_crimes.head(12)

# Balancing Classes - Down Sampling
false_df = factorized_crimes[factorized_crimes[target]==False].sample(n=194889)
true_df = factorized_crimes[factorized_crimes[target]==True]
model_df = pd.concat([false_df, true_df],axis=0)

# Creating Distributions 
X = model_df.drop('Arrest',axis=1)
y = model_df['Arrest']

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, random_state=42)

# create decision tree classifier
dt_clf = DecisionTreeClassifier(criterion='entropy',max_depth=9,min_samples_leaf=4)

# fit decision tree classifier to training data
dt_clf.fit(X_train, y_train)

# make predictions on testing data
y_pred = dt_clf.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {dt_clf.score(X_train, y_train)}")
print(f"Test Accuracy: {accuracy}")


# Confusion Matrix
sns.heatmap(confusion_matrix(y_pred, y_test),cmap="Blues",annot=True,fmt='d'
            ,xticklabels=dt_clf.classes_,yticklabels=dt_clf.classes_)
plt.title('Confusion Matrix, Target=Arrest')
plt.show()

# Classification Report
print(classification_report(y_pred,y_test))


# Create Graph Visualisation

# create dot file with max depth of 3
dot_data = export_graphviz(dt_clf, out_file=None, feature_names=X.columns, class_names=['No Arrest', 'Arrest'], filled=True, rounded=True, special_characters=True, max_depth=4)

# create graph from dot file
graph = graphviz.Source(dot_data)

# show graph
graph.view()