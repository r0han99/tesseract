# Import Packages
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import sc
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
crimes = pd.read_csv('./Crime+Corpus/crimes_binary_v1.csv').drop('Unnamed: 0',axis=1)

# Features and Target
features = ['Month','Day', 'Description', 'Location Description',
            'Community Area','Domestic','Primary Type','District','SocioEconomic-Status']
target = 'Arrest'

# Selecting Features and Making some changes to the feature list
df = crimes[features]
df = df.drop(['Description','District'],axis=1)

# Slicing the Data to contain only these candidates
df = df[df['Primary Type'].isin(['ASSAULT','THEFT','BATTERY'])]
df = df[df['Location Description'].isin(['STREET','APARTMENT','RESIDENCE'])]
df[target] = crimes[target]

# One Hot Encoding 
dum_df = pd.get_dummies(df,columns=['Location Description','Primary Type','SocioEconomic-Status'],drop_first=True)

# Downsampling - Balancing
false_df = dum_df[dum_df[target]==False].sample(n=32420)
true_df = dum_df[dum_df[target]==True]
model_df = pd.concat([false_df, true_df],axis=0)


X = model_df.drop(target,axis=1)
y = model_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Naive Bayes classifier
nb_clf = GaussianNB()

# Fit Naive Bayes classifier to training data
nb_clf.fit(X_train, y_train)

# Make predictions on testing data
y_pred = nb_clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Train Accuracy: {nb_clf.score(X_train, y_train)}")
print(f"Test Accuracy: {accuracy}")

# Confusion Matrix
sns.heatmap(confusion_matrix(y_pred, y_test),cmap="Blues",annot=True,fmt='d'
            ,xticklabels=nb_clf.classes_,yticklabels=nb_clf.classes_)
plt.title('Confusion Matrix, Target=Arrest')
plt.show()

# Classification Report
print(classification_report(y_pred,y_test))

