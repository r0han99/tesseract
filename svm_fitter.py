from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd 


def estimate_svm(data, kernel):

    print('Data Read: ', data.shape())

    # select features and target variable
    features = ['Block','Primary Type', 'Description','Community Area','Location Description','SocioEconomic-Status','dayofweek']
    target = 'Arrest'


    X = data[features]
    y = data[target]   

    print(f"Features {features}")
    print(f'Class: {target}')

    print(f'Class Balance: {y.value_counts()}')

    smote_enn = SMOTEENN(random_state=42,)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    print('- Undersampling Data.')


    # print the class distribution before and after undersampling
    print('Class distribution before undersampling:')
    print(pd.Series(y).value_counts())

    print('Class distribution after undersampling:')
    print(pd.Series(y_resampled).value_counts())

    print('Scaling Numericals')
    # create a StandardScaler object
    scaler = StandardScaler()

    # fit the scaler on the data
    scaler.fit(X)

    # transform the data using the scaler
    X_scaled = scaler.transform(X_resampled)

    print(f'Creating Distributions Train: 70, Test: 30')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=101)


    print(f'Fitting SVM with Kernel: {kernel}')
    for c in [0.1, 1.0, 5, 10]:

        svm_clf = SVC(kernel=kernel,C=c)

        svm_clf.fit(X_train, y_train)

        y_pred = svm_clf.predict(X_test)

        print("Train Accuracy: ", svm_clf.score(X_train,y_train))
        print("Test Accuracy: ", svm_clf.score(X_test,y_test))
        print("***"*25)
        print(classification_report(y_pred, y_test))
        print("***"*25)
        sns.heatmap(confusion_matrix(y_pred, y_test),cmap="Greens",annot=True,fmt='d'
                    ,xticklabels=svm_clf.classes_,yticklabels=svm_clf.classes_)
        plt.title(f'Confusion Matrix, SVM kernel:{kernel}, C={c}, Target=Arrest')
        plt.show()
        

