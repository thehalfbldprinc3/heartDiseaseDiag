import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import eli5

from sklearn.linear_model import LogisticRegression as logr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from eli5.sklearn import PermutationImportance

data = pd.read_excel(r'../Dataset/heart_dataset_complete.xlsx')
data = data.drop(["age", "fbs", "trestbps", "chol", "restecg"], axis=1)

y_df = data.target.values.ravel()
x_df = data.drop(['target'], axis=1)

x = np.array(x_df)
y = np.array(y_df)

ls = ['LR1', 'LR2', 'LR3', 'LR4', 'LR5']
cmResults = pd.DataFrame(columns=ls)
clResults = pd.DataFrame(columns=ls)

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

kf = KFold(5, shuffle=True)

for a in range(5):
    sol = solvers[a]
    print("Solver type is:", sol)
    l = ls[a]

    cm_list = []
    cl_list = []

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        norm = MinMaxScaler()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)

        model = logr(solver=sol)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        cm_LR = confusion_matrix(Y_test, Y_pred)
        LR_cm = (cm_LR[1][0], cm_LR[0][0], cm_LR[0][1], cm_LR[1][1])
        cm_list.append(LR_cm)

        result_LR = (
            accuracy_score(Y_test, Y_pred),
            precision_score(Y_test, Y_pred),
            recall_score(Y_test, Y_pred),
            f1_score(Y_test, Y_pred)
        )
        cl_list.append(result_LR)

        perm = PermutationImportance(model, random_state=1).fit(X_test, Y_test)
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=x_df.columns.tolist())))

    cm_avg = np.mean(cm_list, axis=0)
    cl_avg = np.mean(cl_list, axis=0)
    cmResults[l] = cm_avg
    clResults[l] = cl_avg

    print("Average Confusion Matrix:")
    print(cm_avg)
    print("Average Classification Results:")
    print(cl_avg)

    ax = plt.subplot()
    cm_2x2 = [[round(cm_avg[1]), round(cm_avg[2])], [round(cm_avg[0]), round(cm_avg[3])]]
    sn.heatmap(cm_2x2, annot=True, cmap="Blues")

    ax.set_xlabel('Classifier Prediction')
    ax.set_ylabel('True Value')
    ax.set_title('Average Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.show()