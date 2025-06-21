import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import eli5

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from eli5.sklearn import PermutationImportance

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode='uniform',
                 activation='relu', dropout_rate=0.0, weight_constraint=0,
                 neurons1=1, neurons2=1):
    model = Sequential()
    model.add(Dense(neurons1, input_dim=8, kernel_initializer=init_mode,
                    activation=activation, kernel_constraint=max_norm(weight_constraint)))
    model.add(Dense(neurons2, kernel_initializer=init_mode,
                    activation=activation, kernel_constraint=max_norm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


dataframe = pd.read_excel(r'../Dataset/heart_dataset_complete.xlsx')
dataframe = dataframe.drop(["age", "fbs", "trestbps", "chol", "restecg"], axis=1)

y_df = dataframe.target.values.ravel()
x_df = dataframe.drop(['target'], axis=1)
x = np.array(x_df)
y = np.array(y_df)

kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

model = KerasClassifier(model=create_model, verbose=0)

param_grid = {
    'batch_size': [10, 20],
    'epochs': [50],
    'optimizer': ['adam'],
    'learn_rate': [0.001],
    'momentum': [0.0],
    'init_mode': ['uniform'],
    'activation': ['relu'],
    'dropout_rate': [0.2],
    'weight_constraint': [2],
    'neurons1': [16],
    'neurons2': [8]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = create_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, batch_size=10, verbose=10)

plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

Y_pred = np.round(model.predict(X_test)).astype(int)

print('Results for Model')
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Sick', 'Sick'])
ax.yaxis.set_ticklabels(['Not Sick', 'Sick'])
b, t = plt.ylim()
plt.ylim(b + 0.5, t - 0.5)
plt.show()

perm = PermutationImportance(grid.best_estimator_, random_state=1).fit(X_test, Y_test.ravel())
print(eli5.explain_weights(perm, feature_names=x_df.columns.tolist()))