pip install scikit-learn

import pandas as pd # data manipulation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score # evaluation metrics
from sklearn.model_selection import train_test_split, GridSearchCV # split and tuning
from sklearn.ensemble import AdaBoostClassifier # AdaBoost model
from sklearn.tree import DecisionTreeClassifier # base estimator

# data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.read_csv(url, names = columns) # load dataset

# data preparation
y = df["Outcome"]
X = df.drop("Outcome", axis = 1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# base learner (DT) - weak learner (stump)
base_model = DecisionTreeClassifier(max_depth = 1, random_state = 42)

# model fitting
ada_model = AdaBoostClassifier(
    estimator = base_model,
    random_state = 42
).fit(X_train, y_train)

# prediction
y_pred = ada_model.predict(X_test)
y_prob = ada_model.predict_proba(X_test)[:, 1]

# metrics (accuracy & ROC-AUC)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# hyperparameter grid
ada_params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1],
    "estimator__max_depth": [1, 2, 3],
    "estimator__min_samples_split": [2, 5, 10]
}

# grid search (with ROC-AUC optimization)
ada_cv_model = GridSearchCV(
    ada_model,
    ada_params,
    cv = 10,
    scoring = "roc_auc",
    verbose = 2,
    n_jobs = -1
).fit(X_train, y_train)

# best parameters
print("Best Parameters:", ada_cv_model.best_params_)

# tuned model
ada_tuned = AdaBoostClassifier(
    estimator = DecisionTreeClassifier(
        max_depth = ada_cv_model.best_params_["estimator__max_depth"],
        min_samples_split = ada_cv_model.best_params_["estimator__min_samples_split"],
        random_state = 42
    ),
    n_estimators = ada_cv_model.best_params_["n_estimators"],
    learning_rate = ada_cv_model.best_params_["learning_rate"],
    random_state = 42
).fit(X_train, y_train)

# tuned prediction
y_pred2 = ada_tuned.predict(X_test)
y_prob2 = ada_tuned.predict_proba(X_test)[:, 1]

# tuned metrics
accuracy2 = accuracy_score(y_test, y_pred2)
roc_auc2 = roc_auc_score(y_test, y_prob2)

print("Tuned Accuracy:", accuracy2)
print("Tuned ROC-AUC:", roc_auc2)

# confusion matrix
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print("Tuned Confusion Matrix:\n", conf_matrix2)