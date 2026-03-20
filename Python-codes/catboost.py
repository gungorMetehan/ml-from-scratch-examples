pip install catboost

import pandas as pd # data frame operations
from sklearn.metrics import accuracy_score, confusion_matrix # accuracy scores
from sklearn.model_selection import train_test_split, GridSearchCV # for train-test split & grid search
from catboost import CatBoostClassifier # for fitting

# data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, names=columns)

# data preparation
y = df["Outcome"]
X = df.drop("Outcome", axis = 1)

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
cat_model = CatBoostClassifier(verbose = 0, random_state = 42).fit(X_train, y_train)

# prediction
y_pred = cat_model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# hyperparameter grid
cat_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "iterations": [100, 200, 500],     # n_estimators
    "depth": [3, 5, 7],                # max_depth
    "l2_leaf_reg": [1, 3, 5]           # regularization
}

# grid search
cat_cv_model = GridSearchCV(
    cat_model,
    cat_params,
    cv = 10,
    verbose = 2,
    n_jobs = -1
).fit(X_train, y_train)

# best parameters
print("Best Parameters:", cat_cv_model.best_params_)

# tuned model
cat_tuned = CatBoostClassifier(
    learning_rate = cat_cv_model.best_params_["learning_rate"],
    iterations = cat_cv_model.best_params_["iterations"],
    depth = cat_cv_model.best_params_["depth"],
    l2_leaf_reg = cat_cv_model.best_params_["l2_leaf_reg"],
    verbose = 0,
    random_state = 42).fit(X_train, y_train)

# tuned predictions
y_pred2 = cat_tuned.predict(X_test)

# tuned accuracy
accuracy2 = accuracy_score(y_test, y_pred2)
print("Tuned Accuracy:", accuracy2)

# feature importance plot
import matplotlib.pyplot as plt

importance = cat_tuned.get_feature_importance()

feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending = False)

plt.figure(figsize = (8, 6))
plt.barh(feature_imp["Feature"], feature_imp["Importance"])
plt.xlabel("Importance Score")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()