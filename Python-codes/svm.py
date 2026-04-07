from sklearn.datasets import load_digits # data set
from sklearn.preprocessing import StandardScaler # scaling
from sklearn.model_selection import train_test_split # data set splitting
from sklearn.svm import SVC # model fitting
from sklearn.metrics import classification_report # accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay # confusion matrix visualization
from sklearn.model_selection import GridSearchCV # model tuning
import matplotlib.pyplot as plt # data visualization (images)

# data set
digits = load_digits()

# data visualization (images)
fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10, 5),
                         subplot_kw = {"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(digits.target[i])
plt.show()

# data manipulation
X = digits.data
y = digits.target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
svm_class_model = SVC(random_state = 42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_class_model.fit(X_train_scaled, y_train)

# y_pred
y_pred = svm_class_model.predict(X_test_scaled)

# accuracy
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()

# model tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
    }

grid_svc = GridSearchCV(svm_class_model, param_grid, cv = 5, scoring = 'accuracy')
grid_svc.fit(X_train_scaled, y_train)

# best parameters
print("Best Parameters:", grid_svc.best_params_)

# final model with the best parameters
## model fitting
final_svc = grid_svc.best_estimator_
final_svc.fit(X_train_scaled, y_train)

## y_pred2
y_pred2 = final_svc.predict(X_test_scaled)

## accuracy
print(classification_report(y_test, y_pred2))

# confusion matrix 2
cm2 = confusion_matrix(y_test, y_pred2)
disp2 = ConfusionMatrixDisplay(confusion_matrix = cm2)
disp2.plot()
plt.show()

# bonus: accuracy versus C (visualization)
import numpy as np

## C values
C_values = np.logspace(-2, 2, 10)  # [0.01, 0.026, ..., 100]

## lists to store accuracy scores
train_scores = []
test_scores = []

## train and evaluate the model for each C value
for c in C_values:
    model = SVC(C = c, kernel = "poly", gamma = "scale", random_state = 42)
    model.fit(X_train_scaled, y_train)
    
    ### calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

## plotting
plt.figure(figsize = (10, 6))
plt.plot(C_values, train_scores, marker = 'o', label = 'Train Accuracy')
plt.plot(C_values, test_scores, marker = 's', label = 'Test Accuracy')
plt.xscale('log')
plt.xlabel('C Value (log scale)')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy vs C (Kernel = poly)')
plt.legend()
plt.grid(True)
plt.show()
