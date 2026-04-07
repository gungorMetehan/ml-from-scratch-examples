library(e1071) # SVM implementation in R

# missing values?
anyNA(iris) # check missing values

# making this example reproducible
set.seed(12)

# train - test split (80% - 20%)
index <- sample(seq_len(nrow(iris)), size = 0.8 * nrow(iris))
iris_train <- iris[index, ]   # training data
iris_test  <- iris[-index, ]  # test data

# svm modeling
svm_model <- svm(
  Species ~ .,
  data   = iris_train,
  kernel = "radial",   # nonlinear decision boundary
  cost   = 1,          # regularization parameter
  gamma  = 0.25,       # kernel influence
  scale  = TRUE        # feature scaling
)

# predict test labels
results <- predict(object = svm_model, newdata = iris_test)

# confusion matrix
conf_mat <- table(results, iris_test$Species)
conf_mat

# classification accuracy (100%)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
accuracy
