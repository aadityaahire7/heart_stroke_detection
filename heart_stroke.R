#Installing packages
install.packages("pROC")
install.packages("xgboost")
install.packages("DiagrammeR")
install.packages("plotly")
install.packages("Metrics")
install.packages("tensorflow")
install.packages("keras")
install.packages("caret")
install.packages("ROSE")
install.packages("magrittr")
install.packages("dplyr")

install.packages("microbenchmark")
install.packages("pryr")
install.packages("reshape2")
# Importing packages
library(plotly)
library(tensorflow)
library(xgboost)
library(pROC)
library(tidyverse)
library(naniar)
library(caTools) 
library(ggplot2) 
library(superheat) 
library(scatterplot3d) 
library(ROCR)
library(Metrics)
library(keras)
library(caret)
library(ROSE)
library(reshape2)
library(magrittr)
library(dplyr)


#reading the csv file
data = read.csv("C:\\Users\\Aaditya Ahire\\Desktop\\rML\\healthcare-dataset-stroke-data.csv")
str(data)


# Converting character values to numeric values

clean_data <- data %>% mutate(gender = if_else(gender == "Female", 0, if_else(gender == "Male", 1, 2)), ever_married = if_else(ever_married == "Yes", 1, 0), Residence_type = if_else(Residence_type == "Rural", 0, 1), smoking_status = if_else(smoking_status == "never smoked", 0, if_else(smoking_status == "formerly smoked", 1, if_else(smoking_status == "smokes", 2, 3))))
summary(clean_data)
glimpse(clean_data)

# Handling missing values

miss_scan_count(data = data, search = list("N/A", "Unknown"))

# There are 201 "N/A" values in the bmi column that likely caused this column 
# to be parsed as character, although it should be numerical.   
#  replacing those values with actual NAs. 
# lot of "Unknown" values in smoking_status  
#  We see that we have 1544 unknown values for smoking status and 
# replace those values with 
# NAs.

clean_data <- replace_with_na(data = clean_data, replace = list(bmi = c("N/A"), smoking_status = c(3))) %>% mutate(bmi = as.numeric(bmi))



# Split the data into training and testing sets
set.seed(99)  # Set a seed for reproducible results always producing same random result
split <- sample.split(clean_data$stroke, SplitRatio = 0.8)
train <- subset(clean_data, split == TRUE)
test <- subset(clean_data, split == FALSE)

# Convert data to a format suitable for xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train[, -1]), label = train$stroke)
dtest <- xgb.DMatrix(data = as.matrix(test[, -1]), label = test$stroke)

# Define and train the xgboost model , Parameter Tuning
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.001491,  # Learning rate
  max_depth = 5,  # Maximum depth of trees
  subsample = 0.7,  # Fraction of training data used for each tree
  colsample_bytree = 0.7  # Fraction of features used for each tree
)
#xgboost Model Training
xgb_model <- xgboost(data = dtrain, params = xgb_params, nrounds = 915, early_stopping_rounds = 50, verbose = 1)

predict_test_xgb <- predict(xgb_model, dtest, type = "response")

predictions <- ifelse(predict_test_xgb > 0.699, 1, 0)
confusion_matrix <- table(test$stroke, predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
# Find the threshold that maximizes accuracy


cat("Best Accuracy:", accuracy*100, "\n")

# Check the dimensions
predictions <- as.data.frame(predictions)
predictions <- predictions[-c(1014:1022), ]
predictions <- as.data.frame(predictions)

dim(predictions)

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
set.seed(99)
index <- createDataPartition(clean_data$stroke, p = 0.8, list = FALSE)
train_data <- clean_data[index, ]
test_data <- clean_data[-index, ]

# Normalize numerical features
scaler <- preProcess(train_data[c("age", "avg_glucose_level", "bmi")], method = c("center", "scale"))
train_data[c("age", "avg_glucose_level", "bmi")] <- predict(scaler, train_data[c("age", "avg_glucose_level", "bmi")])
test_data[c("age", "avg_glucose_level", "bmi")] <- predict(scaler, test_data[c("age", "avg_glucose_level", "bmi")])

# Handle missing values (if any)
train_data[is.na(train_data)] <- 0
test_data[is.na(test_data)] <- 0
# Update the input shape in your LSTM layer to match your data

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50, input_shape = c(10, 9), return_sequences = TRUE) %>%
  layer_lstm(units = 50, return_sequences = TRUE) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")

# ... (other parts of your code remain the same)

# Prepare data for LSTM
sequence_length <- 10
n_features <- ncol(train_data) - 1  # Number of features, excluding the target variable

# Initialize x_train and y_train with the correct dimensions
x_train <- array(0, dim = c(nrow(train_data) - sequence_length + 1, sequence_length, n_features))
y_train <- rep(0, length = nrow(train_data) - sequence_length + 1)

# Populate x_train and y_train
for (i in 1:(nrow(train_data) - sequence_length + 1)) {
  x_train[i,,] <- as.matrix(train_data[i:(i + sequence_length - 1), -10])  # Use all features (excluding the target variable)
  y_train[i] <- train_data[i + sequence_length - 1, 10]
}

# Initialize x_test and y_test with the correct dimensions
x_test <- array(0, dim = c(nrow(test_data) - sequence_length + 1, sequence_length, n_features))
y_test <- rep(0, length = nrow(test_data) - sequence_length + 1)

# Populate x_test and y_test
for (i in 1:(nrow(test_data) - sequence_length + 1)) {
  x_test[i,,] <- as.matrix(test_data[i:(i + sequence_length - 1), -10])  # Use all features (excluding the target variable)
  y_test[i] <- test_data[i + sequence_length - 1, 10]
}
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001491),
  metrics = "accuracy"
)
# Train the model
model %>% fit(x_train, y_train, epochs = 20, batch_size = 32)
# Evaluate the model on the test data
# Evaluate the model on the test data
evaluation <- model %>% evaluate(x_test, y_test)

# Extract the accuracy from the evaluation results
accuracy <- evaluation[[2]]  # Assuming accuracy is the first metric

# Print the accuracy percentage
cat("Accuracy: ", accuracy * 100, "%\n")


# Make predictions using the LSTM model
lstm_predictions <- model %>% predict(x_test)



# Check the dimensions of prediction vectors
dim(lstm_predictions)
ensemble_predictions <- (lstm_predictions + predictions) / 2


threshold <- 0.699

# 3. Calculate accuracy
ensemble_predictions_binary <- ifelse(ensemble_predictions > threshold, 1, 0)
true_labels <- test_data[(sequence_length):nrow(test_data), "stroke"]

# Calculate accuracy
accuracy <- mean(ensemble_predictions_binary == true_labels)

# Print the accuracy
cat("Ensemble Model Accuracy: ", accuracy * 100, "%\n")


# confusion metrics
# Confusion matrix for XGBoost model

# Calculate precision, recall, and F1 score for XGBoost model
precision_xgboost <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall_xgboost <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score_xgboost <- 2 * (precision_xgboost * recall_xgboost) / (precision_xgboost + recall_xgboost)

cat("XGBoost Model Precision:", precision_xgboost, "\n")
cat("XGBoost Model Recall:", recall_xgboost, "\n")
cat("XGBoost Model F1 Score:", f1_score_xgboost, "\n")

# Calculate precision, recall, and F1 score for LSTM model
precision_lstm <- sum(ensemble_predictions_binary[true_labels == 1] == 1) / sum(ensemble_predictions_binary == 1)
recall_lstm <- sum(ensemble_predictions_binary[true_labels == 1] == 1) / sum(true_labels == 1)
f1_score_lstm <- 2 * (precision_lstm * recall_lstm) / (precision_lstm + recall_lstm)

cat("LSTM Model Precision:", precision_lstm, "\n")
cat("LSTM Model Recall:", recall_lstm, "\n")
cat("LSTM Model F1 Score:", f1_score_lstm, "\n")

# Calculate precision, recall, and F1 score for ensemble model
precision_ensemble <- sum(ensemble_predictions_binary[true_labels == 1] == 1) / sum(ensemble_predictions_binary == 1)
recall_ensemble <- sum(ensemble_predictions_binary[true_labels == 1] == 1) / sum(true_labels == 1)
f1_score_ensemble <- 2 * (precision_ensemble * recall_ensemble) / (precision_ensemble + recall_ensemble)

cat("Ensemble Model Precision:", precision_ensemble, "\n")
cat("Ensemble Model Recall:", recall_ensemble, "\n")
cat("Ensemble Model F1 Score:", f1_score_ensemble, "\n")

