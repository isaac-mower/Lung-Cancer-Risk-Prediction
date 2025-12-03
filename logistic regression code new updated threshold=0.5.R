library(glmnet)
library(verification)
library(pROC)

cancer_df <- read.csv("Lung Cancer Dataset.csv")

# Clean data
cancer_df$NEW_PULMONARY_DISEASE <- ifelse(cancer_df$PULMONARY_DISEASE == "YES", 1, 0)
cancer_df <- cancer_df[ , -18]

# Split data: 60/20/20
set.seed(123)
n <- nrow(cancer_df)
train_index <- sample(seq_len(n), size = 0.6 * n)

remaining_index <- setdiff(seq_len(n), train_index)
validation_index <- sample(remaining_index, size = 0.5 * length(remaining_index))
test_index <- setdiff(remaining_index, validation_index)

train_data <- cancer_df[train_index, ]
validation_data <- cancer_df[validation_index, ]
test_data <- cancer_df[test_index, ]

# Create matrices
train_x <- as.matrix(train_data[ , 1:17])
train_y <- as.matrix(train_data[ , 18])

validation_x <- as.matrix(validation_data[ , 1:17])
validation_y <- as.matrix(validation_data[ , 18])

test_x <- as.matrix(test_data[ , 1:17])
test_y <- as.matrix(test_data[ , 18])

# Fit logistic regression model on training data
loreg_cancer <- glmnet(train_x, train_y, family = "binomial")

# Plot coefficient paths
plot(loreg_cancer)

# MANUAL tuning using validation set
lambda_seq <- loreg_cancer$lambda

validation_preds <- predict(loreg_cancer, newx = validation_x, type = "response")

validation_mse <- apply(validation_preds, 2, function(pred) {
  mean((pred - validation_y)^2)
})

# Find best lambdas
lambda_min <- lambda_seq[which.min(validation_mse)]

mse_min <- min(validation_mse)
mse_1se_threshold <- mse_min + sd(validation_mse)
lambda_1se <- max(lambda_seq[validation_mse <= mse_1se_threshold])

# Predictions on test set using lambda.min
pred_class_min <- predict(
  loreg_cancer, newx = test_x,
  s = lambda_min, type = "class"
)

conf_mat_min <- table(
  actual = as.numeric(test_y),
  predicted = as.numeric(pred_class_min)
)

prob_min <- predict(
  loreg_cancer, newx = test_x,
  s = lambda_min, type = "response"
)

auc_min <- roc.area(as.numeric(test_y), prob_min)$A

# Predictions on test set using lambda.1se
pred_class_1se <- predict(
  loreg_cancer, newx = test_x,
  s = lambda_1se, type = "class"
)

conf_mat_1se <- table(
  actual = as.numeric(test_y),
  predicted = as.numeric(pred_class_1se)
)

prob_1se <- predict(
  loreg_cancer, newx = test_x,
  s = lambda_1se, type = "response"
)

auc_1se <- roc.area(as.numeric(test_y), prob_1se)$A

# Function to calculate performance metrics
get_metrics <- function(conf_mat, pred_prob, actual) {
  TP <- conf_mat[2, 2]
  TN <- conf_mat[1, 1]
  FP <- conf_mat[1, 2]
  FN <- conf_mat[2, 1]
  
  accuracy <- (TP + TN) / sum(conf_mat)
  recall <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  mse <- mean((pred_prob - actual)^2)
  
  list(
    Accuracy = accuracy,
    Recall = recall,
    Specificity = specificity,
    Precision = precision,
    F1_Score = f1_score,
    MSE = mse
  )
}

# Calculate metrics
metrics_min <- get_metrics(
  conf_mat_min,
  as.numeric(prob_min),
  as.numeric(test_y)
)

metrics_1se <- get_metrics(
  conf_mat_1se,
  as.numeric(prob_1se),
  as.numeric(test_y)
)

# Print results
print("Metrics for lambda.min:")
print(metrics_min)
print(paste("ROC AUC for lambda.min:", auc_min))

print("Metrics for lambda.1se:")
print(metrics_1se)
print(paste("ROC AUC for lambda.1se:", auc_1se))

# Model coefficients
coef_min <- coef(loreg_cancer, s = lambda_min)
coef_1se <- coef(loreg_cancer, s = lambda_1se)

coef_min
coef_1se


# Plot Validation MSE vs Log(Lambda)
plot(
  log(lambda_seq), validation_mse,
  type = "l",
  col = "blue",
  lwd = 2,
  xlab = "Log(Lambda)",
  ylab = "Validation MSE",
  main = "Validation MSE vs Log(Lambda)"
)

# Add vertical lines for best lambdas
abline(v = log(lambda_min), col = "red", lty = 2)    # Best lambda (min)
abline(v = log(lambda_1se), col = "black", lty = 2)  # 1SE rule
legend(
  "topright",
  legend = c("Lambda.min", "Lambda.1se"),
  col = c("red", "black"),
  lty = 2,
  cex = 0.8
)


# Plot confusion matrix for lambda.min
fourfoldplot(
  conf_mat_min,
  color = c("red", "blue"),
  conf.level = 0,
  margin = 1,
  main = "Confusion Matrix for Lambda.min"
)

# Plot confusion matrix for lambda.1se
fourfoldplot(
  conf_mat_1se,
  color = c("red", "blue"),
  conf.level = 0,
  margin = 1,
  main = "Confusion Matrix for Lambda.1se"
)


# ROC curve for lambda.min
roc_obj_min <- roc(as.numeric(test_y), as.numeric(prob_min))

plot(
  roc_obj_min,
  col = "blue",
  lwd = 2,
  main = "ROC Curve for Lambda.min"
)
abline(a = 0, b = 1, lty = 2, col = "black")  # Add reference diagonal
