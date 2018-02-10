# Load required packages
library(caret)
library(gbm)
library(randomForest)
library(rpart)

# Load the data
data(spam, package = "kernlab") #<<

# Partition the data into train/test sets
set.seed(101)  # for reproducibility
trn_id <- createDataPartition(spam$type, p = 0.7, list = FALSE)
trn <- spam[trn_id, ]                # training data
tst <- spam[-trn_id, ]               # test data
xtrn <- subset(trn, select = -type)  # training data features
xtst <- subset(tst, select = -type)  # test data features
ytrn <- trn$type                     # training data response
ytst <- tst$type                     # test data response

# Function to calculate accuracy
accuracy <- function(pred, obs) {
  sum(diag(table(pred, obs))) / length(obs)
}


# Classification tree ----------------------------------------------------------

# Tune a classification tree
ctrl <- trainControl(method = "cv", number = 10)  # Control values for training
tuned_tree <- train(
  type ~ ., 
  data = spam,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = data.frame(cp = seq(from = 0, to = 0.05, by = 0.001))
)
tuned_tree

# Fit a classification tree (cp found using k-fold CV)
spam_tree <- rpart(type ~ ., data = trn, cp = 0.001) #<<
pred <- predict(spam_tree, newdata = xtst, type = "class")

# Compute test set accuracy
(spam_tree_acc <- accuracy(pred = pred, obs = tst$type))


# Bagged classification trees --------------------------------------------------

# Fit a bagger model
set.seed(1633)  # for reproducibility
spam_bag <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 1000,
  mtry = ncol(xtrn),  # use all available features #<<
  xtest = subset(tst, select = -type),
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_bag, newdata = xtst, type = "class")
spam_bag_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
dark2 <- RColorBrewer::brewer.pal(8, "Dark2")
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_bag$ntree), spam_bag$test$err.rate[, "Test"], type = "l", 
     col = dark2[1L], ylim = c(0.04, 0.11), las = 1,
     ylab = "Test error", xlab = "Number of trees")
abline(h = 1 - spam_tree_acc, lty = 2, col = "black")
abline(h = 1 - spam_bag_acc, lty = 2, col = dark2[1L])
legend("topright", c("Single tree", "Bagging"),
       col = c("black", dark2[1L]), lty = c(2, 1), lwd = 1)


# Random forest ----------------------------------------------------------------

# Fit a random forest
set.seed(1633)  # for reproducibility
spam_rf <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 1000,
  mtry = 7,  # floor(sqrt(p))  #<<
  xtest = subset(tst, select = -type),
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_rf, newdata = xtst, type = "class")
spam_rf_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_rf$ntree), spam_rf$test$err.rate[, "Test"], type = "l", 
     col = dark2[4L], ylim = c(0.04, 0.11), 
     ylab = "Test error", xlab = "Number of trees")
lines(seq_len(spam_rf$ntree), spam_bag$test$err.rate[, "Test"], col = dark2[1L])
abline(h = 1 - spam_tree_acc, lty = 2, col = "black")
abline(h = 1 - spam_bag_acc, lty = 2, col = dark2[1L])
abline(h = 1 - spam_rf_acc, lty = 2, col = dark2[4L])
legend("topright", c("Single tree", "Bagging", "Random forest"),
       col = c("black", dark2[c(1, 4)]), lty = c(2, 1, 1), lwd = 1)


# Gradient boosting machines ---------------------------------------------------

trn_label <- ifelse(trn$type == "spam", 1, 0)
tst_label <- ifelse(tst$type == "spam", 1, 0)
dtrn <- xgb.DMatrix(data.matrix(xtrn), label = trn_label)
dtst <- xgb.DMatrix(data.matrix(xtst), label = tst_label)
set.seed(1952)
spam_xgb <- xgb.train(
  params = list(max_depth = 6, eta = 0.01), 
  data = dtrn, 
  nrounds = 10000, 
  watchlist = list(eval = dtst),
  objective = "binary:logistic", 
  eval_metric = "error", 
  early_stopping_round = NULL#100
)
pred <- predict(spam_xgb, newdata = dtst)
pred <- ifelse(pred >= 0.5, "spam", "nonspam")
(spam_xgb_acc <- accuracy(pred = pred, obs = tst$type))
plot(spam_xgb$evaluation_log, type = "l")

# Plot test error
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_rf$ntree), spam_rf$test$err.rate[, "Test"], type = "l", 
     col = dark2[4L], ylim = c(0.04, 0.11), 
     ylab = "Test error", xlab = "Number of trees")
lines(seq_len(spam_rf$ntree), spam_bag$test$err.rate[, "Test"], col = dark2[1L])
lines(spam_xgb$evaluation_log, col = dark2[5L])
abline(h = 1 - spam_tree_acc, lty = 2, col = "black")
abline(h = 1 - spam_bag_acc, lty = 2, col = dark2[1L])
abline(h = 1 - spam_rf_acc, lty = 2, col = dark2[4L])
abline(h = 1 - spam_xgb_acc, lty = 2, col = dark2[5L])
legend("topright", c("Single tree", "Bagging", "Random forest", "XGBoost"),
       col = c("black", dark2[c(1, 4, 5)]), lty = c(2, 1, 1, 1), lwd = 1)

# Fit a random forest
set.seed(1928)  # for reproducibility
spam_gbm <- gbm(
  ifelse(type == "spam", 1, 0) ~ ., 
  data = trn, 
  distribution = "bernoulli",
  n.trees = 10000,
  interaction.depth = 5,
  shrinkage = 0.01,
  bag.fraction = 0.7,
  train.fraction = 0.7,
  # cv.folds = 5,
  verbose = TRUE
)

# Optimal number of iterations
best_iter <- gbm.perf(spam_gbm, method = "test")

# Compute test error
pred <- predict(spam_gbm, newdata = xtst, type = "response", 
                n.trees = best_iter)
pred <- ifelse(pred >= 0.5, "spam", "nonspam")
(spam_gbm_acc <- accuracy(pred = pred, obs = tst$type))
