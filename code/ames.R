# Load required packages
library(ggplot2)
library(pdp)
library(vip)
library(xgboost)

# Construct data set
ames <- AmesHousing::make_ames()

# Feature matrix  # or xgb.DMatrix or sparse matrix
X <- data.matrix(subset(ames, select = -Sale_Price))

# Use k-fold cross-validation to find the "optimal" number of trees
set.seed(1214)  # for reproducibility
ames_xgb_cv <- xgb.cv(
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",
  nrounds = 10000, 
  max_depth = 5, 
  eta = 0.01, 
  subsample = 1,          
  colsample = 1,          
  num_parallel_tree = 1,  
  eval_metric = "rmse",   
  early_stopping_rounds = 50,
  verbose = 0,
  nfold = 5
)

# Plot cross-validation results
plot(test_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, type = "l", 
     ylim = c(0, 200000), xlab = "Number of trees", ylab = "RMSE")
lines(train_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, col = "red2")
abline(v = ames_xgb_cv$best_iteration, lty = 2)
legend("topright", legend = c("Train", "CV"), lty = 1, col = c("red2", 1),
       inset = 0.15)
print(ames_xgb_cv$best_iteration)  # 2771

# Fit an XGBoost model
set.seed(203)  # for reproducibility
ames_xgb <- xgboost(
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",
  nrounds = ames_xgb_cv$best_iteration, 
  max_depth = 5, 
  eta = 0.01, 
  subsample = 1,          
  colsample = 1,          
  num_parallel_tree = 1,  
  eval_metric = "rmse",   
  verbose = 0
)

# Variable importance plots
p1 <- vip(ames_xgb, pred.var = colnames(X), type = "Gain")
p2 <- vip(ames_xgb, pred.var = colnames(X), type = "Cover")
p3 <- vip(ames_xgb, pred.var = colnames(X), type = "Frequency")
grid.arrange(p1, p2, p3, ncol = 3)

# By default, `vip()` plots 10 most important features
vip(ames_xgb, pred.var = colnames(X), type = "Gain", 
    top_n = nrow(X), bar = FALSE)

# Partial dependence plots
oq_ice <- partial(ames_xgb, pred.var = "Overall_Qual", ice = TRUE, 
                  center = TRUE, train = X)
p4 <- autoplot(partial(ames_xgb, pred.var = "Gr_Liv_Area", train = X))
p5 <- autoplot(partial(ames_xgb, pred.var = "Garage_Cars", train = X))
p6 <- autoplot(oq_ice, alpha = 0.1)
grid.arrange(p4, p5, p6, ncol = 3)

# Partial dependence plots for the top/bottom three features
ames_vi <- vi(ames_xgb, pred.var = colnames(X), type = "Gain")
feats <- c(head(ames_vi, n = 3)$Variable, tail(ames_vi, n = 3)$Variable)
pds <- lapply(feats, FUN = function(x) {
  pd <- cbind(x, partial(ames_xgb, pred.var = x, train = X))
  names(pd) <- c("xvar", "xval", "yhat")
  pd
})
pds <- do.call(rbind, pds)
ggplot(pds, aes(x = xval, y = yhat)) +
  geom_line() +
  geom_hline(yintercept = mean(ames$Sale_Price), linetype = 2, col = "red2") +
  facet_wrap( ~ xvar, scales = "free_x") +
  labs(x = "", y = "Partial dependence") +
  theme_light()