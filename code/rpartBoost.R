rpartBoost <- function(X, y, data, num_trees = 100, learn_rate = 0.1, 
                       tree_depth = 6, verbose = FALSE) {
  require(rpart)
  G_b_hat <- matrix(0, nrow = length(y), ncol = num_trees + 1)
  r <- y
  for (tree in seq_len(num_trees)) {
    if (verbose) {
      message("iter ", tree, " of ", num_trees)
    }
    g_b_tilde <- rpart(r ~ X, control = list(cp = 0, maxdepth = tree_depth))
    g_b_hat <- learn_rate * predict(g_b_tilde)
    G_b_hat[, tree + 1] <- G_b_hat[, tree] + matrix(g_b_hat)
    r <- r - g_b_hat
  }
  colnames(G_b_hat) <- paste0("tree_", c(0, seq_len(num_trees)))
  G_b_hat
}

plotIter <- function(object, iter, ...) {
  plot(x, y)
  lines(x, object[, iter + 1], lwd = 3, ...)
}

# Simulate some sine wave data
set.seed(101)
x <- seq(from = 0, to = 2 * pi, length = 500)
y <- sin(x) + rnorm(length(x), sd = 0.3)
plot(x, y)

# gradient boosted decision trees
bst <- rpartBoost(X = x, y = y, num_trees = 500, learn_rate = 0.5, 
                  tree_depth = 1, verbose = TRUE)

par(mfrow = c(4, 4))
for (i in 0:15) {
  plotIter(bst, iter = i, col = "red2")
}