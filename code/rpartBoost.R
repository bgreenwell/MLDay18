# Nicer color palette
cols <- RColorBrewer::brewer.pal(9, "Set1")

# Function to implement gradient boosting with squared-error loss. Based on
# algorithm 17.2 on page 333 of Computer Age Statistical Inference, by Bradley
# Efron amd Trevor Hastie
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

# Function to plot the predictions from a particular boosting iteration
plotIter <- function(object, iter, show_legend = FALSE, ...) {
  plot(x, y, ...)
  lines(x, sin(x), lwd = 3, col = cols[2L])
  lines(x, object[, iter + 1], lwd = 3, col = cols[1L])
  if (show_legend) {
    legend("topright", legend = c("Boosted prediction", "True function"),
           lty = 1L, lwd = 3L, col = cols[1L:2L], inset = 0.01)
  }
}

# Simulate some sine wave data
set.seed(101)
x <- seq(from = 0, to = 2 * pi, length = 500)
y <- sin(x) + rnorm(length(x), sd = 0.3)
plot(x, y)

# gradient boosted decision trees
bst <- rpartBoost(X = x, y = y, num_trees = 1000, learn_rate = 0.1, 
                  tree_depth = 3, verbose = TRUE)

# Plot first 15 iterations
par(mfrow = c(3, 3))
for (i in c(0, 5, 10, 15, 25, 50, 100, 500, 1000)) {
  plotIter(bst, iter = i, main = paste("Iter:", i))
}

# # Make a gif (requires that ImageMagick beinstalled on machine)
# png(file = "gifs/boosted-stumps%02d.png", width = 500, height = 500)
# for (i in c(0:100)){
#   plotIter(bst, iter = i, show_legend = TRUE)
# }
# dev.off()
# system("convert -delay 30 gifs/*.png gifs/boosted_stumps.gif")
# file.remove(list.files(path = "gifs", pattern = ".png", full.names = TRUE))
