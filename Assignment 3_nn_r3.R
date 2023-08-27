
# Load the required libraries
library(h2o )
library(dplyr)
library(caret)


# Start the H2O cluster
h2o.init(nthreads = -1)

# Read the dataset into H2O
data <- h2o.importFile("train_new.csv")

# Specify the response and predictor variables
response <- 1  # Assuming the first column is the response variable
predictors <- 2:ncol(data)  # Assuming the predictor columns start from the second column

# Convert the response variable to a factor
data[, response] <- as.factor(data[, response])


#...............................................................................................
#parameter optimization with reduced dataset

# Split the data into training and testing sets
set.seed(1)
split <- h2o.splitFrame(data, ratios = c(0.2), seed = 1)

train <- split[[1]]

#cross validation
hidden_opt <- list(c(50), c(100), c(200), c(50, 50), c(100,100), c(200,200))
l1_opt <- c(0.001, 1e-5)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)

execution_time <- system.time({

set.seed(1)
model_grid <- h2o.grid(
  "deeplearning",
  hyper_params = hyper_params,
  x = predictors,
  y = response,
  training_frame = train,
  activation = "rectifier",
  variable_importances = FALSE,
  loss = 'CrossEntropy',
  epochs = 10, 
  seed=1, 
  nfolds=5,
  stopping_rounds = 5,               # Number of rounds without improvement before stopping
  stopping_tolerance = 0.0001,
  grid_id = "my_grid")

})[["elapsed"]]

cat("Simulation execution time:", execution_time, "seconds\n")

grid_results <- h2o.getGrid("my_grid", sort_by = "accuracy", decreasing = TRUE)
best_model_id <- grid_results@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Extract the relevant information from grid_results
results <- data.frame(
  hidden = grid_results@summary_table$hidden,
  l1 = grid_results@summary_table$l1,
  acc = grid_results@summary_table$accuracy
)

results$Layers <- ifelse(results$hidden %in% c("50", "100", "200"), "single",
                       ifelse(results$hidden %in% c("[50, 50]", "[100, 100]", "[200, 200]"), "double", NA))

results <- results %>%
  mutate(hidden = ifelse(hidden == "[200, 200]", "200",
                         ifelse(hidden == "[50, 50]", "50",
                                ifelse(hidden == "[100, 100]", "100", hidden))))
results$hidden <- as.numeric(results$hidden)


p2 <- ggplot(results, aes(x = hidden, y = acc, group = interaction(l1, Layers), color = interaction(l1, Layers))) +
  geom_line() +
  labs(x = "Number of Neurons per layer", y = "Accuracy", color = "L1 & Layers") +
  scale_x_continuous(breaks = c(50, 100, 200)) +
  scale_color_manual(values = c("red", "blue", "green", "purple")) +
  theme_minimal()

ggsave("NNcv.png", p2, width = 17.8, height = 8, units = "cm")



#............................................................................................
#train model on full dataset using optimal hyperparams

# Split the data into training and testing sets
set.seed(1)
splitFull <- h2o.splitFrame(data, ratios = c(0.8, 0.19999999), seed = 1)

trainFull <- splitFull[[1]]
testFull <- splitFull[[2]]

set.seed(1)
model_full <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = trainFull,
  activation = "rectifier",
  variable_importances = FALSE,
  loss = 'CrossEntropy',
  hidden = best_model@params$input$hidden,
  l1 = best_model@params$input$l1,
  epochs = 10, 
  seed=1)


#make predictions on the test set
predictions <- h2o.predict(model_full, testFull)
yhat <- as.factor(as.matrix(predictions$predict))

yhat_factor <- factor(ifelse(yhat == "odd", 0, 1), levels = c("0", "1"))
testFull_factor <- factor(ifelse(as.factor(as.matrix(testFull[,1])) == "odd", 0, 1), levels = c("0", "1"))

# Create confusion matrix manually
conf_matrix <- table(Actual = testFull_factor, Predicted = yhat_factor)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Calculate specificity
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

#............................................................................................
#make predictiona=s on test_new.csv data

# Read the dataset
testFinal <- h2o.importFile("test_new.csv")

# create a data frame with the predictions
predFinalNN <- predict(model_full, newdata = testFinal)

# write the predictions to a CSV file
pred <- as.data.frame(predFinalNN)
write.table(pred, 'Digits_Pred_NN_MSMADA002.csv', quote = FALSE, row.names = FALSE, sep = ',')


# Stop the H2O cluster
h2o.shutdown()
