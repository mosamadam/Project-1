library(knitr)
library(flextable)
library(corrplot)
library(ggplot2)
library(dplyr)
library(magrittr)
library(tidyverse)
library(doParallel)
library(caret)
library(mltools)
library(gridExtra)
library(e1071)
library(patchwork)
library(magick)
library(png)
library(grid)
library(h2o )

#....................................................................................................
#................SVM SECTION.................................................................................
#....................................................................................................

#load dataset
data <- read.csv("train_new.csv")

# Specify the response and predictor variables
response <- 1  # Replace with your response variable column index
predictors <- 2:ncol(data)

# Convert the response variable to a factor
data[, response] <- as.factor(data[, response])

# Create a data frame for the predictors and response
df <- data[, c(predictors, response)]

num_cores <- 4
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#create different dataset partitions
part_df <- c(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1)
acc_histSVM <- c()

#run loop to find optimal dataset size for cross val training
for (i in 1:7) {
  
  # Split the data into training and test sets
  set.seed(1)
  reduced_sample <- sample(1:nrow(df), part_df[i] * nrow(df))
  reduced_data <- df[reduced_sample, ]
  train_sample_part <- sample(1:nrow(reduced_data), 0.8 * nrow(reduced_data))
  trainPart <- reduced_data[train_sample_part, ]
  testPart <- reduced_data[-train_sample_part, ]
  
  svm_polPart <- train(
    evenodd ~ .,
    data = trainPart,
    method = "svmPoly",
    trControl = trainControl(method = "none", allowParallel = TRUE),  
    tuneGrid = expand.grid(degree = 2, C = 1, scale = TRUE),
    preProcess = c("center", "scale"),
    metric = "Accuracy"
  )
  
  
  # Predict the class labels for the test data
  predicted_polPart <- predict(svm_polPart, newdata = testPart)
  
  
  # Calculate the misclassification rate
  accuracySVM <- mean(predicted_polPart == testPart$evenodd)
  acc_histSVM <- c(acc_histSVM, accuracySVM)
}

stopCluster(cl)

pltdf <- data.frame(part_df = 100 * part_df, acc_hist = 100 * acc_histSVM)

# Create the plot using ggplot
p <- ggplot(pltdf, aes(x = part_df, y = acc_hist)) +
  geom_line(color = "blue", linetype = "solid") +
  geom_point(color = "blue", shape = 16) +
  labs(x = "Subset percent (%)", y = "Accuracy (%)") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 12),       # Set text size of axis values
    axis.title = element_text(size = 12) )

p

#....................................................................................................
#................NN SECTION.................................................................................
#....................................................................................................

# Start the H2O cluster
h2o.init(nthreads = -1)

# Read the dataset into H2O
data <- h2o.importFile("train_new.csv")

# Specify the response and predictor variables
response <- 1  # Assuming the first column is the response variable
predictors <- 2:ncol(data)  # Assuming the predictor columns start from the second column

# Convert the response variable to a factor
data[, response] <- as.factor(data[, response])

#determine reasonable dataset size to train cross validation model

#create different dataset partitions
part_df <- c(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1)
acc_histNN <- c()

#run loop to find optimal dataset size for cross val training
for (i in 1:7) {
  
  #split data
  splitPart <- h2o.splitFrame(data, ratios = c(0.8*part_df[i], 0.19999999*part_df[i]), seed = 1)
  trainPart <- splitPart[[1]]
  testPart <- splitPart[[2]]
  
  #define the neural network model
  model <- h2o.deeplearning(
    x = predictors,
    y = response,
    training_frame = trainPart,
    activation = "Rectifier",
    l1 = 1e-5,
    hidden = c(200),  # Specify the number of hidden layers and nodes
    epochs = 10,  # Number of training iterations
    variable_importances = FALSE,
    loss = 'CrossEntropy',
    seed=1
  )
  
  #make predictions on the test set
  predictions <- h2o.predict(model, testPart)
  yhat <- predictions$predict
  accuracy <- sum(yhat == testPart[,1])/nrow.H2OFrame(testPart[,1])
  acc_histNN <- c(acc_histNN, accuracy)
  
}

# Stop the H2O cluster
h2o.shutdown()

dfNNS <- data.frame(part_df = 100 * part_df, acc_hist = 100 * acc_hist)

# Create the plot using ggplot
p2 <- ggplot(dfNNS, aes(x = part_df, y = acc_hist)) +
  geom_line(color = "blue", linetype = "solid") +
  geom_point(color = "blue", shape = 16) +
  labs(x = "Subset percent (%)", y = "Accuracy (%)") +
  theme_minimal()+
  theme(
    axis.text = element_text(size = 12),       # Set text size of axis values
    axis.title = element_text(size = 12) )
p2

p_combined <- p + p2 + plot_layout(ncol = 2)

# Display the combined plots
p_combined

ggsave("subset.png", p_combined, width = 17.8, height = 8, units = "cm")
