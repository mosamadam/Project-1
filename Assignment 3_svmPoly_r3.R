# Load the required libraries
library(caret)
library(e1071)
library(doParallel)
library(ggplot2)

# Read the dataset
data <- read.csv("train_new.csv")

# Specify the response and predictor variables
response <- 1  # Replace with your response variable column index
predictors <- 2:ncol(data)

# Convert the response variable to a factor
data[, response] <- as.factor(data[, response])

# Create a data frame for the predictors and response
df <- data[, c(predictors, response)]

# Split the data into training and test sets
set.seed(1)
df_train <- sample(1:nrow(df), 0.2*nrow(df))
train<- df[df_train, ]

#split full data
set.seed(1)
train_sample_full <- sample(1:nrow(df), 0.8*nrow(df))
trainFull <- df[train_sample_full, ]
testFull <- df[-train_sample_full, ]

#...............................................................................................
#parameter optimization with reduced dataset

#partition space for parralel computing
getModelInfo()$svmPoly$parameters
num_cores <- 4
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#set up search grid for cross validation
parameter_grid <- expand.grid(
  degree = c(2,3,4),          # Scaling parameter (set to FALSE for no scaling)
  scale = c(TRUE),
  C = c(0.01, 10)          # Cost parameter
)

#create the control object for parallel processing
ctrl <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE  # Enable parallel processing
)

# Perform cross-validation with parallel processing
set.seed(1)
tuned_model <- train(
  evenodd ~ .,
  data = train,
  method = "svmPoly",
  trControl = ctrl,
  tuneGrid = parameter_grid,
  preProcess = c("center"),
  metric = "Accuracy"
)

# Extract the relevant information from grid_results
results <- data.frame(
  degree = tuned_model$results$degree,
  cost = tuned_model$results$C,
  acc = tuned_model$results$Accuracy
)


p2 <- ggplot(results, aes(x = cost, y = acc, group = interaction(degree), color = interaction(degree))) +
  geom_line() +
  labs(x = "Cost", y = "Accuracy", color = "Degree") +
  scale_x_continuous(breaks = c(0.01, 100)) +
  scale_color_manual(values = c("red", "blue", "green")) +
  theme_minimal()

ggsave("SVMcv.png", p2, width = 17.8, height = 8, units = "cm")

#............................................................................................
#train model on full dataset using optimal hyperparams

best_model <- train(
  evenodd ~ .,
  data = trainFull,
  method = "svmPoly",
  trControl = trainControl(method = "none", allowParallel = TRUE),
  tuneGrid = tuned_model$bestTune,
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

save(best_model, file = "best_model.RData")

#predict on the test set
predictions <- predict(best_model, newdata = testFull)

#assess the performance on the test set
mean(predictions == testFull$evenodd)

#............................................................................................
#make predictiona=s on test_new.csv data

# Read the dataset
testFinal <- read.csv("test_new.csv")

#predict on the test set
predFinalSVM <- predict(best_model, newdata = testFinal)

pred = data.frame((predFinalSVM))
write.table(pred,'Digits_Pred_SVM_MSMADA002.csv',
               quote = F, row.names = F, sep = ',')



stopCluster(cl)


