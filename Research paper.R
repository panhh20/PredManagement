data <- read.csv("/Users/nguyenann/Downloads/MS Machine Learning/ai4i2020.csv",
                  header=TRUE, sep = ',')

# Check for NAs
colSums(is.na(data)) # No NAs
  
# Package for Correlation analysis
#install.packages("corrr")
library(corrr)

#install.packages("ggcorrplot")
library(ggcorrplot)

# Package for multivariate exploratory data analysis, perform PCA
#install.packages("FactoMineR") 
library(FactoMineR)

# Package for visualization of PCA
#install.packages("factoextra") 
library(factoextra) 

# Correlation matrix
pca_data = data[, c(3:8, 10:14)]# exclude the IDs columns and the y-variable (machine failure)
# Change the "Type" variable to a numeric ordered dummy variable
pca_data$Type <- as.factor(pca_data$Type)

# famd_data <- FAMD(pca_data, graph = FALSE)
# famd_data
# Create dummy variables for "Type"
library(fastDummies)
pca_data <- dummy_cols(pca_data, select_columns = "Type") 

data_normalized <- scale(pca_data[, -1]) # Normalize numerical data, exclude "Type" column
head(data_normalized)

corr_matrix <- cor(data_normalized)
ggcorrplot(corr_matrix)

corr_matrix2 <- cor(data_normalized[, c(1:5)])

# PCA

data.pca <- prcomp(corr_matrix2, scale=TRUE)
summary(data.pca)

data.pca$rotation # PCA Loadings

# Plot PCA Loadings

# Change colour of bar plot
c.pc1 <- ifelse(data.pca$rotation[,1] > 0, yes="mediumorchid", no="salmon1")
c.pc2 <- ifelse(data.pca$rotation[,2] > 0, "mediumorchid", "salmon1")
c.pc3 <- ifelse(data.pca$rotation[,3] > 0, "mediumorchid", "salmon1")
#c.pc4 <- ifelse(data.pca$rotation[,4] > 0, "mediumorchid", "salmon1")
#c.pc5 <- ifelse(data.pca$rotation[,5] > 0, "mediumorchid", "salmon1")
#c.pc6 <- ifelse(data.pca$rotation[,6] > 0, "mediumorchid", "salmon1")

# Get position for variable names
n.pc1 <- ifelse(data.pca$rotation[,1] > 0, -0.01, data.pca$rotation[,1]-0.01)
n.pc2 <- ifelse(data.pca$rotation[,2] > 0, -0.01, data.pca$rotation[,2]-0.01)
n.pc3 <- ifelse(data.pca$rotation[,3] > 0, -0.01, data.pca$rotation[,3]-0.01)
#n.pc4 <- ifelse(data.pca$rotation[,4] > 0, -0.01, data.pca$rotation[,4]-0.01)
#n.pc5 <- ifelse(data.pca$rotation[,5] > 0, -0.01, data.pca$rotation[,5]-0.01)
#n.pc6 <- ifelse(data.pca$rotation[,6] > 0, -0.01, data.pca$rotation[,6]-0.01)

# Plot
layout(matrix(c(1,2,3), nrow=1, ncol=3)) # Set up layout
par(mar=c(5,3,2,1), oma=c(7.5,0,0,0)) # Set up margins
# Plot PC1 - PC6

#x_labels = c("Air temperature K", "Process temperature K", "Rotational speed rpm", "Torque Nm",            
             #"Tool wear min", "Type_H", "Type_L", "Type_M" )
b1 <- barplot(data.pca$rotation[,1], main="PC1 Loadings Plot", col=c.pc1, las=2)

b2 <- barplot(data.pca$rotation[,2], main="PC2 Loadings Plot", col=c.pc2, las=2)

b3 <- barplot(data.pca$rotation[,3], main="PC3 Loadings Plot", col=c.pc3, las=2)


#layout(matrix(c(1,2,3), nrow=1, ncol=3)) # Set up layout
#par(mar=c(5,3,2,1), oma=c(7.5,0,0,0)) # Set up margins

#b4 <- barplot(data.pca$rotation[,4], names.arg=x_labels, main="PC4 Loadings Plot", col=c.pc4, las=2)

#b5 <- barplot(data.pca$rotation[,5], names.arg=x_labels, main="PC5 Loadings Plot", col=c.pc5, las=2)

#b6 <- barplot(data.pca$rotation[,6], names.arg=x_labels, main="PC6 Loadings Plot", col=c.pc6, las=2)



# Graph of the variables
fviz_pca_var(data.pca, col.var="contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE) # Avoid text overlapping


# Reset layout

dev.off()
# Random Forest

ml_data <- data[, c(3:14)]
ml_data$Type <- as.factor(ml_data$Type)
ml_data$Machine.failure <- as.factor(ml_data$Machine.failure)

ml_data$Failure.mode <- apply(ml_data[c('TWF', 'HDF', 'PWF', 'OSF', 'RNF')], 1, 
                              function(x) toString(names(x)[x ==1]))
ml_data$Failure.mode[ml_data$Failure.mode == "" | ml_data$Failure.mode == " "] <- "No Failure"

ml_data$Failure.mode <- as.factor(ml_data$Failure.mode)

# Create a train & test set
set.seed(1) 
train <- sample( 1: nrow(ml_data),nrow(ml_data)*0.7) # Training set = 70% of dataset
test <- -train # Test set = 30%

y.test <- ml_data$Machine.failure[-train]
y.train <- ml_data$Machine.failure[train]
data.train <- data.matrix(ml_data[train, c(1:6)])
data.test <- data.matrix(ml_data[-train, c(1:6)])

# Fit a random forest to the training data 
library(randomForest)

set.seed(1)
rf.PredMaint <- randomForest(Machine.failure~., data = ml_data[train, c(1:7)], mtry = 5, importance = TRUE, ntree = 250) 
# Initial mtry = 2 ~ sqrt(6) 

plot(rf.PredMaint, main="Random Forest: Error Rate vs Number of Trees",
     col = c("red", "black", "blue"))

# Variance Importance graph
# install.packages('ggthemes')
# install.packages('gridExtra')
library(ggthemes)
library(gridExtra)
library(grid)

importance(rf.PredMaint)
#varImpPlot(rf.PredMaint, main="Variance Importance Plot: Random Forest")
#varImpPlot(rf.PredMaint, main="Variance Importance Plot: Random Forest", type=2)

imp=importance(rf.PredMaint)
impL=imp[,c(3,4)]
imp.ma=as.matrix(impL)
imp.df=data.frame(imp.ma)

write.csv(imp.df, "imp.df.csv", row.names=TRUE)
imp.df.csv=read.csv("imp.df.csv",header=TRUE)

colnames(imp.df.csv)=c("Variable","MeanDecreaseAccuracy","MeanDecreaseGini")
imp.sort =  imp.df.csv[order(-imp.df.csv$MeanDecreaseAccuracy),] 

imp.sort = transform(imp.df.csv, 
                     Variable = reorder(Variable, MeanDecreaseAccuracy))

VIP=ggplot(data=imp.sort, aes(x=Variable, y=MeanDecreaseAccuracy)) + 
  ylab("Mean Decrease Accuracy")+xlab("")+
  geom_bar(stat="identity",fill="salmon1",alpha=.8,width=.75)+ 
  coord_flip()+theme_few() 

imp.sort.Gini <- transform(imp.df.csv, 
                           Variable = reorder(Variable, MeanDecreaseGini))

VIP.Gini=ggplot(data=imp.sort.Gini, aes(x=Variable, y=MeanDecreaseGini)) + 
  ylab("Mean Decrease Gini")+xlab("")+
  geom_bar(stat="identity",fill="mediumblue",alpha=.8,width=.75)+ 
  coord_flip()+theme_few() 

Var.Imp.Plot=arrangeGrob(VIP, VIP.Gini,ncol=2)
grid.draw(Var.Imp.Plot)


# Cross validation for mtry
set.seed(1)
# Create an empty vector to store the OOB error for each mtry value
oob_error <- rep(NA, ncol(ml_data)-7)

# Perform cross-validation for each mtry value and store the OOB error
for (mtry_val in 1:(ncol(ml_data)-7)) {
  rf_model <- randomForest(Machine.failure ~ ., data = ml_data[train,c(1:7)], 
                           mtry = mtry_val, ntree = 1000, importance = TRUE)
  oob_error[mtry_val] <- rf_model$err.rate[nrow(rf_model$err.rate),1]
}


# Plot the OOB error for each mtry value
plot(1:(ncol(ml_data)-7), oob_error, type = "b", xlab = "mtry", ylab = "OOB error")

# Select the mtry value with the lowest OOB error
best_mtry <- which.min(oob_error)
best_mtry # [1] 2

# Prediction accuracy

yhat_rf <- predict(rf.PredMaint,newdata = ml_data[-train, c(1:7)]) 
rf_table <-table(yhat_rf, y.test)

# Plot confusion matrix
install.packages("cvms")
library(cvms)
library(tibble)

rf.cfm <- as_tibble(rf_table)
plot_confusion_matrix(rf.cfm, 
                      target_col = "y.test", 
                      prediction_col = "yhat_rf",
                      counts_col = "n")
(2887+72)/3000 # Classification accuracy
#[1] 0.9843333

30/(30+72) # False Neg

# XGBoost

# install.packages("xgboost")
library(xgboost)

# convert the train and test data into xgboost matrix type.

#xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

dtrain = xgb.DMatrix(data.train, label=as.numeric(as.character(y.train)))
dtest = xgb.DMatrix(data.test, label=as.numeric(as.character(y.test)))

xgb_train <- xgboost(data = dtrain, label = as.numeric(as.character(y.train)), 
                     max.depth = 4, eta = 0.01, early_stopping_rounds = 50,
                     nthread = 2, nrounds = 1000, objective = "binary:logistic")


xgb_pred <- predict(xgb_train, dtest)
print(head(xgb_pred))

pred.xgb<-ifelse(xgb_pred < 0.5,'0','1') 
xgb_table <- table(pred.xgb, y.test)

xgb.cfm <- as_tibble(xgb_table)
plot_confusion_matrix(xgb.cfm, 
                      target_col = "y.test", 
                      prediction_col = "pred.xgb",
                      counts_col = "n")

xgb_err <- mean(as.numeric(xgb_pred > 0.5) != y.test)
print(paste("Classification error=", round(xgb_err, 5)))
# [1] "test-error= 0.01567"

(2888+65)/3000 # Classification accuracy
(37)/(37+65) # False Neg

# a plot with one of the trees
#install.packages("DiagrammeR")

xgb.plot.tree(model = xgb_train, trees=4) # Plot the 5th tree


###
# Grid Search to find the best hyperparameter combinations for XGBoost
###
watchlist = list(train=dtrain, test=dtest)

max.depths = c(3, 4, 5)
etas = c(1, 0.1, 0.01)

best_params = 0
best_score = 0

count = 1
for( depth in max.depths ){
  for( num in etas){
    
    bst_grid = xgb.train(data = dtrain, 
                         max.depth = depth, 
                         eta=num, 
                         nthread = 2, 
                         nrounds = 1000, 
                         watchlist = watchlist, 
                         objective = "binary:logistic", 
                         early_stopping_rounds = 50, 
                         verbose=0)
    
    if(count == 1){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
      count = count + 1
    }
    else if( bst_grid$best_score < best_score){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
    }
  }
}

best_params
best_score 

# Multivariate Random Forest for Multi-label problem
#install.packages("mlr3")
library(mlr)
library(mlrCPO)
library(mlr3)


#task = makeClassifTask(data = ml_data[train, c(1:6, 13)], target = "Failure.mode")
#mlr.test = ml_data[-train, c(1:6, 13)]
mrf_data[, c(6:10)] <- lapply(mrf_data[, c(6:10)], function (x) as.logical(x))
mldr_data <- mldr_from_dataframe(mrf_data, labelIndices = c(6:10))

# Define features and labels
features <- mrf_data
labels <- colnames(mrf_data[, 6:10])

# Define task and learner
train.task <- makeMultilabelTask(data = mrf_data[train,], target = labels)
test.task <- makeMultilabelTask(data = mrf_data[-train,], target = labels)
learner <- makeLearner("multilabel.cforest", predict.type = "response")

# Create Training & Test set

mrf.model = train(learner, train.task)

mrf.pred = predict(mrf.model, task = test.task)
names(as.data.frame(mrf.pred))

response <- as.data.frame(mrf.pred)

# Resampling
rdesc = makeResampleDesc(method = "CV", stratify = FALSE, iters = 5)
r = resample(learner, task, resampling = rdesc, show.info = FALSE)
r

perf <- getMultilabelBinaryPerformances(mrf.pred, measures = list(acc))
perf <- as.data.frame(perf)
perf <- cbind(FailureMode = rownames(perf), perf)
rownames(perf) <- 1:nrow(perf)
colnames(perf) <- c("Fail
ure Mode", "Classification Accuracy")
perf$`Classification Accuracy` <- round(perf$`Classification Accuracy`, 4)

# Other ways to do Multilabel classification: utiml
# Load the utiml package
#install.packages("utiml")
library(utiml)
library(mldr)

# Load the dataset
mrf_data <- pca_data[, -1]
mrf_data[, -c(6:10)] <- scale(mrf_data[, -c(6:10)]) # normalized data
mldr_data <- mldr_from_dataframe(mrf_data, labelIndices = c(6:10))

# Split the dataset into training and testing sets
train.test <- create_holdout_partition(mldr_data, c(train=0.7, test=0.3))

# Create a Classifier Chains model
br_model <- br(train.test$train, 'RF')
cc_model <- cc(train.test$train, 'RF')

# Make predictions on the testing set
br_pred <- predict(br_model, train.test$test)
cc_pred <- predict(cc_model, train.test$test)
# evaluate the performance of the model
cc_results <- multilabel_evaluate(train.test$test, cc_pred, c("example-based"))
round(cc_results, 4)

br_results <- multilabel_evaluate(train.test$test, br_pred, c("example-based"))
round(br_results, 4)

set.seed(1) 


multilabel_confusion_matrix(train.test$test, cc_pred)
multilabel_confusion_matrix(train.test$test, br_pred)


