library(readr)
library(caret)
library(ROCR)
library(Metrics)
library(ggplot2)
library(rpart)
library(ggplot2)
library(scales)
library(reshape2)
library(lubridate)
library(plyr)
library(irr)
library(caTools)
library(magrittr)
library(dplyr)
library(dummies)
library(Matrix)

#Savings---------
ComputeSavings <- function(amounts, pred.values, true.values) {
  predictions <- data.frame(amounts, pred.values, true.values)
  
  costs <- 0
  for (i in 1:nrow(predictions)) {
    pred.value <- predictions$pred.values[i]
    true.value <- predictions$true.values[i]
    
    if (pred.value == 1) {
      costs <- costs + 20
    } else if (pred.value == 0 & true.value == 1) {
      costs <- costs + predictions$amount[i]
    }
  }
  
  savings <- sum(predictions$amounts[predictions$true.values == 1]) - costs
  
  return(savings)
}

#Load dataset
dataset <- read_csv("dataset_agg.csv", 
                    col_types = cols(datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S")))

#EDA---

# total and fraudulent transactions over time
x = data.frame(dataset$is_fraud)
x$date = dataset$datetime
x$my = floor_date(dataset$datetime, "month")
x$count = rep(1,nrow(x))

y = ddply(x, "my", summarise, Fraudalent_Transactions = sum(dataset.is_fraud))
yy = ddply(x, "my", summarise, Total_Transactions = sum(count))
y$Total_Transactions = yy$Total_Transactions

y2 = melt(y, id.vars = 1)

p = ggplot() +  
  geom_line(aes(my, Fraudalent_Transactions, colour = "Fraudulent Transactions"), y ) +
  geom_line(data =yy, aes(my, Total_Transactions, colour = "Total Transactions")) +
  labs(x = "Time",
       y = "Transactions") +
  theme(legend.key=element_blank())+
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  scale_color_manual(values=c("red", "black"))

p$labels$colour <- "Legend"
p

# boxplot of amount vs fraud/no_fraud
dataset$is_fraud = as.factor(dataset$is_fraud)
levels(dataset$is_fraud) <- c("no_fraud", "fraud")
dataset$amount = log(dataset$amount)
ggplot(dataset, aes(x = is_fraud, y = amount, group = is_fraud)) + geom_boxplot() +
  labs(x = "Predictor Variables", title = "Fraud data")

#split dataset
dataset <- read_csv("dataset_agg.csv", 
                    col_types = cols(datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S")))
set.seed(5)
sample = sample.split(dataset$is_fraud, SplitRatio = .7)
train = subset(dataset, sample == TRUE)
test  = subset(dataset, sample == FALSE)


#data preprocessing for logistic regression, decision tree and random forest
train = data.frame(train)
train$amount_group = as.factor(train$amount_group)
train$pos_entry_mode = as.factor(train$pos_entry_mode)
train$is_upscale = as.factor(train$is_upscale)
train$mcc_group = as.factor(train$mcc_group)
train$type = as.factor(train$type)
train$datetime = as.numeric(as.POSIXct(train$datetime))
train$is_fraud = as.factor(train$is_fraud)
levels(train$is_fraud) <- c("no_fraud", "fraud")
train = subset(train, select = -c(id_issuer))


test = data.frame(test)
test$amount_group = as.factor(test$amount_group)
test$pos_entry_mode = as.factor(test$pos_entry_mode)
test$is_upscale = as.factor(test$is_upscale)
test$mcc_group = as.factor(test$mcc_group)
test$type = as.factor(test$type)
test$datetime = as.numeric(as.POSIXct(test$datetime))
test$is_fraud = as.factor(test$is_fraud)
levels(test$is_fraud) <- c("no_fraud", "fraud")
test = subset(test, select = -c(id_issuer))


#logistic regression with AUC as performance measure 
fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

fit.glm_1 <- train(is_fraud~.,data=train,
                    method="glm",
                    trControl=fitControl,
                    metric = "ROC")
fit.glm_1
summary(fit.glm_1)

#predict and performance measure

test.glm = predict(fit.glm_1, test)

test.glm = as.numeric(test.glm)-1
pred.glm = prediction(test.glm, as.numeric(test$is_fraud)-1)
performance(pred.glm, "auc")


glm.frame = data.frame(test.glm)
glm.frame$true = as.numeric(test$is_fraud)-1
kappa2(glm.frame)


ComputeSavings(test$amount,test.glm, as.numeric(test$is_fraud)-1)

#theoretical max savings
ComputeSavings(test$amount, as.numeric(test$is_fraud)-1, as.numeric(test$is_fraud)-1)


#logistic regression with Kappa as performance measure 
fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5)

fit.glm_1 <- train(is_fraud~.,data=train,
                   method="glm",
                   trControl=fitControl,
                   metric = "Kappa")
fit.glm_1
summary(fit.glm_1)

#predict and performance measure

test.glm = predict(fit.glm_1, test)

test.glm = as.numeric(test.glm)-1
pred.glm = prediction(test.glm, as.numeric(test$is_fraud)-1)
performance(pred.glm, "auc")


glm.frame = data.frame(test.glm)
glm.frame$true = as.numeric(test$is_fraud)-1
kappa2(glm.frame)


ComputeSavings(test$amount,test.glm, as.numeric(test$is_fraud)-1)



#decision tree with AUC as performance meaure--------------

tuneGrid <- expand.grid(cp = seq(0.0001,0.001, length.out = 10))

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5, 
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

fit.tree1 <- train(is_fraud~.,data=train,
                   method="rpart",
                   trControl=fitControl,
                   tuneGrid = tuneGrid,
                   metric = "ROC")
fit.tree1
plot(fit.tree1)


#predict and performance measure
test.tree = predict(fit.tree1, test)

test.tree = as.numeric(test.tree)-1
pred.tree = prediction(test.tree, as.numeric(test$is_fraud)-1)
performance(pred.tree, "auc")


tree.frame = data.frame(test.tree)
tree.frame$true = as.numeric(test$is_fraud)-1
kappa2(tree.frame)


ComputeSavings(test$amount, test.tree, as.numeric(test$is_fraud)-1)


#decision tree with Kappa as performance meaure--------------

tuneGrid <- expand.grid(cp = seq(0.0001,0.001, length.out = 10))

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5)

fit.tree1 <- train(is_fraud~.,data=train,
                   method="rpart",
                   trControl=fitControl,
                   tuneGrid = tuneGrid,
                   metric = "Kappa")
fit.tree1
plot(fit.tree1)


#predict and performance measure
test.tree = predict(fit.tree1, test)

test.tree = as.numeric(test.tree)-1
pred.tree = prediction(test.tree, as.numeric(test$is_fraud)-1)
performance(pred.tree, "auc")


tree.frame = data.frame(test.tree)
tree.frame$true = as.numeric(test$is_fraud)-1
kappa2(tree.frame)


ComputeSavings(test$amount, test.tree, as.numeric(test$is_fraud)-1)


#random forest with AUC as performance measure-----------------

tuneGrid = expand.grid(.mtry = c(7,13,25))

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

rf_model<-train(is_fraud~.,data=train,
                method="rf",
                trControl=fitControl,
                tuneGrid = tuneGrid,
                ntree = 50,
                metric = "ROC")
rf_model
plot(rf_model)


##predict and performance measure
test.forest = predict(rf_model, test)

test.forest = as.numeric(test.forest)-1
pred.forest = prediction(test.forest, as.numeric(test$is_fraud)-1)
performance(pred.forest, "auc")


forest.frame = data.frame(test.forest)
forest.frame$true = as.numeric(test$is_fraud)-1
kappa2(forest.frame)


ComputeSavings(test$amount, test.forest, as.numeric(test$is_fraud)-1)


#random forest with Kappa as performance measure-----------------

tuneGrid = expand.grid(.mtry = 13)

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5)

rf_model<-train(is_fraud~.,data=train,
                method="rf",
                trControl=fitControl,
                tuneGrid = tuneGrid,
                ntree = 50,
                metric = "Kappa")
rf_model
plot(rf_model)



##predict and performance measure
test.forest = predict(rf_model, test)

test.forest = as.numeric(test.forest)-1
pred.forest = prediction(test.forest, as.numeric(test$is_fraud)-1)
performance(pred.forest, "auc")


forest.frame = data.frame(test.forest)
forest.frame$true = as.numeric(test$is_fraud)-1
kappa2(forest.frame)


ComputeSavings(test$amount, test.forest, as.numeric(test$is_fraud)-1)


#XGBoost tree-------
dataset <- read_csv("dataset_agg.csv", 
                    col_types = cols(datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S")))
#split dataset
set.seed(5)
sample = sample.split(dataset$is_fraud, SplitRatio = .7)
train = subset(dataset, sample == TRUE)
test  = subset(dataset, sample == FALSE)


#data preprocessing for XGBoost
train = data.frame(train)
is_fraud = as.factor(train$is_fraud)
levels(is_fraud) <- c("no_fraud", "fraud")

train = subset(train, select = -c(id_issuer, is_fraud))
train$amount_group = as.numeric(as.factor(train$amount_group))
train$pos_entry_mode = as.numeric(as.factor(train$pos_entry_mode))
train$is_upscale = as.numeric(as.factor(train$is_upscale))
train$mcc_group = as.numeric(as.factor(train$mcc_group))
train$type = as.factor(train$type)
train$type = as.numeric(train$type)-1
train$datetime = as.numeric(as.POSIXct(train$datetime))

train_m = as.matrix(train)
train_m = as(train_m, "dgCMatrix")


test = data.frame(test)
trueval <- as.numeric(test$is_fraud)
test = subset(test, select = -c(id_issuer, is_fraud))
test$amount_group = as.numeric(as.factor(test$amount_group))
test$pos_entry_mode = as.numeric(as.factor(test$pos_entry_mode))
test$is_upscale = as.numeric(as.factor(test$is_upscale))
test$mcc_group = as.numeric(as.factor(test$mcc_group))
test$type = as.factor(test$type)
test$type = as.numeric(test$type)-1
test$datetime = as.numeric(as.POSIXct(test$datetime))


test_m = as.matrix(test)
test_m = as(test_m, "dgCMatrix")

#XGBoost with AUC as performance measure
tuneGrid = expand.grid(nrounds = 100,               # # Boosting Iterations
                       max_depth = c(4, 7, 20),       # Max Tree Depth
                       eta = 0.3,                     # Shrinkage
                       gamma = c(0, 0.8),             # Minimum Loss Reduction
                       colsample_bytree = c(0.5,1),   # Subsample Ratio of Columns
                       min_child_weight = c(1,8),     # Minimum Sum of Instance Weight
                       subsample = 1)                 # Subsample Percentage             

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5, 
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           allowParallel = TRUE)

xgboost_1<-train(x = train_m,
                y = is_fraud,
                method="xgbTree",
                trControl=fitControl,
                tuneGrid = tuneGrid,
                ntree = 50,
                metric = "ROC",
                max_delta_step = 1,
                scale_pos_weight = 42,
                objective = "binary:logistic") 

xgboost_1
plot(xgboost_1)


##predict and performance measure
test.xgboost = predict(xgboost_1, test_m, type = "prob")[,2]#first predict probabiloties
test.xgboost = as.numeric(test.xgboost>0.5)
pred.xgboost = prediction(test.xgboost, trueval)
performance(pred.xgboost, "auc")


xgboost.frame = data.frame(test.xgboost)
xgboost.frame$true = as.numeric(trueval)
kappa2(xgboost.frame)


ComputeSavings(test$amount, test.xgboost, trueval)


#XGBoost with Kappa as performance measure
tuneGrid = expand.grid(nrounds = 100,               # # Boosting Iterations
                       max_depth = 20,       # Max Tree Depth
                       eta = 0.3,                     # Shrinkage
                       gamma = 0,             # Minimum Loss Reduction
                       colsample_bytree = 1,   # Subsample Ratio of Columns
                       min_child_weight = 1,     # Minimum Sum of Instance Weight
                       subsample = 1)                 # Subsample Percentage             

fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=5,
                           classProbs = TRUE)

xgboost_1<-train(x = train_m,
                 y = is_fraud,
                 method="xgbTree",
                 trControl=fitControl,
                 tuneGrid = tuneGrid,
                 ntree = 50,
                 metric = "Kappa",
                 max_delta_step = 1,
                 scale_pos_weight = 42,
                 objective = "binary:logistic") 

xgboost_1
plot(xgboost_1)


##predict and performance measure
test.xgboost = predict(xgboost_1, test_m, type = "prob")[,2]#first predict probabiloties
test.xgboost = as.numeric(test.xgboost>0.5)
pred.xgboost = prediction(test.xgboost, trueval)
performance(pred.xgboost, "auc")


xgboost.frame = data.frame(test.xgboost)
xgboost.frame$true = as.numeric(trueval)
kappa2(xgboost.frame)


ComputeSavings(test$amount, test.xgboost, trueval)

