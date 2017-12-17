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


#create aggregated features
card.fraud <- read_csv("~/Documents/Ing.Informatica/Exchange/Intelligent Data Analytics/Final/dataset.csv", 
                       col_types = cols(datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S")))

# Time windows
name <- c("1d", "2d", "7d", "30d")
secs <- c(60 * 60 * 24,       # 1 day
          60 * 60 * 24 * 2,   # 2 days
          60 * 60 * 24 * 7,   # 7 days
          60 * 60 * 24 * 30)  # 30 days
time.windows <- data.frame(name, secs)

clients <- unique(card.fraud$tokenized_pan)

for (i in 1:nrow(time.windows)) {
  tw <- time.windows[i, ]
  card.fraud$Id <- seq.int(nrow(card.fraud))
  
  cnt.attr.name <- paste("cnt_", tw$name, sep = "")
  sum.attr.name <- paste("sum_", tw$name, sep = "")
  
  card.fraud[, cnt.attr.name] <- NA
  card.fraud[, sum.attr.name] <- NA
  
  for (client in clients) {
    client.trxs <- card.fraud[card.fraud$tokenized_pan == client, ]
    
    for (j in 1:nrow(client.trxs)) {
      trx <- client.trxs[j, ]
      current.datetime <- trx$datetime
      related.trxs <- client.trxs[client.trxs$datetime > current.datetime - tw$secs & 
                                    client.trxs$datetime <= current.datetime  &
                                    client.trxs$Id != trx$Id &
                                    client.trxs$is_fraud == 0, ]
      
      card.fraud[card.fraud$Id == trx$Id, cnt.attr.name] <- nrow(related.trxs)
      card.fraud[card.fraud$Id == trx$Id, sum.attr.name] <- sum(related.trxs$amount)
    }
  }
  
  card.fraud <- card.fraud[, !names(card.fraud) %in% c("Id")]
}

#fraud ratio per id_issuer
issuers <- unique(card.fraud$id_issuer)
card.fraud[, "frd_by_id_issuer"] <- NA
for (issuer in issuers) {
  issuer.frauds <- nrow(card.fraud[card.fraud$is_fraud == 1 & card.fraud$id_issuer == issuer, ])
  issuer.count <- nrow(card.fraud[card.fraud$id_issuer == issuer, ])
  card.fraud$frd_by_id_issuer[card.fraud$id_issuer == issuer] <- issuer.frauds/issuer.count
}

#fraud ratio per merchant
merchants <- unique(card.fraud$id_merchant)
card.fraud[, "frd_by_id_merchant"] <- NA
for (merchant in merchants) {
  merchant.frauds <- nrow(card.fraud[card.fraud$is_fraud == 1 & card.fraud$id_merchant == merchant, ])
  merchant.count <- nrow(card.fraud[card.fraud$id_merchant == merchant, ])
  card.fraud$frd_by_id_merchant[card.fraud$id_merchant == merchant] <- merchant.frauds/merchant.count
}

write.csv(card.fraud,
          file = "~/Documents/Ing.Informatica/Exchange/Intelligent Data Analytics/Final/dataset_agg.csv",
          row.names=FALSE)

ComputeSavings <- function(amounts, pred.values, true.values) {
  predictions <- data.frame(amounts, pred.values, true.values)
  
  costs <- 0
  for (i in 1:nrow(predictions)) {
    pred.value <- predictions$pred.values[i, ]
    true.value <- predictions$true.values[i, ]
    
    if (pred.value == 1) {
      costs <- costs + 20
    } else if (pred.values == 0 & true.value == 1) {
      costs <- costs + predictions$amount[i, ]
    }
  }
  
  savings <- sum(predictions$amounts[predictions$true.values == 1, ]) - costs
  
  return(savings)
}


#calculate Savings---------
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


#XGBoost tree-------
dataset <- read_csv("dataset.csv") 
                    
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
                       max_depth = c(7,20),       # Max Tree Depth
                       eta = 0.3,                     # Shrinkage
                       gamma = 0,             # Minimum Loss Reduction
                       colsample_bytree = 1,   # Subsample Ratio of Columns
                       min_child_weight = 1,     # Minimum Sum of Instance Weight
                       subsample = 1)                # Subsample Percentage             

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
                           allowParallel = TRUE,
                           verboseIter = TRUE)

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

