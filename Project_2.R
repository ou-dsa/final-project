library(caret)
library(ROCR)
library(Metrics)
library(ggplot2)
library(dummies)
library(rpart)
library(ggplot2)
library(scales)
library(reshape2)
library(lubridate)
library(plyr)

#EDA---

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

dataset$is_fraud = as.factor(dataset$is_fraud)
#dataset$amount = log(dataset$amount)
ggplot(dataset, aes(x = is_fraud, y = amount, group = is_fraud)) + geom_boxplot() +
  labs(x = "Predictor Variables", title = "Glass Data")

#split dataset
require(caTools)
set.seed(5)
sample = sample.split(dataset_agg$is_fraud, SplitRatio = .7)
train = subset(dataset_agg, sample == TRUE)
test  = subset(dataset_agg, sample == FALSE)


#data preprocessing for logistic regression
train = data.frame(train)
train$id_issuer = as.factor(train$id_issuer)
train$amount_group = as.factor(train$amount_group)
train$pos_entry_mode = as.factor(train$pos_entry_mode)
train$is_upscale = as.factor(train$is_upscale)
train$mcc_group = as.factor(train$mcc_group)
train$type = as.factor(train$type)
train$datetime = as.numeric(as.POSIXct(train$datetime))
train$is_fraud = as.factor(train$is_fraud)
levels(train$is_fraud) <- c("no_fraud", "fraud")
#delete fraud id issuer

test = data.frame(test)
test$id_issuer = as.factor(test$id_issuer)
test$amount_group = as.factor(test$amount_group)
test$pos_entry_mode = as.factor(test$pos_entry_mode)
test$is_upscale = as.factor(test$is_upscale)
test$mcc_group = as.factor(test$mcc_group)
test$type = as.factor(test$type)
test$datetime = as.numeric(as.POSIXct(test$datetime))
test$is_fraud = as.factor(test$is_fraud)
levels(test$is_fraud) <- c("no_fraud", "fraud")

#logistic regression
fitControl <- trainControl(method="repeatedcv",
                           number=10, 
                           repeats=1,
                           verboseIter = TRUE, 
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

fit.glm_1 <- train(is_fraud~.,data=train,
                    method="glm",
                    trControl=fitControl,
                    metric = "ROC")
fit.glm_1
summary(fit.glm_1)
# Best model
#  ROC        Sens      Spec     
#0.9690768  0.997089  0.5568035

#predict and performance measure

test.glm = predict(fit.glm_1, test)
test.glm = as.numeric(test.glm)-1
pred = prediction(as.numeric(test.glm)-1, as.numeric(test$is_fraud)-1)

perf1 = performance(pred, "prec", "rec")
plot(perf1)
#0.7808035war


ComputeSavings(test$amount, as.numeric(test.glm)-1, as.numeric(test$is_fraud)-1)
#413512.3 dollars

#ComputeSavings(test$amount, as.numeric(test$is_fraud)-1, as.numeric(test$is_fraud)-1)
#561712.9

#decision tree--------------
#data preprocessing for decision tree

tree.1 <- rpart(data=train, is_fraud~.,control=rpart.control(cp=0.0001))
printcp(tree.1)
plotcp(tree.1)

# based on the plot:  the min CV error is with cp=0.00058 
# (which produces a tree with 202 leaf nodes)--> overfitting?
pfit<-prune(tree.1,cp=0.00058)

pred = predict(pfit,test, type="prob")[,2]
confusionMatrix(predict(pfit,test, type = "class"), as.numeric(as.character(test$is_fraud)), positive="1")

# assess the classifier
#Confusion Matrix and Statistics

#              Reference
#Prediction      0      1
#0 152866   1427
#1    393   2228
lloss = logLoss(as.numeric(as.character(test$is_fraud)),pred)

Evaluation(as.numeric(as.character(test$is_fraud)), pred)

#random forest-----------------

tuneGrid = expand.grid(.mtry = c(1:12))

fitControl <- trainControl(method="repeatedcv",number=10, repeats=5, verboseIter = TRUE)

rf_model<-train(is_fraud~.,data=train,
                method="rf",
                trControl=fitControl,
                tuneGrid = tuneGrid,
                ntree = 20)

plot(rf_model)


#now build single model with ntree = 1500 and with optimal mtry = 13
rf_final <- randomForest(is_fraud ~ ., data = train, ntrees=1500, mtry=13)


pred_fr = predict(rf_final, type="class")
confusionMatrix(pred_fr, dataset$is_fraud)

# assess the classifier
prediction.randomforest = predict(rf_final, newdata=dataset, type = "prob")[,2]

Evaluation(dataset_agg_m[,45], prediction.randomforest)

#Confusion Matrix and Statistics--> overfitting

#             Reference
#Prediction  0     1
#0         40289     0
#1            1  2710

#boosted tree-------
dataset$country_code = as.factor(dataset$country_code)

#caret
fitControl <- trainControl(method="repeatedcv",number=10, repeats=5, verboseIter = TRUE)

tuneGrid <- expand.grid(mfinal = c(6,9,21,63,100), maxdepth = c(1,3),
                        coeflearn = "Freund")

boost_model<-train(is_fraud~.,data=dataset,
                   method="AdaBoost.M1",
                   trControl=fitControl,
                   tuneGrid = tuneGrid,
                   boos = F,
                   verbose = TRUE)

plot(boost_model)
#Fitting mfinal = 100, maxdepth = 3, coeflearn = Freund on full training set

#single model with optimal parameters
fit_boost<-boosting(is_fraud ~ ., data = dataset, boos = F, mfinal = 100, coeflearn = 'Freund', maxdepth = 3)

#Evaluation
prediction.boostedtree = predict(fit_boost, newdata=dataset, type = "prob") 
prediction.boostedtree = prediction.boostedtree$prob[,2]
Evaluation(dataset_agg_m[,45], prediction.boostedtree)

#Confusion Matrix and Statistics

#           Reference
#Prediction  0     1
#0         39837  1725
#1          453   985


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
