library(caret)
library(ROCR)
library(Metrics)
library(ggplot2)
library(dummies)
library(rpart)

#Evaluation function
Evaluation = function (y, prob){ 
  
  #create data frame with the three requires variables
  eval = as.data.frame(y)
  eval$prob = prob
  eval$pred<-as.numeric(prob>0.5)
  
  
  #ROC curve for training data --> works
  pred <- prediction(eval$prob, eval$y)    
  perf <- performance(pred,"tpr","fpr") 
  plot(perf,colorize=TRUE, print.cutoffs.at = c(0.25,0.5,0.75)); 
  abline(0, 1, col="red")
  
  #Concordant Pairs and AUC --> works
  Con_Dis_Data = cbind(eval$y,eval$prob) 
  
  ones = Con_Dis_Data[Con_Dis_Data[,1] == 1,]
  zeros = Con_Dis_Data[Con_Dis_Data[,1] == 0,]
  
  conc=matrix(0, dim(zeros)[1], dim(ones)[1])   #build a matrix of 0's 
  disc=matrix(0, dim(zeros)[1], dim(ones)[1])
  ties=matrix(0, dim(zeros)[1], dim(ones)[1])
  
  for (j in 1:dim(zeros)[1])
  {
    for (i in 1:dim(ones)[1])
    {
      if (ones[i,2]>zeros[j,2])
      {conc[j,i]=1}
      
      else if (ones[i,2]<zeros[j,2])
      {disc[j,i]=1}
      
      else if (ones[i,2]==zeros[j,2])
      {ties[j,i]=1}
    }
  }
  
  Pairs=dim(zeros)[1]*dim(ones)[1]              #total number of pairs
  PercentConcordance=(sum(conc)/Pairs)*100
  PercentDiscordance=(sum(disc)/Pairs)*100
  PercentTied=(sum(ties)/Pairs)*100
  AUC=PercentConcordance +(0.5 * PercentTied)
  
  #D statistic (2009) --> works
  honors.1<-eval[eval$y==1,]
  honors.0<-eval[eval$y==0,]
  dstat = mean(honors.1$prob) - mean(honors.0$prob)
  
  #Log loss -->
  lloss = logLoss(eval$y,eval$prob)
  
  #K-S chart  (Kolmogorov-Smirnov chart) --> works
  # measures the degree of separation 
  # between the positive (y=1) and negative (y=0) distributions
  
  group<-cut(eval$prob,seq(1,0,-.1),include.lowest=T)
  xtab<-table(group,eval$y)
  
  #make empty dataframe
  KS<-data.frame(Group=numeric(10),
                 CumPct0=numeric(10),
                 CumPct1=numeric(10),
                 Dif=numeric(10))
  
  #fill data frame with information: Group ID, 
  #Cumulative % of 0's, of 1's and Difference
  for (i in 1:10) {
    KS$Group[i]<-i
    KS$CumPct0[i] <- sum(xtab[1:i,1]) / sum(xtab[,1])
    KS$CumPct1[i] <- sum(xtab[1:i,2]) / sum(xtab[,2])
    KS$Dif[i]<-abs(KS$CumPct0[i]-KS$CumPct1[i])
  }
  
  KS[KS$Dif==max(KS$Dif),]
  
  maxGroup<-KS[KS$Dif==max(KS$Dif),][1,1]
  
  #and the K-S chart
  p=ggplot(data=KS)+
    geom_line(aes(Group,CumPct0),color="blue")+
    geom_line(aes(Group,CumPct1),color="red")+
    geom_segment(x=maxGroup,xend=maxGroup,
                 y=KS$CumPct0[maxGroup],yend=KS$CumPct1[maxGroup])+
    labs(title = "K-S Chart", x= "Deciles", y = "Cumulative Percent")
  
  print(p)    
  
  #Distribution of predicted probabilities values for true positives and true negatives
  plot(0,0,type="n", xlim= c(0,1), ylim=c(0,7),     
       xlab="Prediction", ylab="Density",  
       main="How well do the predictions separate the classes?")
  
  for (runi in 1:length(pred@predictions)) {
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==1]), col= "blue")
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==0]), col="green")
  }
  
  #Cumulative gains chart , missing: implement extra kink for green line
  pred = prediction(eval$prob, eval$y)
  gain = performance(pred, "tpr", "rpp")  
  plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
       ylab="Percent of target captured", 
       xlab="Percent of population")
  lines(x=c(0,floor(sum(eval$y ==1)/(length(eval$y)/10))*0.1,(sum(eval$y==1) %% (length(eval$y)/10))/length(eval$y)+(floor(sum(eval$y ==1)/(length(eval$y)/10))*0.1) , 1), y=c(0, ((length(eval$y)/10)*(floor(sum(eval$y ==1)/(length(eval$y)/10))))/sum(eval$y==1),1, 1), col="darkgreen", lwd=2)
  
  gain.x = unlist(slot(gain, 'x.values'))
  gain.y = unlist(slot(gain, 'y.values'))
  
  lines(x=gain.x, y=gain.y, col="orange", lwd=2)
  
  return (list(confusionMatrix(eval$pred, eval$y, positive="1"), #confustion matrix function and assessment metrics 
               "Percent Concordance"=PercentConcordance,
               "Percent Discordance"=PercentDiscordance,
               "Percent Tied"=PercentTied,
               "Pairs"=Pairs,
               "AUC"=AUC,
               "D statistic"= dstat,
               "Log Loss"=lloss))
}    

#logistic regression
#data preprocessing for logistic regression
dataset = data.frame(dataset)
dataset$id_issuer = as.factor(dataset$id_issuer)
dataset$amount_group = as.factor(dataset$amount_group)
dataset$pos_entry_mode = as.factor(dataset$pos_entry_mode)
dataset$is_upscale = as.factor(dataset$is_upscale)
dataset$mcc_group = as.factor(dataset$mcc_group)
dataset$type = as.factor(dataset$type)
dataset$country_code = as.factor(dataset$country_code)
dataset$datetime = as.numeric(as.POSIXct(dataset$datetime))


fit <- glm(data=dataset, is_fraud ~ ., family="binomial")
summary(fit)
Evaluation(dataset$is_fraud, fit$fitted.values)

#Confusion Matrix and Statistics

#             Reference
#Prediction     0     1
#0            39915  2005
#1             375   705

#logistic regression with aggregated features
#data preprocessing for logistic regression
dataset_agg = data.frame(dataset_agg)
dataset_agg$id_issuer = as.factor(dataset_agg$id_issuer)
dataset_agg$amount_group = as.factor(dataset_agg$amount_group)
dataset_agg$pos_entry_mode = as.factor(dataset_agg$pos_entry_mode)
dataset_agg$is_upscale = as.factor(dataset_agg$is_upscale)
dataset_agg$mcc_group = as.factor(dataset_agg$mcc_group)
dataset_agg$type = as.factor(dataset_agg$type)
dataset_agg$country_code = as.factor(dataset_agg$country_code)
dataset_agg$datetime = as.numeric(as.POSIXct(dataset_agg$datetime))


fit <- glm(data=dataset_agg, is_fraud ~ ., family="binomial")
summary(fit)
Evaluation(dataset_agg$is_fraud, fit$fitted.values)


#elastic net regularized logistic regression model
#preprocessinf for elastic net
dataset = data.frame(dataset)
dataset$id_issuer = as.factor(dataset$id_issuer)
dataset$amount_group = as.factor(dataset$amount_group)
dataset$pos_entry_mode = as.factor(dataset$pos_entry_mode)
dataset$mcc_group = as.factor(dataset$mcc_group)
dataset$datetime = as.numeric(as.POSIXct(dataset$datetime))

dataset_m = dummy.data.frame(dataset)
dataset_m = as.matrix(dataset_m)
dataset_m = apply(dataset_m,2, as.numeric)

#apply caret package to determine lambda
fitControl <- trainControl(method="repeatedcv",number=10, repeats=5,verboseIter = TRUE)
enetGrid <- expand.grid(lambda=seq(0,0.001,length=10),
                        alpha = seq(0.1,0.9,length=10))

fit.glmnet <- train(dataset_m[,1:44],
                    as.factor(dataset_m[,45]),
                    method="glmnet",
                    trControl=fitControl,
                    tuneGrid=enetGrid)
fit.glmnet
plot(fit.glmnet)

#The final values used for the model were alpha = 0.9 and lambda = 0.0005555556
#Evaluation

prob = predict(fit.glmnet, dataset_m[,1:44], s= 0.0005555556, type = "prob")[,2] #predicted probability, here for glmnet
Evaluation(dataset$is_fraud, prob)

#Confusion Matrix and Statistics
#         Reference
#Prediction 0     1
#0        39945  2092
#1         345   618

#decision tree--------------
#data preprocessing for decision tree
dataset = data.frame(dataset)
dataset$id_issuer = as.factor(dataset$id_issuer)
dataset$amount_group = as.factor(dataset$amount_group)
dataset$pos_entry_mode = as.factor(dataset$pos_entry_mode)
dataset$is_upscale = as.factor(dataset$is_upscale)
dataset$mcc_group = as.factor(dataset$mcc_group)
dataset$type = as.factor(dataset$type)
dataset$country_code = as.factor(dataset$country_code)
dataset$datetime = as.numeric(as.POSIXct(dataset$datetime))
dataset$is_fraud = as.factor(dataset$is_fraud)

tree.1 <- rpart(data=dataset, is_fraud~.,control=rpart.control(cp=0.0001))
printcp(tree.1)
plotcp(tree.1)

# based on the plot:  the min CV error is with cp=0.00058 
# (which produces a tree with 202 leaf nodes)--> overfitting?
pfit<-prune(tree.1,cp=0.00058)

pred = predict(pfit, type="class")
confusionMatrix(pred, dataset$is_fraud)

# assess the classifier
prediction.decisiontree = predict(pfit, newdata=dataset, type = "prob")[,2]

Evaluation(dataset_m[,45], prediction.decisiontree)

#Confusion Matrix and Statistics

#          Reference
#Prediction 0     1
#0        40030  1059
#1         260  1651

#random forest-----------------

tuneGrid = expand.grid(.mtry = c(1:12))

fitControl <- trainControl(method="repeatedcv",number=10, repeats=5, verboseIter = TRUE)

rf_model<-train(is_fraud~.,data=dataset,
                method="rf",
                trControl=fitControl,
                tuneGrid = tuneGrid,
                ntree = 20)

plot(rf_model)


#now build single model with ntree = 1500 and with optimal mtry = 12
dataset$country_code = as.numeric(dataset$country_code)
rf_final <- randomForest(is_fraud ~ ., data = dataset, ntrees=1500, mtry=12)


pred_fr = predict(rf_final, type="class")
confusionMatrix(pred_fr, dataset$is_fraud)

# assess the classifier
prediction.randomforest = predict(rf_final, newdata=dataset, type = "prob")[,2]

Evaluation(dataset_m[,45], prediction.randomforest)

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
Evaluation(dataset_m[,45], prediction.boostedtree)

#Confusion Matrix and Statistics

#           Reference
#Prediction  0     1
#0         39837  1725
#1          453   985

#svm----
gc()
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))

#with linear kernel
svm_Linear <- train(is_fraud ~., data = dataset, 
                    method = "svmLinear",
                    trControl=fitControl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
tuneGrid = grid
test_pred <- predict(svm_Linear, newdata = dataset)

#with Non-Linear Kernel (Radial Basis Function)
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                             C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2,5))

svm_Radial_Grid <- train(is_fraud ~., data = dataset, 
                           method = "svmRadial",
                           trControl=trctrl,
                           preProcess = c("center", "scale"),
                           tuneGrid = grid_radial,
                           tuneLength = 10)






