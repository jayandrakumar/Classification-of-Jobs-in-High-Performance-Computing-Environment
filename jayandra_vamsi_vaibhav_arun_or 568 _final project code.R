#obtaining data from package applied predictive modelling 
library(AppliedPredictiveModeling)
#data about HPC scheduler
data("schedulingData")

library(nnet)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # Package for svm
library(pROC)
library(AUC)

#setting seed to get random sample same for every run
set.seed(2017)
#dividing data into training and testing with 60% and 40% of total data respectively.
trainfit=sample(nrow(schedulingData), nrow(schedulingData)*0.6,replace = FALSE)
training=schedulingData[trainfit,]
testing=schedulingData[-trainfit,]

#mulitnomial logistic regression
mlr<-multinom(Class~Protocol+Compounds+InputFields+Iterations+NumPending+Hour+Day,data = training)
mlr
summary(mlr)

#predicitng for multinomial logistic Regression
mlr_predict=predict(mlr,newdata=testing)

#confusion matrix for predicted values of multinomial logistic Regression
confusionMatrix(mlr_predict,testing$Class)

#area under curve for multinomial logistic Regression
mlr_auc=multiclass.roc(as.numeric(mlr_predict),as.numeric(testing$Class))
mlr_auc$auc

#Roc curve for multinomial logistic Regression
plot(roc(mlr_predict,testing$Class),main=c(paste("ROC Curves\nAUC:                      "),paste(round(mlr_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")


#backward for multinomial logistic Regression
backward=step(mlr,direction = "backward")
backward

#predicting for backward multinomial logistic Regression
backward_predict=predict(backward,newdata=testing)

#Confusion matrix for backward  multinomial logistic Regression
confusionMatrix(backward_predict,testing$Class)

#Area under Curve for backward multinomial logistic Regression
backward_auc=multiclass.roc(as.numeric(backward_predict),as.numeric(testing$Class))
backward_auc$auc

#Roc Curve for backward multinomial logistic Regression
plot(roc(backward_predict,testing$Class),main=c(paste("ROC Curves\nAUC:                      "),paste(round(backward_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")



#loading package for naive bayes
library(naivebayes)
#performing naive bayes on trainig data
nb<-naive_bayes(Class~.,data = training)
nb

#predicitng naive bayes
nb_predict=predict(nb,newdata=testing)

#refactoring the data
nb_predict=factor(nb_predict,levels = c("VF","F","M","L"))

#confusion matrix for naive bayes
confusionMatrix(nb_predict,testing$Class)

#Area under Curve for Naive bayes
nb_auc=multiclass.roc(as.numeric(nb_predict),as.numeric(testing$Class))
nb_auc$auc

#Roc Curve for  Naive Bayes
plot(roc(nb_predict,testing$Class),main=c(paste("ROC Curves AUC:"),paste(round(nb_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")

#loading package for bagging
library(adabag)
#bagging
bag=bagging(Class~.,data = training,boos=TRUE,mfinal = 200)
summary(bag)
bag
#predicting for bagging
pre= predict.bagging(bag,newdata = testing)

#loading a predicted variables to a variable
bag_pre=pre$class

#refactoring the data
bag_pre=factor(bag_pre,levels = c("VF","F","M","L"))

#confusion matrix for bagging
confusionMatrix(bag_pre,testing$Class)

#area under curve for bagging
bag_auc=multiclass.roc(as.numeric(as.factor(bag_pre)),as.numeric(testing$Class))
bag_auc$auc

#Roc Curve for  Bagging
plot(roc(bag_pre,testing$Class),main=c(paste("ROC Curves AUC:"),paste(round(bag_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")

# cross validation for bagging with 10folds
bag_cv=bagging.cv(Class~.,data = schedulingData,v=5,mfinal = 200)
bag_cv

#random forest with m=sqrt(p) since it is classification
library(randomForest)
rf<-randomForest(Class~.,data = training,mtry=round(sqrt(7)))
rf

#prediction for random forest
rf_predict=predict(rf,newdata = testing)

#confusion matrix for random forest
confusionMatrix(rf_predict,testing$Class)

#Area under curve for random Forest
rf_auc=multiclass.roc(as.numeric(rf_predict),as.numeric(testing$Class))
rf_auc$auc

#Roc cuurve for RAndom forest
plot(roc(rf_predict,testing$Class),main=c(paste(" Random Forest\nROC Curves\nAUC:                      "),paste(round(rf_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")

#variable importance plot for random foresr
varImpPlot(rf)

#random forest with 10 fold cross validation
rf_cv<-rfcv(schedulingData[,-8],schedulingData[,8],cv.fold = 5)
rf_cv$error.cv

#Support vector machine
# CONSTRUCTING THE SVM MODEL (PART - 1)

SVM_BASIC<- svm(Class ~., data = training) # BASIC SVM MODEL WITH DEFAULT PARAMETERS

summary(SVM_BASIC)# SUMMARY OF THE MODEL

prediction_basic <- predict(SVM_BASIC, newdata = testing[1:7]) # PREDICTION FOR THE BASIC MODEL


Accuracy(prediction_basic, testing$Class) #ACCURACY OF THE MODEL
SVM_BASIC_auc <- multiclass.roc(as.numeric(prediction_basic),as.numeric(testing$Class))
SVM_BASIC_auc # AREA UNDER CURVE

CM_BASIC = confusionMatrix(prediction_basic, testing$Class) #CONFUSION MATRIX FOR THE BASIC MODEL
CM_BASIC

#ROC CURVE for basic model
plot(roc(prediction_basic,testing$Class),main=c(paste("ROC Curves for Basic Model   AUC: "),paste(round(SVM_BASIC_auc$auc,3))),col="darkblue",  lwd = 3) # ROC CURVE FOR THE BASIC MODEL
grid(col="aquamarine")



# TUNING OUR MODEL ( PART - 2)

SVM_tune <- tune.svm(Class ~ ., data = schedulingData, kernel='radial',cost=10^(-1:2), gamma=c(.5,1,2)) #TUNING THE MODEL USING GRID SEARCH

# BEST PARAMETERS FOR THE MODEL

SVM_tune$best.parameters

summary(SVM_tune) # SUMMARY OF THE TUNING MODEL

plot(SVM_tune) # PERFORMANCE OF THE TUNING SVM MODEL



# CONTRUCTING THE BEST MODEL

svm_final <- svm(Class ~.,data = training, kernel = 'radial', cost = 100, gamma = 0.5) # SVM MODEL FOR BEST PARAMETERS

summary(svm_final)

prediction_final <- predict(svm_final, newdata = testing[1:7])

CM_TUNED <-  confusionMatrix(prediction_final, testing$Class) # CONFUSION MATRIX FOR THE TUNED MODEL
CM_TUNED

SVM_FINAL_auc <- multiclass.roc(as.numeric(prediction_final),as.numeric(testing$Class))
SVM_FINAL_auc # AREA UNDER CURVE


#ROC CURVE FOR FINAL MODEL
plot(roc(prediction_final,testing$Class),main=c(paste("ROC Curves  AUC: "),paste(round(SVM_FINAL_auc$auc,3))),col="darkblue",  lwd = 3) # ROC CURVE FOR THE BASIC MODEL
grid(col="aquamarine")


### Boosting ###### 
set.seed(2017)
library(adabag) # Loading Adabag package
sd.boost <- boosting(Class ~ .,data = training)
summary(sd.boost)

# Boosting prediction on testing data 
pr.boost <- predict.boosting(sd.boost,newdata = testing)

#refactoring the data
boost_pre=factor(pr.boost$class,levels = c("VF","F","M","L"))

#confusion matrix for boosting
confusionMatrix(boost_pre,testing$Class)

#area under curve for boosting
boost_auc=multiclass.roc(as.numeric(as.factor(boost_pre)),as.numeric(testing$Class))
boost_auc$auc

#Roc Curve for  Boosting
plot(roc(boost_pre,testing$Class),main=c(paste("ROC Curves AUC:"),paste(round(boost_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")

#cross validation of the Boosting model
sd.boost.cv <- boosting.cv(Class ~ ., schedulingData, v = 5)
sd.boost.cv


#variable importance Plot
importanceplot(sd.boost)




#Decision tree 
set.seed(2017)
library(rpart) # loading rpart package 

#training a decision tree on training data
sd.tree <- rpart(Class ~., data = training)
dev.new()
plot(sd.tree) # plotting the tree
text(sd.tree) # adding text to tree
sd.tree 

#Prediction using the decison tree on the testing data
pd.tree.sd <- predict(sd.tree, newdata = testing,type = "class")

#confusion matrix 
confusionMatrix(pd.tree.sd,testing$Class)

#Area under curve for Descision Tree
tree_auc=multiclass.roc(as.numeric(pd.tree.sd),as.numeric(testing$Class))
tree_auc$auc

#Roc cuurve for Descision Tree
plot(roc(pd.tree.sd,testing$Class),main=c(paste(" ROC Curves AUC:  "),paste(round(tree_auc$auc,3))),col="darkblue",  lwd = 3)
grid(col="aquamarine")

