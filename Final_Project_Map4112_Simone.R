# MAP4112 Project #
# Group Members: Sarah Halverson & Nathaniel Simone #
# Load Library needed
library(MASS)
library(class)

###############################################################################
#Get the data directly from the link
df_red <- read.delim("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";", header = TRUE)

#Retrieve data of white wines from link
df_white <- read.delim("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";", header = TRUE)

#Merge the two datasets
df = rbind(df_red, df_white)

#Standardize Each Variable
df$fixed.acidity <- (df$fixed.acidity - mean(df$fixed.acidity))/sd(df$fixed.acidity)
df$volatile.acidity <- (df$volatile.acidity - mean(df$volatile.acidity))/sd(df$volatile.acidity)
df$citric.acid <- (df$citric.acid - mean(df$citric.acid))/sd(df$citric.acid)
df$residual.sugar <- (df$residual.sugar - mean(df$residual.sugar))/sd(df$residual.sugar)
df$chlorides <- (df$chlorides - mean(df$chlorides))/sd(df$chlorides)
df$free.sulfur.dioxide <- (df$free.sulfur.dioxide - mean(df$free.sulfur.dioxide))/sd(df$free.sulfur.dioxide)
df$total.sulfur.dioxide <- (df$total.sulfur.dioxide - mean(df$total.sulfur.dioxide))/sd(df$total.sulfur.dioxide)
df$density <- (df$density - mean(df$density))/sd(df$density)
df$pH <- (df$pH - mean(df$pH))/sd(df$pH)
df$sulphates <- (df$sulphates - mean(df$sulphates))/sd(df$sulphates)
df$alcohol <- (df$alcohol - mean(df$alcohol))/sd(df$alcohol)
###############################################################################
#Split into training and Test Group
set.seed(1)
train=sample(1:nrow(df),nrow(df)/2)
test=(-train)
Train = df[train,]
Test = df[test,]

###############################################################################
#Linear Discriminant Analysis
Linear_DA <- function(Training, Testing){
  lda.fit = lda(quality ~  .,
                data = Training)
  print(lda.fit)
  
  #Testing Error Rate
  lda.pred = predict(lda.fit, Testing)
  lda.class=lda.pred$class
  print(table(lda.class, Testing$quality))
  print(mean(lda.class != Testing$quality))
  
  #Training Error Rate
  lda.pred = predict(lda.fit, Training)
  lda.class=lda.pred$class
  print(table(lda.class, Training$quality))
  print(mean(lda.class != Training$quality))
}

start <- Sys.time()
Linear_DA(Train, Test)
end <- Sys.time()
print(end-start)
###############################################################################
#Quadratic Discriminant Analysis
Test2 <- subset(Test, Test$quality != 9)
Train2 <- subset(Train, Train$quality != 9)

Quad_DA <- function(Training, Testing){
  qda.fit = qda(quality ~  .,
                data = Training)
  print(qda.fit)
  
  #Testing Error Rate
  qda.pred = predict(qda.fit, Testing)
  qda.class=qda.pred$class
  print(table(qda.class, Testing$quality))
  print(mean(qda.class != Testing$quality))
  
  #Training Error Rate
  qda.pred = predict(qda.fit, Training)
  qda.class=qda.pred$class
  print(table(qda.class, Training$quality))
  print(mean(qda.class != Training$quality))
}

start <- Sys.time()
Quad_DA(Train2, Test2)
end <- Sys.time()
print(end-start)
###############################################################################
#KNN
KNN <- function(Training, Testing, i){
  set.seed(2)
  start <- Sys.time()
  knn.pred=knn(train = Training, test = Testing, Training$quality, k=i)
  
  #Testing Error Rate
  table(knn.pred, Testing$quality)
  knn.table <- table(knn.pred, Testing$quality)
  print(knn.table)
  
  knn.table <- as.matrix(knn.table)
  trace <- sum(diag(knn.table))
  total <- sum(knn.table)
  error <- 1 - (trace/total)
  print(error)
  
  #Training Error Rate
  knn.pred=knn(train = Training, test = Training, Training$quality, k=i)
  table(knn.pred, Training$quality)
  knn.table <- table(knn.pred, Training$quality)
  print(knn.table)
  
  knn.table <- as.matrix(knn.table)
  trace <- sum(diag(knn.table))
  total <- sum(knn.table)
  error <- 1 - (trace/total)
  print(error)
  
  end <- Sys.time()
  print(end-start)
}

KNN(Train, Test, 1)
KNN(Train, Test, 5)
KNN(Train, Test, 10)
KNN(Train, Test, 25)

###############################################################################
#SVM - Radial
start <- Sys.time()
library(e1071)
Factor_test = Test
Factor_train = Train
Factor_test$quality = as.factor(Factor_test$quality)
Factor_train$quality = as.factor(Factor_train$quality)

tuning = tune(svm, quality ~  .,
                data = Factor_train, kernel="radial",
                ranges = list(cost=c(0.01, 0.1, 1, 5, 10), gamma = c(.5, 1, 2, 3, 4)))
summary(tuning)

svm.fit=svm(quality ~  .,
              data = Factor_train, method = "C-classification", kernel="radial", cost=5, gamma = .5, scale = FALSE)
summary(svm.fit)

#Testing Error Rate
pred = predict(svm.fit, Factor_test)
svm.table <- table(predict = pred, truth=Factor_test$quality)
print(svm.table)

svm.table <- as.matrix(svm.table)
trace <- sum(diag(svm.table))
total <- sum(svm.table)
error <- 1 - (trace/total)
print(error)

#Training Error Rate
pred = predict(svm.fit, Factor_train)
svm.table <- table(predict = pred, truth=Factor_train$quality)
print(svm.table)

svm.table <- as.matrix(svm.table)
trace <- sum(diag(svm.table))
total <- sum(svm.table)
error <- 1 - (trace/total)
print(error)

end <- Sys.time()S
print(end-start)
###############################################################################
#SVM- Linear
start <- Sys.time()
tuning = tune(svm, quality ~  .,
              data = Factor_train, kernel="linear",
              ranges = list(cost=c(0.01, 0.1, 1, 5, 10)))
summary(tuning)


svm.fit=svm(quality ~  .,
            data = Factor_train, method = "C-classification", kernel="linear", cost=10, scale = FALSE)
summary(svm.fit)

#Testing Error Rate
pred = predict(svm.fit, Factor_test)
svm.table <- table(predict = pred, truth=Factor_test$quality)
print(svm.table)

svm.table <- as.matrix(svm.table)
trace <- sum(diag(svm.table))
total <- sum(svm.table)
error <- 1 - (trace/total)
print(error)

#Training Error Rate
pred = predict(svm.fit, Factor_train)
svm.table <- table(predict = pred, truth=Factor_train$quality)
print(svm.table)

svm.table <- as.matrix(svm.table)
trace <- sum(diag(svm.table))
total <- sum(svm.table)
error <- 1 - (trace/total)
print(error)

end <- Sys.time()
print(end-start)

###############################################################################

