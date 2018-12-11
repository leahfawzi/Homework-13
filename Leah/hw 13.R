library(nnet)
library(caret)
library(dplyr)

mtrain <- read.csv("mnist_train.csv", header=F) %>% as.matrix
train_classification <- mtrain[,1]

mtrain<- mtrain[,-1]/256 #x matrix
x <- mtrain[1:1000,]
y <- factor(train_classification[1:1000], levels=0:10,labels=c(0,0,0,1,0,0,0,0,0,0,0)) %>% factor
colnames(x)<- 1:784


# fit the data to a neural net first with decay 0
fitControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2)

tuning_df <- data.frame(size=9:12, decay=0)

t_out <- caret::train(x=x, y=y, method="nnet",
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=1000, MaxNWts=100000)
#optimal size was 11 --> Accuracy was 0.9645

tuning_df2 <- data.frame(size=11, decay=c(0,0.5,1,2))
t_out2 <- caret::train(x=x, y=y, method="nnet",
                      trControl = fitControl,
                      tuneGrid=tuning_df2, maxit=1000, MaxNWts=100000)

#optimal decay was 1 --> Accuracy = 0.9655109

# check against train2

mtest <- read.csv("mnist_test.csv",header=F) %>% as.matrix
x2 <- mtest

train_classification2 <- mtest[,1]
mtest<- mtest[,-1]/256 #x matrix
y2 <- factor(train_classification2, levels=0:10,labels=c(0,0,0,1,0,0,0,0,0,0,0)) %>% factor
colnames(mtest)<- 1:784

true_y <- y2
pred_y1 <- predict(t_out, newdata = mtest)
pred_y2 <- predict(t_out2, newdata = mtest)


n_samples <- nrow(x2)
error1 <- sum(true_y != pred_y1)/n_samples
cat("test prediction error with t_out", error1, "\n")

error2 <- sum(true_y != pred_y2)/n_samples
cat("test prediction error with t_out2", error1, "\n")



