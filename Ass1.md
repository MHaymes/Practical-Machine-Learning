Predicting Dumbell Exercise Classes using Accelerometer Data 
=============================================================

The purpose of this analysis is to predict exercise movement classes using data collected as part of the Weight Lifting Exercise Dataset (WLE Dataset). The dataset includes information derived from wearable sensors employed on six human participants while they performed barbell exercises.   The dataset was developed using accelerometers on the belt, forearm, arm, and dumbell.  The participants performed barbell lifts correctly and incorrectly in five different lift exercises.   The purpose of the study is to use machine learning techniques, specifically random forest prediction methods to correctly classify the exercises in a validation set of 20 individual exercise classes. This study is being performed as an assignment for the practical machine learning course offered by John Hopkins University's Bloomberg School of Public Health, through the Coursera online learning platform.  The analysis was conducted using the R programming language. 

## Data

From the initial 160 variables included in the WLE Dataset, variables that contained mostly missing or zero values were removed, leaving a dataframe of 51 variables plus the classe variable representing the exercise being performed. Of the over 19,000 initial data rows, each corresponding to a measurement set at a given time, a random subsample of 3,000 observations was drawn.  This was done to reduce memory requirements and computing time in performing the analysis. This may ultimately reduce the accuracy of the final model; however, it was believed that the remaining 3000 observations would preduce a sufficiently accurate prediction model. 



```r

# load required packages
library(caret)
library(ggplot2)

# Read the training and validation datasets from local working directory
trainDF <- read.csv("pml-training.csv")
testDF <- read.csv("pml-testing.csv")
head(testDF)

# drop those variables that are mostly NAs
trainDF <- trainDF[sapply(testDF, function(testDF) !any(is.na(testDF)))]
testDF <- testDF[sapply(testDF, function(testDF) !any(is.na(testDF)))]

# drop variables that are mostly blank values
drop <- c(1, grep("kurt", names(trainDF)), grep("skew", names(trainDF)), grep("amplitude", 
    names(trainDF)), grep("yaw", names(trainDF)), grep("timestamp", names(trainDF)))
trainDF <- trainDF[, -drop]
testDF <- testDF[, -drop]

# take a sample of 3000 rows from the original dataframe (this is done to
# limit memory usage and improve runtime)
set.seed(1234)
trainDF <- trainDF[sample(1:nrow(trainDF), 3000, replace = FALSE), ]
```


Model Design
=============

A Random Forest classification approach was employed to predict the exercise class of the participants.  The design is based on supervised learning from the WLE dataset, with final predictions made on a sample of 20 exercises performed for which the class is unknown.  All 51 variables were used as features in the prediction model, with the exercise class variable (classe) used as the outcome to be predicted. 

## Cross Validation

The training sample was split into both a testing and traning set, with the final testing dataset (the pml-testing.csv file) being used as a final validation set.  This choice was driven by the large number of observations available.  The test and training datasets were created from a random sample of 3000 rows of data, with 70% of the data assigned to the training class.  


```r

# create a cross-validation training and test set from the training dataset
set.seed(1234)
inTrain <- createDataPartition(y = trainDF$classe, p = 0.7, list = FALSE)
training <- trainDF[inTrain, ]
testing <- trainDF[-inTrain, ]
```




```r

# Train the random forest model on the training data.
modFit <- train(classe ~ ., data = training, method = "rf", prox = TRUE)
```


## Out of Sample Error

The out of sample error is estimated using prediction on the testing dataset.  This is based on 897 random observations assigned from the training-pml dataset.  The findings (see below) estimate a final out-of-sample accuracy of approximately 96%.  This varies by exercise class, with Class B type of exercise having the lowest overall sensitivity (true positive rate), at approximately 92%.  



```r

# Predict classe variable using the testing data.
predictions <- predict(modFit, newdata = testing)
```



```r
# Display model performance.
confusionMatrix(predictions, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 254   6   0   0   0
##          B   0 160   6   0   2
##          C   1   5 158   5   1
##          D   0   2   1 137   4
##          E   0   0   0   2 153
## 
## Overall Statistics
##                                         
##                Accuracy : 0.961         
##                  95% CI : (0.946, 0.973)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.951         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.925    0.958    0.951    0.956
## Specificity             0.991    0.989    0.984    0.991    0.997
## Pos Pred Value          0.977    0.952    0.929    0.951    0.987
## Neg Pred Value          0.998    0.982    0.990    0.991    0.991
## Prevalence              0.284    0.193    0.184    0.161    0.178
## Detection Rate          0.283    0.178    0.176    0.153    0.171
## Detection Prevalence    0.290    0.187    0.190    0.161    0.173
## Balanced Accuracy       0.993    0.957    0.971    0.971    0.977
```



## Predicting the Validation Set

Finally, the random forest model is used to predict the exercise classes from the validation set (testing-pml.csv).  These predictions are assigned to a vector that will be used to evaluate the accuracy of the final model in generating predictions. 


```r
# run the predictions on the final validation dataframe.

predictionsFinal <- predict(modFit, testDF)
predictionsFinal
```


## Study Design Rationale

#### Model Design
The random forest model design was chosen due to its relatively high degree of accuracy for out-of-sample predictions with a large number of features.   

#### Pre-Processing
Pre-processing of the data was not employed for this exercise.  Pre-processing using a Principal Component Analysis approach was initially considered as a possible option to reduce the number of features required for the analysis.  However, it was believed that retaining the entire set of features would, in the end, lead to better overall accuracy of predictions in the final validation set.  

## Conclusions

Overall, the random forest predictive model employed appears to have relatively high accuracy against the random test set.  Additional factors, such as the use of pre-processing, or removal of features to reduce noise and the possibility of overfitting could be examined to strengthen the overall strength of the work and reduce computing resource requirements. 
