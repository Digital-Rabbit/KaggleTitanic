## Titanic Competition
## B. Laden
## 1 October 2014
## Updated 18 Dec 2015 with prediction 10
## Titanic.R
## B. Laden
## Code used to learn feature engineering and the submission process for
## Kaggle competitions
## Most code is from following an amazing tutorial by Trevor Stephens
## http://trevorstephens.com
## See: http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r

require(rattle)
require(rpart.plot)
require(RColorBrewer)
require(randomForest)
require(party)

## data available from www.kaggle.com
## read training and test sets
titanicTrain <- read.csv("train.csv")
titanicTest <- read.csv("test.csv")

## look at dataframe structure
str(titanicTrain)

## see who survived
titanicTrain$Survived

## table of who survived tells the grim stats
table(titanicTrain$Survived)

## proportion survival 
prop.table(table(titanicTrain$Survived))

############ Prediction 1: Everyone dies  ############

## More perished then not, so it's a start
## Add a prediction column to the test data that indicates everyone died
titanicTest$Survived <- rep(0, 418)

## extract the information necessary to send to Kaggle for a prediction
submit <- data.frame(PassengerId = titanicTest$PassengerId, 
                     Survived = titanicTest$Survived)
write.csv(submit, file = "theyallperish.csv", row.names = FALSE)

############ Prediction 2: Gender plays a rols ############

## women and children supposedly evacuated first
summary(titanicTrain$Sex)

## look at gender and survival as a proportion table
prop.table(table(titanicTrain$Sex, titanicTrain$Survived),1)

## add a prediction column to the test data
titanicTest$Survived <- 0

## update the predictions to indicate females survived
titanicTest$Survived[titanicTest$Sex == 'female'] <- 1

## prepare for Kaggle submission
submit <- data.frame(PassengerId = titanicTest$PassengerId, 
                     Survived = titanicTest$Survived)
write.csv(submit, file = "womensurvive.csv", row.names = FALSE)

############ Prediction 3: Age and ticket price ############

## explore how age fits into the prediction
summary(titanicTrain$Age)

## there are 177 missing age values
## proceed with the assumption the missing values are the average

## distinguish children from adults
## create a column to capture children, i.e. < 18 years of age
## the missing values will be categorized as adult, i.e., the average age
titanitcTrain$Child <- 0
titanicTrain$Child[titanicTrain$Age < 18] <- 1

## the boolean  evaluates all adults to NA, so clean up the NA values
titanicTrain$Child[titanicTrain$Age >= 18] <- 0
titanicTrain$Child[is.na(titanicTrain$Age)] <- 0

## create a table of both gender and age and see who survived
## this shows the total number of survivors
aggregate(Survived ~ Child + Sex, data=titanicTrain, FUN=sum)

## look at the total number of people in each subset
aggregate(Survived ~ Child + Sex, data=titanicTrain, FUN=length)

## now look at the proportions
aggregate(Survived ~ Child + Sex, data=titanicTrain, 
          FUN=function(x) {sum(x)/length(x)})

## look at class and cost of ticket
## first bin the fares into 4 bins
## the bins are arbitrary, but this will do for now
titanicTrain$Fare2 <- '30+'
titanicTrain$Fare2[titanicTrain$Fare < 30 & titanicTrain$Fare >= 20] <- '20-30'
titanicTrain$Fare2[titanicTrain$Fare < 20 & titanicTrain$Fare >= 10] <- '10-20'
titanicTrain$Fare2[titanicTrain$Fare < 10] <- '<10'

## aggregate the data and look for possible predictors
aggregate(Survived ~ Fare2 + Pclass + Sex, data=titanicTrain, 
          FUN=function(x) {sum(x)/length(x)})

## class 3 women who paid more than $20 didn't do as well as others
## figure this into the prediction
titanicTest$Survived <- 0
titanicTest$Survived[titanicTest$Sex == 'female'] <- 1
titanicTest$Survived[titanicTest$Sex == 'female' & titanicTest$Pclass == 3 & titanicTest$Fare >= 20] <- 0

## prepare for Kaggle submission
submit <- data.frame(PassengerId = titanicTest$PassengerId, 
                     Survived = titanicTest$Survived)
write.csv(submit, file = "genderclassurvive.csv", row.names = FALSE)


############ Prediction 4: Build a decision tree ############

## import the rpart library for recursive partitioning and regression trees
library(rpart)

## build a decision tree from all variables except the identifiers
## such as, name cabin number, and so on
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
             data=titanicTrain, method="class")
## look at the fit data
plot(fit)
text(fit)

## not too interpretable, so install other packages
## create a fancier plot
fancyRpartPlot(fit)

## prepare for Kaggle submission
Prediction <- predict(fit, titanicTest, type = "class")
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)


############ Prediction 5: This is an example of overfitting  ############

## customize rpart to control splits and when splits stop
## first max out the cp (when to stop) and the minsplit (when to split) variables
## this actually overfits, so I am not going to submit! 
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
             data=titanicTrain, method="class", 
             control=rpart.control(minsplit=2, cp=0))
fancyRpartPlot(fit)

## a more reasonable customization TO DO 
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
             data=titanitTrain,
             method="class", control=rpart.control( your controls ))
new.fit <- prp(fit,snip=TRUE)$obj
fancyRpartPlot(new.fit)

## Not submitting this!

############ Prediction 6: Social status and class ############

## engineer features to capture social status
## look at titles to see if rank/class can predict survival
## merge training and test sets prior to creating a new variable
titanicTrain <- read.csv("train.csv")
titanicTest <- read.csv("test.csv")

titanicTest$Survived <- NA
combi <- rbind(titanicTrain, titanicTest)

## cast name column as a character column
combi$Name <- as.character(combi$Name)

## split the string, using comma and full stops as delimiters
strsplit(combi$Name[1], split='[,.]')

## need to go a level deeper to extract the part of interest, due to the
## way strsplit nests containers
strsplit(combi$Name[1], split='[,.]')[[1]][2]

##uUse sapply to run through all rows and split the string to extract title
## put result into a new column
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

## strip off the leading spaces
combi$Title <- sub(' ', '', combi$Title)

## see how many in each category
table(combi$Title)

## reduce some of the titles by combining
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'

## combine the rich folks
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

## change the title to a factor
combi$Title <- factor(combi$Title)

## look at family size by combining sibline and parch data, add one for the person
combi$FamilySize <- combi$SibSp + combi$Parch + 1

## factor in surname, maybe same names are related?
## extract the surname
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

## convert family size to string, combine with surname, output as factor
## this gives a familyID factor
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")

## families 2 or less, classify as small
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'

## check what the familyID looks like
table(combi$FamilyID)

## clean up for the small families that "slipped through the cracks"
famIDs <- data.frame(table(combi$FamilyID))

## subset to see which families didn't fit the assumptions
famIDs <- famIDs[famIDs$Freq <= 2,]

## overwrite what needs to be changed and convert to a factor
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

## now that the engineered variables are set, split about the training
## and test sets
titanicTrain <- combi[1:891,]
titanicTest <- combi[892:1309,]

## try fitting the data with the engineered features
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + 
                 Fare + Embarked + Title + FamilySize + FamilyID,
                  data=titanicTrain, method="class")

## create a fancier plot
fancyRpartPlot(fit)

## prepare for Kaggle submission
Prediction <- predict(fit, titanicTest, type = "class")
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction)
write.csv(submit, file = "engineeredfeatures.csv", row.names = FALSE)


############  Prediction 7: Random forest  ############ 

## grow a tree on the subset of the data with the age values available, 
## and then replace those that are missing
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + 
                    Title + FamilySize,
                  data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

## check for other issues. Randon forest can't tolerate NA
summary(combi)
summary(combi$Embarked)
## replace embarked blanks
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

## fix the mising fare
summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

## Random Forests in R can only digest factors with up to 32 levels.
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

## install package
install.packages('randomForest')
library(randomForest)
set.seed(415)

## split the larger set back to training and test
titanicTrain <- combi[1:891,]
titanicTest <- combi[892:1309,]
## fit the model
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                        Parch + Fare + Embarked + Title + FamilySize +
                        FamilyID2, 
                        data=titanicTrain, importance=TRUE, ntree=2000)
## look at the important variables
varImpPlot(fit)
## make the prediction, but don't submit yet!  To be continued!
Prediction <- predict(fit, titanicTest)
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

############  Prediction 8: Conditional Inference Trees ############ 
install.packages('party')
library(party)
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                   Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                  data = titanicTrain, 
                  controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, titanicTest, OOB=TRUE, type = "response")
## Prepare for submission
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction)
write.csv(submit, file = "conditionalforest.csv", row.names = FALSE)

############  Prediction 9: Add a Deck Variable ############ 

## this is my own doing. I'm trying to improve on the ranking I achieved
## through Trevor's tutorial. I improved my ranking with this!

setwd("~/Desktop/Kaggle/TitanicCompetition")

## create a deck variable. deck data is incomplete, so I add a deck when I
## knew the deck and then coded the unknown as deck Z.  
combiDeck <- read.csv("combideck.csv")
combiDeck$Deck 

## split into training and testing
titanicTrain <- combiDeck[1:891,]
titanicTest <- combiDeck[892:1309,]
## now look at the proportions
aggregate(Survived ~ Deck + Sex, data=titanicTrain, FUN=function(x) {sum(x)/length(x)})
## Run the model

set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                   Parch + Fare + Embarked + Title + FamilySize + Deck,
               data = titanicTrain, 
               controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, titanicTest, OOB=TRUE, type = "response")

## Prepare for submission
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction)
write.csv(submit, file = "condforestDeckNoFamID.csv", row.names = FALSE)

############  Prediction 10: Use Deck Variable and FamilyID ############ 

## Try to improve on deck variable alone by adding back in the Family ID
## previously engineered
## I did not improve my ranking with this
set.seed(415)
fit2 <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                   Parch + Fare + Embarked + Title + FamilySize + FamilyID2 + Deck,
               data = titanicTrain, 
               controls=cforest_unbiased(ntree=2000, mtry=3))
## Did not improve 
fit2 <- cforest(as.factor(Survived) ~ Pclass + Sex + Age  + Fare + Embarked + Title + FamilySize + FamilyID2 + Deck,
                data = titanicTrain, 
                controls=cforest_unbiased(ntree=2000, mtry=3))
##
fit3 <- cforest(as.factor(Survived) ~ Pclass + Sex + Age  + SibSp + 
                    Parch + Fare + Title  + FamilyID2 + Deck,
                data = titanicTrain, 
                controls=cforest_unbiased(ntree=2000, mtry=3))
##
fit4 <- cforest(as.factor(Survived) ~ Pclass + Sex + Age  + SibSp + 
                    Parch + Fare + Embarked + Title  + FamilyID2 + FamilySize + Deck,
                data = titanicTrain, 
                controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction10b <- predict(fit4, titanicTest, OOB=TRUE, type = "response")

## Prepare for submission
submit <- data.frame(PassengerId = titanicTest$PassengerId, Survived = Prediction10b)
write.csv(submit, file = "prediction10c.csv", row.names = FALSE)
