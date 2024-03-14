# LASSO regression
library(glmnet)
X <- as.matrix(df[, setdiff(names(df), "outcome")])
y <- df$outcome
set.seed(123) 
fit <- glmnet(X, y, alpha = 1, family = "binomial")
library(lavaan)
library(RColorBrewer)
cv_fit <- cv.glmnet(X, y, alpha = 1, family = "binomial",nfolds = 10)
plot(cv_fit)
lambda.min<-cv_fit$lambda.min
lambda.min
lambda_1se <- cv_fit$lambda.1se
lambda_1se
coef_cv<-coef(cv_fit, s = "lambda.1se")
coef_cv
min_binomial_deviance <- min(cv_fit$cvm)
std_error_of_min_binomial_deviance <- cv_fit$cvsd[which.min(cv_fit$cvm)]
plot(fit, xvar = "lambda", label = TRUE)
plot(fit, xvar = "lambda", label = F,lwd=1.5,col=my_palette)

#data processing
dt$gender<-as.factor(dt$gender)
dt$headneck<-as.factor(dt$headneck)
dt$thoracic<-as.factor(dt$thoracic)
dt$digestive<-as.factor(dt$digestive)
dt$gynecological<-as.factor(dt$gynecological)
dt$nervous<-as.factor(dt$nervous)
dt$hematologic<-as.factor(dt$hematologic)
dt$metastatic_cancer<-as.factor(dt$metastatic_cancer)
dt$HP<-as.factor(dt$HP)
dt$DM<-as.factor(dt$DM)
dt$puncture_vein<-as.factor(dt$puncture_vein)
dt$position_adjustment<-as.factor(dt$position_adjustment)
dt$incision_infection<-as.factor(dt$incision_infection)
dt$localizedskin_infection<-as.factor(dt$localizedskin_infection)
dt$chemotherapy<-as.factor(dt$chemotherapy)
dt$outcome<-as.factor(dt$outcome)

#machine learning
library(mlr3verse)
library(mlr3)
library(mlr3learners)
library(mlr3misc)
library(dplyr)
library(tidyr)
library(DataExplorer)
library(ggplot2)
library(gridExtra)
et.seed(7832)
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")
## 1.Define task
task_t = as_task_classif(dt, target = "outcome", positive = "1")
## 2.Definition learner
library(e1071)
library(kknn)
# Declarative learner: includes logistic regression, KNN, classification tree, and random forest
learners = list(
  learner_lda = lrn("classif.lda", predict_type = "prob",
                    predict_sets = c("train", "test")),
  learner_nb = lrn("classif.naive_bayes", predict_type = "prob",
                   predict_sets = c("train", "test")),
  learner_knn = lrn("classif.kknn", scale = FALSE,
                    predict_type = "prob"),
  learner_rpart = lrn("classif.rpart",
                      predict_type = "prob"),
  learner_rf = lrn("classif.ranger", num.trees = 1000,
                   predict_type = "prob")
)
## 3.Define parameters
tune_ps_knn = ps(
  k = p_int(lower = 3, upper = 50), # Number of neighbors considered
  distance = p_dbl(lower = 1, upper = 3),
  kernel = p_fct(levels = c("rectangular", "gaussian", "rank", "optimal"))
)
tune_ps_rpart = ps(
  # Minimum number of observations that must exist in a node in order for a
  # split to be attempted
  minsplit = p_int(lower = 10, upper = 40),
  cp = p_dbl(lower = 0.001, upper = 0.1) # Complexity parameter
)
tune_ps_rf = ps(
  # Minimum size of terminal nodes
  min.node.size = p_int(lower = 10, upper = 50),
  # Number of variables randomly sampled as candidates at each split
  mtry = p_int(lower = 1, upper = 6)
)
## 4.AutoTuner
# Oversampling minority class to get perfectly balanced classes
po_over = po("classbalancing", id = "oversample", adjust = "minor",
             reference = "minor", shuffle = FALSE, ratio = 416/167)
table(po_over$train(list(task_t))$output$truth()) # Check class balance
# Learners with balanced/oversampled data
learners_bal = lapply(learners, function(x) {
  GraphLearner$new(po_over %>>% x)
})
lapply(learners_bal, function(x) x$predict_sets = c("train", "test"))

## 5.5-fold cross-validation

resampling_outer = rsmp(id = "cv", .key = "cv", folds = 5L)

# Stratification
task_t$col_roles$stratum = task_t$target_names
## 6.(benchmarking
# Stratification
task_t$col_roles$stratum = task_t$target_names

design = benchmark_grid(
  tasks = task_t,
  learners = c(learners, learners_bal),
  resamplings = resampling_outer
)

bmr = benchmark(design, store_models = FALSE)
bmr2 = bmr
#All learners were compared through the AUC, with or without oversampling, as well as training and test data
measures = list(
  msr("classif.auc", predict_sets = "train", id = "auc_train"),
  msr("classif.auc", id = "auc_test")
)

tab = bmr2$aggregate(measures)
tab_1 = tab[,c('learner_id','auc_train','auc_test')]
print(tab_1)
write.csv(tab_1,"tab1.csv")
## 7.Use box plots to show the predictive performance of all models
# boxplot of AUC values across the 5 folds
autoplot(bmr2, measure = msr("classif.auc"))

library(precrec)
autoplot(bmr2,type = "roc")+
  scale_color_discrete() +
  theme_bw()
autoplot(bmr2, type = "prc")

tab2 = bmr2$aggregate(msrs(c('classif.auc',	'classif.sensitivity','classif.specificity',
                             'classif.fnr',	'classif.fpr')))
tab2 = tab2[,c('learner_id','classif.auc','classif.sensitivity','classif.specificity',
               'classif.fnr',	'classif.fpr')]
print(tab2)
write.csv(tab2,"tab2.csv")

#Through auc comparison, the model with the best performance is selected, in this case it is the random forest model, and the following is the construction
task_t2<-as_task_classif(dt,target = "outcome",positive = "1",id="tivap2")
#Construct a random forest model
learner = lrn("classif.ranger",predict_type="prob")
print(learner)
learner$param_set$values = list(num.trees=100,mtry=5)
learner
#Data set partitioning
spilt <- partition(task_t2,ratio = 0.6, stratify = T)
spilt$train
learner$train(task_t2, row_ids = spilt$train)
print(learner$model)
prediction <- learner$predict(task_t2, row_ids = spilt$test)
print(prediction)
prediction$confusion
#Model performance
measure <- msr("classif.acc")
prediction$score(measure)
measures <- msrs(c('classif.auc',	'classif.sensitivity','classif.specificity',
                   'classif.fnr',	'classif.fpr','classif.ce'))
prediction$score(measures)
conf <- prediction$confusion
print(conf)
autoplot(prediction, type = "roc")
autoplot(prediction, type = "prc")
#Feature selection
set.seed(123)
ll = po("subsample") %>>% lrn("classif.ranger") 
search_space = ps(
  classif.rpart.cp = p_dbl(lower = 0.001, upper = 0.1),
  classif.rpart.minsplit = p_int(lower = 1, upper = 10),
  subsample.frac = p_dbl(lower = 0.1, upper = 1, tags = "budget")
) # tags
instance = TuningInstanceSingleCrit$new(
  task = task_t2,
  learner = ll,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = trm("none"), # hyperband terminates itself
  search_space = search_space
)
library(mlr3hyperband)
tuner <- tnr("hyperband", eta = 3)
lgr::get_logger("bbotk")$set_threshold("warn")


filter = flt("jmim")
task = task_t2
filter$calculate(task)
as.data.table(filter)
filter1<-as.data.table(filter)
write.csv(filter1,"3_.csv")
filter_cor = flt("correlation")
filter_cor$param_set
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set
#Importance feature selection
lrn = lrn("classif.ranger", importance = "impurity")
task <- task_t2
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
filter2<-as.data.table(filter)
write.csv(filter2,"3_2.csv")



task_t3<-as_task_classif(dt2,target = "outcome",positive = "1",id="tivap2")
learner = lrn("classif.ranger",predict_type="prob")
print(learner)
learner$param_set$values = list(num.trees=100,mtry=5)
learner

spilt <- partition(task_t2,ratio = 0.6, stratify = T)
spilt$train
learner$train(task_t3, row_ids = spilt$train)
print(learner$model)
prediction <- learner$predict(task_t3, row_ids = spilt$test)
print(prediction)
prediction$confusion

measure <- msr("classif.acc")
prediction$score(measure)
measures <- msrs(c('classif.auc',	'classif.sensitivity','classif.specificity',
                   'classif.fnr',	'classif.fpr','classif.ce'))
prediction$score(measures)
conf <- prediction$confusion
print(conf)
autoplot(prediction, type = "roc")
autoplot(prediction, type = "prc")

set.seed(123)
ll = po("subsample") %>>% lrn("classif.ranger") 
search_space = ps(
  classif.rpart.cp = p_dbl(lower = 0.001, upper = 0.1),
  classif.rpart.minsplit = p_int(lower = 1, upper = 10),
  subsample.frac = p_dbl(lower = 0.1, upper = 1, tags = "budget")
) 
instance = TuningInstanceSingleCrit$new(
  task = task_t3,
  learner = ll,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = trm("none"), # hyperband terminates itself
  search_space = search_space
)
library(mlr3hyperband)
tuner <- tnr("hyperband", eta = 3)
lgr::get_logger("bbotk")$set_threshold("warn")


filter = flt("jmim")
task = task_t2
filter$calculate(task)
as.data.table(filter)
filter1<-as.data.table(filter)
write.csv(filter1,"5_.csv")
filter_cor = flt("correlation")
filter_cor$param_set
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set

lrn = lrn("classif.ranger", importance = "impurity")
task <- task_t2
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
filter2<-as.data.table(filter)
write.csv(filter2,"4_2.csv")

#Model interpretation
library(iml)
library(mlr3learners)
set.seed(1)
learner = lrn("classif.ranger",predict_type="prob")
learner$train(task_t3)
learner$model
x <- dt2[which(names(dt2) != "outcome")]
model <- Predictor$new(learner, data = x, y = dt2$outcome)
#FeatureEffects
num_features <- setdiff(names(dt2), "outcome")
effect <- FeatureEffects$new(model)
plot(effect, features = num_features)

model <- Predictor$new(learner, data = dt2, y = "outcome")
x.interest <- data.frame(dt[1, ])
shapley <- Shapley$new(model, x.interest = x.interest)
plot(shapley)
#FeatureImp
effect <- FeatureImp$new(model, loss = "ce")
effect$plot(features = num_features)

model <- Predictor$new(learner, data = dt[train_set, ], y = "outcome")
effect <- FeatureEffects$new(model)
plot(effect, features = num_features)

#Independent test data
train_set = sample(task_t3$nrow, 0.8 * task_t3$nrow)
test_set = setdiff(seq_len(task_t3$nrow), train_set)
learner$train(task_t3, row_ids = train_set)
prediction = learner$predict(task_t3, row_ids = test_set)

#First, we compare the importance of features on the training set and the test set
# Training set
model <- Predictor$new(learner, data = dt2[train_set, ], y = "outcome")
effect <- FeatureImp$new(model, loss = "ce")
plot_train <- plot(effect, features = num_features)

# Test set
model <- Predictor$new(learner, data = dt2[test_set, ], y = "outcome")
effect <- FeatureImp$new(model, loss = "ce")
plot_test <- plot(effect, features = num_features)

# Put together
library("patchwork")
plot_train + plot_test
