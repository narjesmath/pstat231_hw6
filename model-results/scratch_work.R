library(here)
library(janitor)
library(corrplot) #to create a correlation matrix
library(tidyverse)
library(tidymodels)
library(glmnet)
library(rpart.plot)
library(vip)
library(randomForest)
library(ranger) #for random forest
library(xgboost)


class_rf_model = rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification") 


class_rf_wflw = workflow() %>% 
  add_recipe(model_recipe) %>% 
  add_model(class_rf_model) 

rf_grid <- grid_regular(
  mtry(range = c(1, 8)), #set mtry to what?
  trees(range = c(100,200)), #set trees to what range? 100 to 200
  min_n(range = c(2, 20)), # set min_n between 2 and 20
  levels = 10) #what's a good level?10

rf_tune <- tune_grid( 
  class_rf_wflw,
  resamples = pok_rare_fold, #use cv folds
  grid = rf_grid, #add parameter grid from -3 to -1?
  metrics = metric_set(roc_auc) 
)

# #autoplot of tuned data
autoplot(rf_tune) 

# write to rda
write_rds(rf_tune,"homework-6.Rmd","decision_tree_results")




