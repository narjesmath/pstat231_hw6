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

abalone_new <- abalone %>% 
  mutate(type = as.factor(type))%>% 
  mutate(age = 1.5 + rings) 


set.seed(3435)
abalone_split <- abalone_new%>% 
  initial_split(strata = age, prop = 0.7)
abalone_train <- training(abalone_split)
aalone_test <- testing(abalone_split)

folds = vfold_cv(abalone_train, v = 5, strata = age) 


abalone_recipe <-
  recipe(age ~ type + longest_shell + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight, data = abalone_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ type:shucked_weight + longest_shell:diameter + shucked_weight:shell_weight) %>% 
  step_normalize(all_numeric_predictors()) #center &scale



abalone_rf <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression") 


abalone_rf_wflw = workflow() %>% 
  add_recipe(abalone_recipe) %>% 
  add_model(abalone_rf) 


abalone_rf_grid <- grid_regular(
  mtry(range = c(1, 8)), 
  trees(range = c(10,2000)), 
  min_n(range = c(2, 40)), 
  levels = 8) 

abalone_rf_tune <- tune_grid( 
  abalone_rf_wflw, 
  resamples = folds, 
  grid = abalone_rf_grid,  
  metrics = metric_set(rmse) 
)

write_rds(abalone_rf_tune,here("homework-6.Rmd","abalone_tree_results"))