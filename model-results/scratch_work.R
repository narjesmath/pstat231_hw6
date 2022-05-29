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
library(kableExtra)
tidymodels_prefer()


## Tree-Based Models

### Exercise 1


abalone <- read_csv(here("data","abalone.csv"))
pokemon_raw <- read_csv(here("data","Pokemon.csv"))

pokemon <- pokemon_raw %>% 
  clean_names() %>% 
  filter(type_1 %in% c("Bug", "Fire", "Grass", "Normal", "Water", "Psychic")) %>% 
  mutate(type_1 = factor(type_1), legendary= factor(legendary),generation = factor(generation))


set.seed(3435)
pok_rare_split <- pokemon %>% 
  initial_split(strata = type_1, prop = 0.7)
pok_rare_train <- training(pok_rare_split)
pok_rare_test <- testing(pok_rare_split)


pok_rare_fold <- vfold_cv(pok_rare_train, v = 5, strata = type_1)




model_recipe <- recipe(type_1 ~ legendary + generation + sp_atk + attack + speed + defense + hp + sp_def, data = pok_rare_train) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())



### Exercise 2

pok_rare_train %>% 
  select(where(is.numeric)) %>% 
  select(-number, - total) %>% 
  cor() %>% 
  corrplot(type = "lower", method = "number", diag = FALSE)

### Exercise 3

tree_spec <- decision_tree() %>%
  set_engine("rpart")

class_tree_spec <- tree_spec %>%
  set_mode("classification")



class_tree_wf <- workflow() %>%
  add_recipe(model_recipe) %>%  
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) 

param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

tune_res <- tune_grid(
  class_tree_wf, 
  resamples = pok_rare_fold, 
  grid = param_grid, 
  metrics = metric_set(roc_auc)
)


### Exercise 4

tree_metrics <- collect_metrics(tune_res) %>% 
  arrange(desc((mean)))

best_complexity <- select_best(tune_res)



### Exercise 5


class_tree_final <- finalize_workflow(class_tree_wf, best_complexity)

class_tree_final_fit <- fit(class_tree_final, data = pok_rare_train)

class_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot()



### Exercise 5

class_rf_model = rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification") 


class_rf_wflw = workflow() %>% 
  add_recipe(model_recipe) %>% 
  add_model(class_rf_model) 



rf_grid <- grid_regular(
  mtry(range = c(1, 8)), #set mtry to 1-8
  trees(range = c(100,200)), #set trees 100 to 200
  min_n(range = c(2, 20)), # set min_n between 2 and 20
  levels = 8) #what's a good level?

rf_tune <- tune_grid( 
  class_rf_wflw,
  resamples = pok_rare_fold, 
  grid = rf_grid, 
  metrics = metric_set(roc_auc) 
)


# write to rds
write_rds(rf_tune, here("pstat231_hw6","model-results"),"decision_tree_results.rds", compress ="none")

autoplot(tune_res)


# decision_tree_results <- read_rds("decision_tree_results.rds")
# autoplot(decision_tree_results)


autoplot(rf_tune) 




### Exercise 7

 

rf_tree_metrics <- collect_metrics(rf_tune) %>% 
  arrange(desc((mean)))

best_complexity <- select_best(rf_tune)



### Exercise 8

rf_final <- finalize_workflow(class_rf_wflw, select_best(rf_tune)) 

rf_fit <- fit(rf_final, pok_rare_train)  

vip_rf <- rf_fit %>% 
  extract_fit_engine() %>%
  vip()

vip_rf


### Exercise 9


boost_model = boost_tree(trees = tune()) %>% #set up boosted tree model
  set_engine("xgboost") %>% # xgboost engine calculates lambda #high lambda is good 
  set_mode("classification") 


boost_wflw = workflow() %>%
  add_recipe(model_recipe) %>% 
  add_model(boost_model) 

boost_grid <- grid_regular(
  trees(range = c(10,2000)), 
  levels = 10) 

boost_tune_res <- tune_grid( 
  boost_wflw, 
  resamples = pok_rare_fold, 
  grid = boost_grid, 
  metrics = metric_set(roc_auc) 
)


write_rds(boost_tune_res, here("pstat231_hw6","boost_results.rds"))

autoplot(boost_tune_res) 



boost_metrics <- collect_metrics(boost_tune_res) %>% 
  arrange(desc((mean)))

best_complexity <- select_best(boost_tune_res)



### Exercise 10


models = full_join(tree_metrics, rf_tree_metrics)
joined_models = full_join(models, boost_metrics)
joined_models 
 


rf_test_fit = augment(rf_fit, new_data = pok_rare_test) 
roc_auc(rf_test_fit, truth = type_1, .pred_Bug, .pred_Fire, .pred_Grass, .pred_Normal, .pred_Water, .pred_Psychic) 



autoplot(roc_curve(rf_test_fit, truth = type_1, .pred_Bug, .pred_Fire, .pred_Grass, .pred_Normal, .pred_Water, .pred_Psychic)) 



conf_mat(rf_test_fit, truth = type_1, estimate = .pred_class) %>% 
  autoplot(type = "heatmap") + 
  theme(axis.text.x = element_text(angle = 90, hjust=1)) 




## For 231 Students

### Exercise 11

# Using the `abalone.txt` data from previous assignments, fit and tune a random forest model to predict `age`. Use stratified cross-validation and select ranges for `mtry`, `min_n`, and `trees`. Present your results. What was the model's RMSE on your testing set?
 
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
  trees(range = c(100,200)), 
  min_n(range = c(2, 40)), 
  levels = 5) 



abalone_rf_tune <- tune_grid( 
  abalone_rf_wflw, 
  resamples = folds, 
  grid = abalone_rf_grid,  
  metrics = metric_set(rmse) 
)



write_rds(abalone_rf_tune, here("pstat231_hw6","abalone_results.rds"), compress ="none")



autoplot(abalone_rf_tune)
# autoplot(abalone_tree_results) 



select_best(abalone_rf_tune)




collect_metrics(abalone_rf_tune) %>% 
  arrange(-mean) 




abalone_rf_final <- finalize_workflow(abalone_rf_wflw, select_best(abalone_rf_tune)) 
abalone_rf_fit <-  fit(abalone_rf_final, abalone_train) 



augment(abalone_rf_fit, new_data = abalone_test) %>% 
  rmse(truth = age, estimate = .pred) 




