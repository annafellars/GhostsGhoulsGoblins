library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)

missing_data <- vroom("./trainWithMissingValues.csv")
missing_data <- missing_data |>
  mutate(color = as.factor(color))

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

missing_recipe <- recipe(type~., data = missing_data) |>
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors = 3) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors = 3) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors = 3)

prepped_recipe <- prep(missing_recipe)
imputed_data <- bake(prepped_recipe, new_data = missing_data)


rmse_vec(train_data[is.na(missing_data)],
         imputed_data[is.na(missing_data)])

#0.1473116

my_recipe <- recipe(type~., data = train_data) |>
  step_mutate_at(color, fn = factor) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |>
  step_normalize(all_nominal_predictors())

prepped_recipe <- prep(my_recipe)
show <- bake(prepped_recipe, new_data = train_data)


######################################################################################
#KNN
knn_model <- nearest_neighbor(neighbors = 20) |>
  set_mode('classification') |>
  set_engine('kknn')
 
#set workflow
knn_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(knn_model) |>
  fit(data = train_data)
 
#Predict
knn_preds = predict(knn_wf, 
  new_data = test_data, type = "class")

## Format predictions for Kaggle
kaggle <- knn_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGKNN.csv", delim=",")

#####################################################################################
#NB
nb_mod <- naive_Bayes(Laplace= tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)

## Set up grid and tuning values
nb_tuning_params <- grid_regular(Laplace(),
                                 smoothness(),
                                 levels = 5)

##Split data for CV
nb_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
nb_CV_results <- nb_wf |>
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
nb_best_tune <- nb_CV_results |>
  select_best(metric = "roc_auc")

##finalize the workflow and fit it
nb_final <- nb_wf |>
  finalize_workflow(nb_best_tune) |>
  fit(data = train_data)

##predict
nb_preds <- nb_final |>
  predict(new_data = test_data, type = "class")

kaggle <- nb_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGNB.csv", delim=",")
