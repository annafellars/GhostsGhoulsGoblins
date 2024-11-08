library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)
library(keras)

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

nn_recipe <- recipe(type~., data = train_data) |>
  step_mutate_at(color, fn = factor) |>
  step_dummy(color) |>
  step_range(all_numeric_predictors(), min = 0, max = 1)

prepped_recipe <- prep(nn_recipe)
show <- bake(prepped_recipe, new_data = train_data)

nn_model <- mlp(hidden_units = tune(),
                epochs = 100) |>
  set_engine("keras") |>
  set_mode("classification")

nn_wf <- workflow() |>
  add_recipe(nn_recipe) |>
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1,15)),
                            levels = 5)

nn_folds <- vfold_cv(train_data, v = 5, repeats = 1)

nn_results <- tune_grid(
  nn_wf,
  resamples = nn_folds,
  grid = nn_tuneGrid,
  metrics = metric_set(roc_auc)
)

tuned_nn |>
  collect_metrics() |>
  filter(.metric=="accuracy") |>
  ggplot(aes(x=hidden_units, y = mean)) + geom_line()



##Run the CV
nn_CV_results <- nn_wf |>
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
nn_best_tune <- nn_CV_results |>
  select_best(metric = "roc_auc")

##finalize the workflow and fit it
nn_final <- nn_wf |>
  finalize_workflow(nn_best_tune) |>
  fit(data = train_data)

##predict
nn_preds <- nn_final |>
  predict(new_data = test_data, type = "class")

kaggle <- nn_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGNN.csv", delim=",")
