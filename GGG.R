library(vroom)
library(tidymodels)
library(tidyverse)

missing_data <- vroom("./trainWithMissingValues.csv")
missing_data <- missing_data |>
  mutate(color = as.factor(color))

comp_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

missing_recipe <- recipe(type~., data = missing_data) |>
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors = 3) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors = 3) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors = 3)

prepped_recipe <- prep(missing_recipe)
imputed_data <- bake(prepped_recipe, new_data = missing_data)


rmse_vec(comp_data[is.na(missing_data)],
         imputed_data[is.na(missing_data)])
#0.1473116