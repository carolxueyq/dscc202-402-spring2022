# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Grid Search with MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Import the housing data
# MAGIC  - Perform grid search using scikit-learn
# MAGIC  - Log the best model on MLflow
# MAGIC  - Load the saved model
# MAGIC  
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Import
# MAGIC 
# MAGIC Load in same Airbnb data and create train/test split.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Grid Search using scikit-learn
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values is the most effective. Fill in the code below to perform <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> grid search using `sklearn`</a> over the 2 hyperparameters we looked at in the 02 notebook, `n_estimators` and `max_depth`.

# COMMAND ----------

# TODO
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# dictionary containing hyperparameter names and list of values we want to try
parameters = {'n_estimators': [*range(50,300,50)] , 
              'max_depth': [1,3,5,10,15,20]}

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Best Model on MLflow
# MAGIC 
# MAGIC Log the best model as `grid-random-forest-model`, its parameters, and its MSE metric under a run with name `RF-Grid-Search` in our new MLflow experiment.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

with mlflow.start_run(run_name= "RF GridSearch") as run:
    # Create predictions of X_test using best model
    params = {
    "n_estimators": 100,
    "max_depth": 20,
    "random_state": 42}
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model with name
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    mlflow.log_params(params)

    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"R-Squared: {r2}")
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# MAGIC %md
# MAGIC Check on the MLflow UI that the run `RF-Grid-Search` is logged has the best parameter values found by grid search.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Load the Saved Model
# MAGIC 
# MAGIC Load the trained and tuned model we just saved. Check that the hyperparameters of this model matches that of the best model we found earlier.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Use the `artifactURI` variable declared above.

# COMMAND ----------

# TODO
artifactURI = 'runs:/'+"b1e21c91e1184f2abb32b6ec6d3b8de8"+"/random-forest-model"

model = mlflow.sklearn.load_model(artifactURI)

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, continue to grid search over a wider number of parameters and automatically save the best performing parameters back to `mlflow`.

# COMMAND ----------

# TODO
parameters = {'n_estimators': [*range(10,200,10)] , 
              'max_depth': [5,10,15,20,25,30],
              'random_state': [42]}

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))

# COMMAND ----------

best_rf

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

with mlflow.start_run(run_name= "RF GridSearch 2") as run:
    # Create predictions of X_test using best model
    rf = best_rf
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model with name
    mlflow.sklearn.log_model(rf, "random-forest-model-2")

    # Log params
    params = {
    "n_estimators": 70,
    "max_depth": 25,
    "random_state": 42}
    mlflow.log_params(params)

    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"R-Squared: {r2}")
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, use the `MlflowClient` to interact programatically with your run.

# COMMAND ----------

# TODO
from  mlflow.tracking import MlflowClient

client = MlflowClient()


# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-models")

    # Log params
    mlflow.log_params(params)

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(predictions, y_test, lowess=True)
    plt.xlabel("Predicted values for Price ($)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    return run.info.run_uuid

# COMMAND ----------

#best
params = {
  "n_estimators": 25,
  "max_depth": 70,
  "random_state": 42
}

log_rf(experimentID, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

client.list_experiments()

# COMMAND ----------

client.search_runs(experimentID)

# COMMAND ----------

runs = client.search_runs(experimentID, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

artifactURI = 'runs:/'+runs[0].info.run_id+"/random-forest-models"

# COMMAND ----------

import mlflow.sklearn
2
 
3
model = mlflow.sklearn.load_model(artifactURI)
4
model.feature_importances_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See the solutions folder for an example solution to this lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Packaging ML Projects]($../03-Packaging-ML-Projects ).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
