# Databricks notebook source
# Create widget for parameter passing into the notebook
dbutils.widgets.text("data_input_path", "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")

# COMMAND ----------

# Read from the widget
data_input_path = dbutils.widgets.get("data_input_path").strip()

# COMMAND ----------

# Log an artifact from the input path
import mlflow
name = 'multistep'
with mlflow.start_run(run_name=name) as run:
  # Log the data
  data_path = "data-csv"
  mlflow.log_artifact(data_input_path, data_path)
  
  run_id = run.info.run_id
  path = data_path
  #artifactURI = mlflow.get_artifact_uri()
  #full_path = dbutils.fs.ls(artifactURI + "/" + data_path)[0].path

# COMMAND ----------

# Report the results back to the parent notebook
#import json

#dbutils.notebook.exit(json.dumps({
#  "status": "OK",
#  "data_input_path": data_input_path,
#  "run_id":run_id,
#  "path":path,
#  #"data_output_path": full_path.replace("dbfs:", "/dbfs")
#}))
print("run_id:",run_id)
print("path:",path)



# COMMAND ----------

# TODO
#Create 3 widgets for parameter passing into the notebook:
#  - n_estimators with a default of 100
# - learning_rate with a default of .1
#  - max_depth with a default of 1 
# Note that only strings can be used for widgets
# Create widget for parameter passing into the notebook
dbutils.widgets.text("run_id", run_id)
dbutils.widgets.text("path", path)
dbutils.widgets.text("n_estimators", "100")
dbutils.widgets.text("learning_rate", "0.1")
dbutils.widgets.text("max_depth", "1")



# COMMAND ----------

import os
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
local_dir = "/tmp/artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
local_path = client.download_artifacts(dbutils.widgets.get("run_id").strip(), dbutils.widgets.get("path").strip(), local_dir)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))

# Read from the widget
artifact_URI = local_path + "/" + os.listdir(local_path)[0]
n_estimators = int(dbutils.widgets.get("n_estimators"))
learning_rate = float(dbutils.widgets.get("learning_rate"))  
max_depth = int(dbutils.widgets.get("max_depth"))   

# COMMAND ----------

# TODO: Train and log the results from a model. Try using Gradient Boosted Trees
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

with mlflow.start_run() as run:
  # Import the data
  df = pd.read_csv('/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv')
  X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
  # Create model, train it, and create predictions
  gb = GradientBoostingRegressor(n_estimators=n_estimators,learning_rate = learning_rate, max_depth = max_depth)
  gb.fit(X_train, y_train)
  predictions = gb.predict(X_test)

  # Log model
  model_path = "gradient-boosting-model"
  mlflow.sklearn.log_model(gb, model_path)
    
  # Log params
  mlflow.log_param("n_estimators", n_estimators)
  mlflow.log_param("learning_rate", learning_rate)
  mlflow.log_param("max_depth", max_depth)

  # Log metrics
  mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
  mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
  mlflow.log_metric("r2", r2_score(y_test, predictions)) 
  
  artifactURI = mlflow.get_artifact_uri()
  model_output_path = "runs:/" + run.info.run_id + "/" + model_path

# COMMAND ----------

# TODO: Train and log the results from a model.  Try using Gradient Boosted Trees
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

# COMMAND ----------

# TODO
# Report the results back to the parent notebook
#import json

#dbutils.notebook.exit(json.dumps({
#  "status": "OK",
#  "model_output_path": model_output_path, #.replace("dbfs:", "/dbfs")
#  "data_path": artifact_URI
#}))

print("model_output_path:", model_output_path)
print("data_path:",artifact_URI)


# COMMAND ----------

# Create widget for parameter passing into the notebook
dbutils.widgets.text("model_path", model_output_path)
dbutils.widgets.text("data_path",artifact_URI)

# COMMAND ----------

# Read from the widget
model_path = dbutils.widgets.get("model_path").strip()
data_path = dbutils.widgets.get("data_path").strip()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import tempfile

with mlflow.start_run() as run:
  # Import the data
  df = pd.read_csv(data_path).drop(["price"], axis=1)
  model = mlflow.sklearn.load_model(model_path)

  predictions = model.predict(df)
  
  temp = tempfile.NamedTemporaryFile(prefix="predictions_", suffix=".csv")
  temp_name = temp.name
  try:
    pd.DataFrame(predictions).to_csv(temp_name)
    mlflow.log_artifact(temp_name, "predictions.csv")
  finally:
    temp.close() # Delete the temp file
    
  artifactURI = mlflow.get_artifact_uri()
  predictions_output_path = artifactURI + "/predictions.csv"
  run_id = run.info.run_id
  data_dir = temp_name
  

# COMMAND ----------

# Report the results back to the parent notebook
import json

dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "run_id": run_id,
  "data_dir": data_dir
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
