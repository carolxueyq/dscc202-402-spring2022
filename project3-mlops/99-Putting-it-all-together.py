# Databricks notebook source
# MAGIC %md
# MAGIC # Putting it all together: Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC Create a workflow that includes pre-processing logic, the optimal ML algorithm and hyperparameters, and post-processing logic.
# MAGIC 
# MAGIC ## Instructions
# MAGIC 
# MAGIC In this course, we've primarily used Random Forest in `sklearn` to model the Airbnb dataset.  In this exercise, perform the following tasks:
# MAGIC <br><br>
# MAGIC 0. Create custom pre-processing logic to featurize the data
# MAGIC 0. Try a number of different algorithms and hyperparameters.  Choose the most performant solution
# MAGIC 0. Create related post-processing logic
# MAGIC 0. Package the results and execute it as its own run
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

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing
# MAGIC 
# MAGIC Take a look at the dataset and notice that there are plenty of strings and `NaN` values present. Our end goal is to train a sklearn regression model to predict the price of an airbnb listing.
# MAGIC 
# MAGIC 
# MAGIC Before we can start training, we need to pre-process our data to be compatible with sklearn models by making all features purely numerical. 

# COMMAND ----------

import pandas as pd

airbnbDF = spark.read.parquet("/mnt/training/airbnb/sf-listings/sf-listings-correct-types.parquet").toPandas()

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we will walk you through the most basic pre-processing step necessary. Feel free to add additional steps afterwards to improve your model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, convert the `price` from a string to a float since the regression model will be predicting numerical values.

# COMMAND ----------

# TODO
airbnbDF['price'] = airbnbDF['price'].apply(lambda x: x.replace("$","").replace(",","")).astype(float)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

airbnbDF.isna().sum()

# COMMAND ----------

# TODO
airbnbDF.info()

# COMMAND ----------

airbnbDF.columns.tolist()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
 
plt.figure(figsize=(18,15))
sns.heatmap(airbnbDF.corr(),cmap = "coolwarm", annot=True,fmt=".2%")
plt.show()

# COMMAND ----------

# Based on the above heatmap, I decided to remove the numerical variables that are not that highly correlated(under 10%) to the price, such as longtitude. For categorical variables, remove zipcode since there are 30 null values and create dummies for other variables. 

airbnbcopy = airbnbDF

columns = ['host_is_superhost',
 'cancellation_policy',
 'instant_bookable',
 'latitude',
 'property_type',
 'room_type',
 'accommodates',
 'bathrooms',
 'bedrooms',
 'beds',
 'bed_type',
 'number_of_reviews',
 'review_scores_rating',
 'price']

airbnbcopy = airbnbcopy.loc[:,airbnbcopy.columns.isin(columns)]

# COMMAND ----------

airbnbcopy.shape

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

# TODO
# Encode dummy variables
dummies1 = pd.get_dummies(airbnbcopy['host_is_superhost']).rename(columns=lambda x: 'host_is_superhost_' + str(x))
dummies2 = pd.get_dummies(airbnbcopy['cancellation_policy']).rename(columns=lambda x: 'cancellation_policy_' + str(x))
dummies3 = pd.get_dummies(airbnbcopy['instant_bookable']).rename(columns=lambda x: 'instant_bookable_' + str(x))
dummies4 = pd.get_dummies(airbnbcopy['property_type']).rename(columns=lambda x: 'property_type_' + str(x))
dummies5 = pd.get_dummies(airbnbcopy['room_type']).rename(columns=lambda x: 'room_type_' + str(x))
dummies6 = pd.get_dummies(airbnbcopy['bed_type']).rename(columns=lambda x: 'bed_type_' + str(x))

# COMMAND ----------

df = pd.concat([airbnbcopy, dummies1,dummies2,dummies3,dummies4,dummies5,dummies6], axis=1)

# COMMAND ----------

# Plot the heatmap again for checking the correlation among dummies and price
plt.figure(figsize=(18,15))
sns.heatmap(df.corr(),cmap = "coolwarm", annot=False)
plt.show()

# COMMAND ----------

#filter independent variables again!! Only keep the dummies for room type

c = ['accommodates',
 'bathrooms',
 'bedrooms',
 'beds',
 'number_of_reviews',
 'review_scores_rating',
 'price',
 'room_type_Entire home/apt',
 'room_type_Private room',
 'room_type_Shared room']
df = df.loc[:,df.columns.isin(c)]

# COMMAND ----------

df=df.fillna(df.median())

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X = df.drop("price", axis=1)
y = df['price']

# min_max_scaler = MinMaxScaler()
# X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC After cleaning our data, we can start creating our model!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Firstly, if there are still `NaN`'s in your data, you may want to impute these values instead of dropping those entries entirely. Make sure that any further processing/imputing steps after the train test split is part of a model/pipeline that can be saved.
# MAGIC 
# MAGIC In the following cell, create and fit a single sklearn model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear Regression

# COMMAND ----------

# TODO
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# TODO
 
## RANDOM FOREST
 
# Initializing my pipeline
estimators = [('model', RandomForestRegressor())]
 
pipe = Pipeline(estimators)
 
# These are the hyperparamaters and models I want to tune
param_grid = {'model': [RandomForestRegressor()], 
               'model__n_estimators': [*range(50,200,50)], 
              'model__max_depth': [50,150,300,None]}
 
# 5 fold cross validation
grid = GridSearchCV(pipe, param_grid, cv = 3, verbose = 3)
fitted_grid = grid.fit(X_train, y_train)

# COMMAND ----------

results = pd.DataFrame(fitted_grid.cv_results_).sort_values('mean_test_score', ascending = False)
results

# COMMAND ----------

model = fitted_grid.best_estimator_[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosting

# COMMAND ----------

from sklearn import ensemble

params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 200,
    "learning_rate": 0.02
}
GB = ensemble.GradientBoostingRegressor(**params)
GB.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO
# MSE for linear regression
import numpy as np
from sklearn import metrics
y_pred = reg.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)

# COMMAND ----------

# TODO
# MSE for RF
y_pred1 = model.predict(X_test)
metrics.mean_squared_error(y_test, y_pred1)

# COMMAND ----------

# TODO
# MSE for GB
y_pred2 = GB.predict(X_test)
metrics.mean_squared_error(y_test, y_pred2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# RF
def mlflow_rf(df, n_estimators, max_depth):
    
 with mlflow.start_run(run_name="Random Forest") as run:
   X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), test_size=0.2, random_state=22)
   rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
   rf.fit(X_train, y_train)
   predictions = rf.predict(X_test)

   # Log model
   mlflow.sklearn.log_model(rf, "random-forest-model")

   # Log params
   mlflow.log_param("n_estimators", n_estimators)
   mlflow.log_param("max_depth", max_depth)

   # Log metrics
   mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
   mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
   runID = run.info.run_uuid
   experimentID = run.info.experiment_id
   print(f"RF model with run_id `{runID}` and experiment_id `{experimentID}`, having mse of '{mean_squared_error(y_test, predictions)}'")

# LR
def mlflow_lr(df):
    
 with mlflow.start_run(run_name="Linear Regression") as run:
   X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), test_size=0.2, random_state=22)
   lr = LinearRegression()
   lr.fit(X_train, y_train)
   predictions = lr.predict(X_test)

   # Log model
   mlflow.sklearn.log_model(lr, "linear regression")

   # Log metrics
   mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
   mlflow.log_metric("mae", mean_absolute_error(y_test, predictions)) 

   runID = run.info.run_uuid
   experimentID = run.info.experiment_id
   print(f"LR model with run_id `{runID}` and experiment_id `{experimentID}`, having mse of '{mean_squared_error(y_test, predictions)}'")
   return(runID)

# GB
def mlflow_gb(df,n_estimators,max_depth,split,rate):
    
 with mlflow.start_run(run_name="Gradient Boosting") as run:
    
   X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), test_size=0.2, random_state=22)
   params = {
   "n_estimators": n_estimators,
   "max_depth": max_depth,
   "min_samples_split": split,
   "learning_rate": rate}
    
   gb = ensemble.GradientBoostingRegressor(**params)
   gb.fit(X_train, y_train)
   predictions = gb.predict(X_test)
    
   # Log model
   mlflow.sklearn.log_model(gb, "Gradient Boosting")

   # Log params
   mlflow.log_param("n_estimators", n_estimators)
   mlflow.log_param("max_depth", max_depth)
   mlflow.log_param("learning_rate", rate)

   # Log metrics
   mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
   mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))

   runID = run.info.run_uuid
   experimentID = run.info.experiment_id
   print(f"GB model with run_id `{runID}` and experiment_id `{experimentID}`, having mse of '{mean_squared_error(y_test, predictions)}'")
   return(runID)

import tensorflow as tf
tf.random.set_seed(22)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def mlflow_nn(df,unit1,input_dim,unit2):
    
 with mlflow.start_run(run_name="Gradient Boosting") as run:
    
   X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), test_size=0.2, random_state=22)
   nn = Sequential([
   Dense(unit1, input_dim=input_dim, activation='relu'),
   Dense(unit2, activation='relu'),
   Dense(1, activation='linear')])
    
   nn.compile(optimizer="adam", loss="mse")
   nn.fit(X_train, y_train, epochs=100, verbose=0)
   predictions = nn.predict(X_test)
    
   # Log model
   mlflow.sklearn.log_model(nn, "Neural Network")

   # Log metrics
   mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
   mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))

   runID = run.info.run_uuid
   experimentID = run.info.experiment_id
   print(f"NN model with run_id `{runID}` and experiment_id `{experimentID}`, having mse of '{mean_squared_error(y_test, predictions)}'")
   return(runID)



# COMMAND ----------

# rf1
mlflow_rf(df, 100, 50)
#rf2
mlflow_rf(df, 50, 50)
#rf3
mlflow_rf(df, 150, 100)

#lr
mlflow_lr(df)

#gb1
mlflow_gb(df,500,5,200,0.02)
#gb2
mlflow_gb(df,400,5,100,0.01)
#gb3
mlflow_gb(df,300,5,200,0.01)
#gb4
mlflow_gb(df,500,3,200,0.01)
#gb5
mlflow_gb(df,500,3,200,0.02)

# COMMAND ----------

#gb6
mlflow_gb(df,500,3,100,0.02)
#gb7
mlflow_gb(df,500,3,100,0.01)

# COMMAND ----------

# GB model with run_id `a5aad5beef5441ff8c74cd7fce67acea` and experiment_id `3805545008156378`, having mse of '22540.361541607865'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Change and re-run the above 3 code cells to log different models and/or models with different hyperparameters until you are satisfied with the performance of at least 1 of them.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Look through the MLflow UI for the best model. Copy its `URI` so you can load it as a `pyfunc` model.

# COMMAND ----------

# TODO
import mlflow.pyfunc

gb_pyfunc_model = mlflow.pyfunc.load_model(model_uri='runs:/a5aad5beef5441ff8c74cd7fce67acea/Gradient Boosting')
type(gb_pyfunc_model)

# COMMAND ----------

gb_pyfunc_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing
# MAGIC 
# MAGIC Our model currently gives us the predicted price per night for each Airbnb listing. Now we would like our model to tell us what the price per person would be for each listing, assuming the number of renters is equal to the `accommodates` value. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fill in the following model class to add in a post-processing step which will get us from total price per night to **price per person per night**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://www.mlflow.org/docs/latest/models.html#id13" target="_blank">the MLFlow docs for help.</a>

# COMMAND ----------

# TODO

class Airbnb_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), test_size=0.2, random_state=22)

# COMMAND ----------

# TODO

from mlflow.exceptions import MlflowException
final_model_path =  f"{working_path}/final-model-1"
gb_model = Airbnb_Model(model = gb_pyfunc_model)

dbutils.fs.rm(final_model_path, True) # Allows you to rerun the code multiple times

mlflow.pyfunc.save_model(path=final_model_path.replace("dbfs:", "/dbfs"), python_model=gb_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# TODO
loaded_model = mlflow.pyfunc.load_model(final_model_path)
import pandas as pd

model_output = loaded_model.predict(X_test)

model_output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packaging your Model
# MAGIC 
# MAGIC Now we would like to package our completed model! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First save your testing data at `test_data_path` so we can test the packaged model.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When using `.to_csv` make sure to set `index=False` so you don't end up with an extra index column in your saved dataframe.

# COMMAND ----------

# TODO
test_data = pd.DataFrame(X_test)
test_data['price'] = y_test


# save the testing data 
test_data_path = f"{working_path}/test_data.csv"
test_data.to_csv(test_data_path, index = False)
 
prediction_path = f"{working_path}/predictions.csv"
y_pred = pd.Series(model_output)
y_pred.to_csv(prediction_path, index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC First we will determine what the project script should do. Fill out the `model_predict` function to load out the trained model you just saved (at `final_model_path`) and make price per person predictions on the data at `test_data_path`. Then those predictions should be saved under `prediction_path` for the user to access later.
# MAGIC 
# MAGIC Run the cell to check that your function is behaving correctly and that you have predictions saved at `demo_prediction_path`.

# COMMAND ----------

def model_predict(final_model_path, test_data_path, prediction_path):
    model = mlflow.pyfunc.load_model(final_model_path)
    X_test = pd.read_csv(test_data_path).iloc[:,0:9]
    prediction = model.predict(X_test)
    prediction = pd.Series(prediction)
    return prediction

# COMMAND ----------

model_predict(final_model_path, test_data_path, prediction_path)

# COMMAND ----------

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)
def model_predict(final_model_path, test_data_path, prediction_path):
    model = mlflow.pyfunc.load_model(final_model_path)
    X_test = pd.read_csv(test_data_path).iloc[:,0:9]
    prediction = model.predict(X_test)
    prediction = pd.Series(prediction)
    prediction.to_csv(prediction_path, index = False)
    

# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"

from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a MLproject file and put it under our `workingDir`. Complete the parameters and command of the file.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Capstone-Project

conda_env: conda.yaml

entry_points:
  main:
    parameters: 
      final_model_path: {type: str, default: "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv"}
      test_data_path: {type: str, default: "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv"}
      prediction_path: {type: str, default: "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv"}
    command:  "python predict.py --final_model_path {final_model_path} --test_data_path {test_data_path} --prediction_path {prediction_path}"
'''.strip(), overwrite=True)

# COMMAND ----------

print(prediction_path)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a `conda.yaml` file to list the dependencies needed to run our script.
# MAGIC 
# MAGIC For simplicity, we will ensure we use the same version as we are running in this notebook.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn, sys

version = sys.version_info # Handles possibly conflicting Python versions

file_contents = f"""
name: Capstone
channels:
  - defaults
dependencies:
  - python={version.major}.{version.minor}.{version.micro}
  - cloudpickle={cloudpickle.__version__}
  - numpy={numpy.__version__}
  - pandas={pandas.__version__}
  - scikit-learn={sklearn.__version__}
  - pip:
    - mlflow=={mlflow.__version__}
""".strip()

dbutils.fs.put(f"{workingDir}/conda.yaml", file_contents, overwrite=True)

print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will put the **`predict.py`** script into our project package.
# MAGIC 
# MAGIC Complete the **`.py`** file by copying and placing the **`model_predict`** function you defined above.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/predict.py", 
'''
import click
import mlflow.pyfunc
import pandas as pd

def model_predict(final_model_path, test_data_path, prediction_path):
   model = mlflow.pyfunc.load_model(final_model_path)
   X_test = pd.read_csv(test_data_path).iloc[:,0:9]
   prediction = model.predict(X_test)
   prediction = pd.Series(prediction)
   prediction.to_csv(prediction_path, index = False)
    
if __name__ == "__main__":
   model_predict()

'''.strip(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check all the files we've created are in the `workingDir` folder. You should have at least the following 3 files:
# MAGIC * `MLproject`
# MAGIC * `conda.yaml`
# MAGIC * `predict.py`

# COMMAND ----------

display(dbutils.fs.ls(workingDir))

# COMMAND ----------

# MAGIC %md
# MAGIC Under **`workingDir`** is your completely packaged project.
# MAGIC 
# MAGIC Run the project to use the model saved at **`final_model_path`** to predict the price per person of each Airbnb listing in **`test_data_path`** and save those predictions under **`second_prediction_path`** (defined below).

# COMMAND ----------

final_model_path

# COMMAND ----------

# TODO
df = df
second_prediction_path = f"{working_path}/predictions-2.csv"
final_model_path = f"{working_path}/final-model-1"
test_data_path = f"{working_path}/test_data.csv"

mlflow.projects.run(working_path,
  parameters={"final_model_path": final_model_path,
              "test_data_path": test_data_path,
              "prediction_path": second_prediction_path})

# COMMAND ----------

def model_predict(final_model_path, test_data_path, prediction_path):
   model = mlflow.pyfunc.load_model(final_model_path)
   X_test = pd.read_csv(test_data_path).iloc[:,0:9]
   prediction = model.predict(X_test)
   prediction = pd.Series(prediction)
   prediction.to_csv(prediction_path, index = False)

second_prediction_path = f"{working_path}/predictions-2.csv"
final_model_path = f"{working_path}/final-model-1/"
test_data_path = f"{working_path}/test_data.csv"

model_predict(final_model_path, test_data_path, second_prediction_path)


# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
