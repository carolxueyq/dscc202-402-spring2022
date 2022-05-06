# Databricks notebook source
# MAGIC %md
# MAGIC ## Token Recommendation
# MAGIC <table border=0>
# MAGIC   <tr><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/rec-application.png'></td>
# MAGIC     <td>Your application should allow a specific wallet address to be entered via a widget in your application notebook.  Each time a new wallet address is entered, a new recommendation of the top tokens for consideration should be made. <br> **Bonus** (3 points): include links to the Token contract on the blockchain or etherscan.io for further investigation.</td></tr>
# MAGIC   </table>

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address = Utils.create_widgets()
print(wallet_address)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your code starts here...

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql import functions as F6

# Load the data from the lake
wallet_count_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/wctables")
tokens_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/tokentables")

# COMMAND ----------

from pyspark.sql import Window
2
from pyspark.sql.functions import dense_rank
3
wallet_count_df = wallet_count_df.withColumn("new_tokenId",dense_rank().over(Window.orderBy("token_address")))
4
 
5
 
6
 
7
wallet_count_df = wallet_count_df.withColumn("new_walletId",dense_rank().over(Window.orderBy("wallet_address")))
8
 
9
wallet_count_df = wallet_count_df.withColumnRenamed("token_address","tokenId")
10
wallet_count_df = wallet_count_df.withColumnRenamed("wallet_address","walletId")

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from delta.tables import *
import random

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

modelName = "111"
def mlflow_als(rank,maxIter,regParam):

    with mlflow.start_run(run_name = modelName+"-run") as run:
        seed = 42
        (split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)
        training_df = split_60_df.cache()
        validation_df = split_a_20_df.cache()
        test_df = split_b_20_df.cache()
        input_schema = Schema([ColSpec("integer", "new_tokenId"),ColSpec("integer", "new_walletId")])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
        # Create model
        # Initialize our ALS learner
        als = ALS(rank=rank, maxIter=maxIter, regParam=regParam,seed=42)
        als.setItemCol("new_tokenId")\
           .setRatingCol("buy_count")\
           .setUserCol("new_walletId")\
           .setColdStartStrategy("drop")
        reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")

        alsModel = als.fit(training_df)
        validation_metric = reg_eval.evaluate(alsModel.transform(validation_df))
    
        mlflow.log_metric('valid_' + reg_eval.getMetricName(), validation_metric) 
    
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
    
        # Log model
        mlflow.spark.log_model(spark_model=alsModel, signature = signature,artifact_path='als', registered_model_name=modelName)
    return alsModel, validation_metric

# COMMAND ----------

mlflow_als(rank = 5,maxIter = 5, regParam = 0.6)

# COMMAND ----------

client = MlflowClient()
model_versions = []
    
for mv in client.search_model_versions(f"name='{modelName}'"):
    model_versions.append(dict(mv)['version'])
    if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        # Archive the currently staged model
        client.transition_model_version_stage(
            name= modelName,
            version=dict(mv)['version'],
            stage="Archived"
        )

# COMMAND ----------

client.transition_model_version_stage(name=modelName,version=model_versions[0],stage="Staging")

# COMMAND ----------

wallet_address = Utils.create_widgets()
walletId = wallet_count_df.where(col("walletId") == wallet_address).select(new_walletId)

def recommend(walletId: int)->(DataFrame,DataFrame):
    bought_token = wallet_count_df.filter(wallet_count_df.walletId == walletId).join(tokens_df, 'address').select('new_tokenId', 'symbol', 'name','buy_count')
    unbought_token = wallet_count_df.filter(~ wallet_count_df['new_tokenId'].isin([token['new_tokenId'] for token in bought_token.collect()])).select('new_tokenId').withColumn('new_walletId', F.lit(walletId)).distinct()
 
    model = mlflow.spark.load_model('models:/'+modelName+'/Staging')
    predicted_buy_counts = model.transform(unbought_token)
 
    return (bought_token.select('symbol','name','buy_count').orderBy('buy_count', ascending = False), predicted_buy_counts.join(wallet_count_df, 'new_tokenId') \
                .join(tokens_df, 'address') \
                .select('symbol', 'name', 'prediction') \
                .distinct() \
                .orderBy('prediction', ascending = False)
                .limit(5))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
