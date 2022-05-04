# Databricks notebook source
# MAGIC %md
# MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
# MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
# MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
# MAGIC - **Receipts** - the cost of gas for specific transactions.
# MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
# MAGIC - **Tokens** - Token data including contract address and symbol information.
# MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
# MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
# MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
# MAGIC 
# MAGIC ### Rubric for this module
# MAGIC Answer the quetions listed below.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %fs ls /mnt/dscc202-datasets/misc/G06

# COMMAND ----------

# MAGIC %python
# MAGIC # Grab the global variables
# MAGIC wallet_address,start_date = Utils.create_widgets()
# MAGIC print(wallet_address,start_date)
# MAGIC spark.conf.set('wallet.address',wallet_address)
# MAGIC spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the maximum block number and date of block in the database

# COMMAND ----------

# MAGIC %sql
# MAGIC select number as MAX_BLOCK_NUMBER,
# MAGIC from_unixtime(timestamp, "yyyy/MM/dd") as DATE_OF_MAX_BLOCK_NUMBER
# MAGIC from ethereumetl.blocks
# MAGIC where number in (select max(number)
# MAGIC                  from ethereumetl.blocks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: At what block did the first ERC20 transfer happen?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT number as BLOCKNUMBER_FIRST_ERC20
# MAGIC FROM (SELECT number, timestamp
# MAGIC FROM ethereumetl.blocks 
# MAGIC INNER JOIN g06_db.erc20_token_transfer 
# MAGIC ON ethereumetl.blocks.number = g06_db.erc20_token_transfer.block_number 
# MAGIC WHERE g06_db.erc20_token_transfer.is_erc20 = "True"
# MAGIC ORDER BY timestamp ASC
# MAGIC LIMIT 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(is_erc20) as Number_Of_ERC20
# MAGIC FROM ethereumetl.silver_contracts
# MAGIC WHERE is_erc20 = "True"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT sum(counts) AS DISTINCT_COUNT
# MAGIC FROM (SELECT count(*) AS counts
# MAGIC FROM ethereumetl.silver_contracts
# MAGIC WHERE is_erc20 = "True"
# MAGIC GROUP BY address)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Q: What percentage of transactions are calls to contracts

# COMMAND ----------

# MAGIC %python 
# MAGIC transactions_to_adress_DF = spark.sql('SELECT COUNT (to_address) FROM ethereumetl.transactions INNER JOIN ethereumetl.silver_contracts ON ethereumetl.transactions.to_address = ethereumetl.silver_contracts.address')

# COMMAND ----------

# MAGIC %python
# MAGIC transactions_to_adress_all_DF = spark.sql('SELECT COUNT (to_address) FROM ethereumetl.transactions')

# COMMAND ----------

# MAGIC %python
# MAGIC spark.conf.set("spark.sql.shuffle.partitions",1000)
# MAGIC n1 = transactions_to_adress_DF.first()['count(to_address)']
# MAGIC n2 = transactions_to_adress_all_DF.first()['count(to_address)']

# COMMAND ----------

from pyspark.sql.functions import round 
perc = n1/n2
print(perc*100,'% of transactions are calls to contracts')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What are the top 100 tokens based on transfer count?

# COMMAND ----------

transferDF = spark.sql('select * from ethereumetl.token_transfers')

# COMMAND ----------

from pyspark.sql.functions import col, count, mean,first

transferDF = transferDF.groupBy("token_address").agg(count("transaction_hash").alias("transfer_count_for_each_token_address"))
q5 = transferDF.sort(col("transfer_count_for_each_token_address").desc()).limit(100)

# COMMAND ----------

token_join = spark.sql('select address,symbol,name from ethereumetl.tokens')
new = token_join.join(q5, token_join.address == q5.token_address,"inner").select(q5.token_address, token_join.symbol, q5.transfer_count_for_each_token_address) 
new = new.groupBy("token_address").agg(mean("transfer_count_for_each_token_address").alias("transfer_count_for_each_symbol"), first("symbol").alias("symbol"))
new = new.sort(col("transfer_count_for_each_symbol").desc())

# COMMAND ----------

display(new)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
# MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

# COMMAND ----------

transferDF1 = spark.sql('select token_address, from_address, to_address, transaction_hash from ethereumetl.token_transfers')
ERC20 = spark.sql('select * from g06_db.erc20_contract')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT (to_address) 
# MAGIC FROM ethereumetl.transactions 
# MAGIC INNER JOIN ethereumetl.silver_contracts 
# MAGIC ON ethereumetl.transactions.to_address = ethereumetl.silver_contracts.address
# MAGIC WHERE ethereumetl.silver_contracts.is_erc20 = "True"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM ethereumetl.token_transfers
# MAGIC INNER JOIN ethereumetl.silver_contracts 
# MAGIC ON ethereumetl.token_transfers.token_address = ethereumetl.silver_contracts.address
# MAGIC WHERE ethereumetl.silver_contracts.is_erc20 = "True"
# MAGIC LIMIT 100

# COMMAND ----------

ERC20_TRANSFER = spark.sql('SELECT * FROM ethereumetl.token_transfers INNER JOIN ethereumetl.silver_contracts ON ethereumetl.token_transfers.token_address = ethereumetl.silver_contracts.address WHERE ethereumetl.silver_contracts.is_erc20 = "True"')

# COMMAND ----------

ERC20_TRANSFER = ERC20_TRANSFER.groupBy("to_address").agg(count("transaction_hash").alias("transfer_count"))
ERC20_TRANSFER_new = ERC20_TRANSFER.where(ERC20_TRANSFER.transfer_count ==1)

# COMMAND ----------

ERC20_TRANSFER_new.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT (to_address) 
# MAGIC FROM ethereumetl.transactions 

# COMMAND ----------

q6 = transferDF1.groupBy("to_address").agg(count("transaction_hash").alias("transfer_count_for_each_token_address"))

# COMMAND ----------

q6.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
# MAGIC - hint: find a block with multiple transactions 

# COMMAND ----------

transactionsDF = spark.sql('select hash, block_number, transaction_index, gas, gas_price from ethereumetl.transactions')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What was the highest transaction throughput in transactions per second?
# MAGIC hint: assume 15 second block time

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TBD

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the total Ether volume?
# MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT sum(value)/(1x10^18) 
# MAGIC AS total_Ether_volume
# MAGIC FROM ethereumetl.transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the total gas used in all transactions?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT SUM(gas_used)
# MAGIC AS total_gas_used
# MAGIC FROM ethereumetl.receipts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

# COMMAND ----------

ERC20_TRANSFER = spark.sql('SELECT * FROM ethereumetl.token_transfers INNER JOIN ethereumetl.silver_contracts ON ethereumetl.token_transfers.token_address = ethereumetl.silver_contracts.address WHERE ethereumetl.silver_contracts.is_erc20 = "True"')
ERC20_TRANSFER1 = ERC20_TRANSFER.groupBy(ERC20_TRANSFER.token_address).agg(sum("value").alias("TRANSFER_AMOUNT"))
ERC20_TRANSFER1 = ERC20_TRANSFER1.sort(ERC20_TRANSFER1.TRANSFER_AMOUNT).dsc()
# show the row with the maximum amount of transfers of ERC-20
ERC20_TRANSFER1.limit(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: Token balance for any address on any date?

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TBD

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viz the transaction count over time (network use)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TBD

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viz ERC-20 transfer count over time
# MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TBD

# COMMAND ----------

# MAGIC %python
# MAGIC # Return Success
# MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
