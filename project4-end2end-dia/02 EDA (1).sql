-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
-- MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
-- MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
-- MAGIC - **Receipts** - the cost of gas for specific transactions.
-- MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
-- MAGIC - **Tokens** - Token data including contract address and symbol information.
-- MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
-- MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
-- MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
-- MAGIC 
-- MAGIC ### Rubric for this module
-- MAGIC Answer the quetions listed below.

-- COMMAND ----------

-- MAGIC %run ./includes/utilities

-- COMMAND ----------

-- MAGIC %run ./includes/configuration

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Grab the global variables
-- MAGIC wallet_address,start_date = Utils.create_widgets()
-- MAGIC print(wallet_address,start_date)
-- MAGIC spark.conf.set('wallet.address',wallet_address)
-- MAGIC spark.conf.set('start.date',start_date)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the maximum block number and date of block in the database

-- COMMAND ----------

-- TBD
-- maximum block number 
select max(number) as Maximum_number from ethereumetl.blocks

-- COMMAND ----------

-- TBD
-- maximum date
select max(date(cast(timestamp as timestamp))) from ethereumetl.blocks limit 100

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

--TBD
select number,cast(timestamp as timestamp) block_date from ethereumetl.blocks 
where number in 
(
select block_number from ethereumetl.token_transfers 
where block_number is not null
group by block_number 
)
order by block_date
limit 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

select count(distinct address) from ethereumetl.silver_contracts
WHERE is_erc20 = "True"

-- COMMAND ----------

select count(address) from ethereumetl.silver_contracts
WHERE is_erc20 = "True"

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q: What percentage of transactions are calls to contracts

-- COMMAND ----------

-- TBD
select (a.con_tran_num/b.tran_num) con_tran_rate
from
(
  (
    select count(*) con_tran_num from ethereumetl.transactions
    where to_address in 
    (
      select address from ethereumetl.silver_contracts
      group by address
    )
  )a
  join
  (
    select count(*) tran_num from ethereumetl.transactions
  )b
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

-- TBD
select token_address,count(transaction_hash) tran_cnt
from ethereumetl.token_transfers
group by token_address
order by tran_cnt desc
limit 100


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

-- await

-- COMMAND ----------

-- TBD
select a.erc20_new_cnt/b.erc20_cnt
from 
(
  select count(*) erc20_new_cnt
  from
  (
    select token_address,count(transaction_hash) tran_cnt
    from ethereumetl.token_transfers
    group by token_address
    having tran_cnt==1
  )
)a
join
(
  select count(distinct token_address) erc20_cnt
  from ethereumetl.token_transfers
)b

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

select block_number,count(*)cnt from (select * from ethereumetl.transactions limit 10000)
group by block_number
order by cnt desc

-- COMMAND ----------

-- TBD
--sample 4405004
select * from ethereumetl.transactions
where block_number=4405004
order by transaction_index

-- COMMAND ----------

-- MAGIC %md order by gas price

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

select number,transaction_count / 15 from ethereumetl.blocks
order by transaction_count desc 
limit 1

-- COMMAND ----------

-- TBD
--from transactions
select block_number,count(*)/15 throughput
from ethereumetl.transactions
group by block_number
order by throughput desc
limit 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

-- TBD
select sum(value)/10e+18 total_ether
from ethereumetl.transactions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total gas used in all transactions?

-- COMMAND ----------

-- TBD
SELECT sum(gas_used) gas_used
from ethereumetl.receipts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

-- TBD
select  *
from ethereumetl.token_transfers
order by value desc
limit 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Token balance for any address on any date?

-- COMMAND ----------

-- TBD
select address,block_date,sum(value) over (partition by address order by block_date) from
( select address,block_number,sum(value) value
from
(
(select from_address address,-value value,block_number
from ethereumetl.token_transfers
where from_address='0xff29d3e552155180809ea3a877408a4620058086'
)
union 
(select to_address address,value value,block_number
from ethereumetl.token_transfers
where to_address='0xff29d3e552155180809ea3a877408a4620058086'

)
)
group by address,block_number
)a
left join
(
select date(cast(timestamp as timestamp)) block_date,number from ethereumetl.blocks
group by block_date,number
)b
on a.block_number=b.number


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz the transaction count over time (network use)

-- COMMAND ----------

-- TBD
select address,block_date,sum(value) over (partition by address order by block_date) from
( select hash,
from ethereumetl.transactions
where value is not null and block_number is not null
)a
left join
(
select date(cast(timestamp as timestamp)) block_date,number from ethereumetl.blocks
group by block_date,number
)b
on a.block_number=b.number
limit 100

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

-- TBD


-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
