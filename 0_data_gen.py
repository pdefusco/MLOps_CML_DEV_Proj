#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata


class BankDataGen:

    '''Class to Generate Banking Data'''

    def __init__(self, spark, username):
        self.spark = spark
        self.username = username

    def bankDataGen(self, shuffle_partitions_requested = 8, partitions_requested = 8, data_rows = 10000):

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(self.spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("name", percentNulls=0.1, text=FakerTextUS("name") )
                    .withColumn("address", text=FakerTextUS("address" ))
                    .withColumn("email", text=FakerTextUS("ascii_company_email") )
                    .withColumn("age", "decimal", minValue=10, maxValue=100, random=True)
                    .withColumn("credit_card_balance", "decimal", minValue=100, maxValue=30000, random=True)
                    .withColumn("bank_account_balance", "decimal", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("mortgage_balance", "decimal", minValue=0.01, maxValue=1000000, random=True)
                    .withColumn("sec_bank_account_balance", "decimal", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("savings_account_balance", "decimal", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("sec_savings_account_balance", "decimal", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("total_est_nworth", "decimal", minValue=10000, maxValue=500000, random=True)
                    .withColumn("primary_loan_balance", "decimal", minValue=0.01, maxValue=5000, random=True)
                    .withColumn("secondary_loan_balance", "decimal", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("college_loan_balance", "decimal", minValue=0.01, maxValue=10000, random=True)
                    .withColumn("aba_routing", text=FakerTextUS("aba" ))
                    .withColumn("bank_country", text=FakerTextUS("bank_country") )
                    .withColumn("account_no", text=FakerTextUS("bban" ))
                    .withColumn("int_account_no", text=FakerTextUS("iban") )
                    .withColumn("swift11", text=FakerTextUS("swift11" ))
                    .withColumn("credit_card_number", text=FakerTextUS("credit_card_number") )
                    .withColumn("credit_card_provider", text=FakerTextUS("credit_card_provider") )
                    .withColumn("event_type", "string", values=["purchase", "cash_advance"],random=True)
                    .withColumn("longitude", "float", minValue=-180, maxValue=180, random=True)
                    .withColumn("latitude", "float", minValue=-90, maxValue=90, random=True)
                    .withColumn("transaction_currency", values=["USD", "EUR", "KWD", "BHD", "GBP", "CHF", "MEX"])
                    .withColumn("transaction_amount", "decimal", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("fraud", values=["YES", "NO"], random=True, weights=[9, 1])
                    )
        df = fakerDataspec.build()

        return df


# Sample in-code customization of spark configurations
from pyspark import SparkContext
SparkContext.setSystemProperty('spark.executor.cores', '2')
SparkContext.setSystemProperty('spark.executor.memory', '4g')

import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
#from pyspark import SparkContext
#SparkContext.setSystemProperty('spark.executor.cores', '1')
#SparkContext.setSystemProperty('spark.executor.memory', '2g')

CONNECTION_NAME = "go01-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()
username = os.environ["PROJECT_OWNER"]
dbname = "MLOPS"

#---------------------------------------------------
#       SQL CLEANUP: DATABASES, TABLES, VIEWS
#---------------------------------------------------
print("JOB STARTED...")
#spark.sql("DROP DATABASE IF EXISTS {} CASCADE".format(dbname))

spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(dbname))

print("SHOW DATABASES LIKE '{}'".format(dbname))
spark.sql("SHOW DATABASES LIKE '{}'".format(dbname)).show()
print("\n")

#---------------------------------------------------
#               CREATE BATCH DATA
#---------------------------------------------------

print("CREATING BANKING TRANSACTIONS\n")

dg = BankDataGen(spark, username)

bankTransactionsDf = dg.bankDataGen()

def createOrAppend(df, dbname, username):
  """
  Method to create or append data to the BANKING TRANSACTIONS table
  The table is used to simulate batches of new data
  The table is meant to be updated periodically as part of a CML Job
  """
  
  try:
    print("TRY TO APPEND NEW BATCH OF DATA\n")
    df.writeTo("{0}.BANKING_TRANSACTIONS_{1}".format(dbname, username))\
      .using("iceberg").tableProperty("write.format.default", "parquet").append()
    print("TABLE WAS FOUND AND DATA WAS APPENDED\n")
    
  except:
    print("TABLE WAS NOT FOUND\n")
    print("CREATING TABLE NOW\n")
    df.writeTo("{0}.BANKING_TRANSACTIONS_{1}".format(dbname, username))\
        .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()
    print("TABLE CREATED AND POPULATED WITH DATA\n")
    
createOrAppend(bankTransactionsDf, dbname, username)
    
print("BATCH LOAD JOB COMPLETED\n")
