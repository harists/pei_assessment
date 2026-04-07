# Databricks notebook source
# Databricks notebook source
import pandas as pd
import re
import json
from pyspark.sql import functions as F

def load_raw_data(spark):
    df_products = spark.read.format('csv')\
                    .option('header','true')\
                    .option('inferSchema','false')\
                    .option('quote','"')\
                    .option('escape', '"')\
                    .load('/Volumes/workspace/default/raw_data/Products.csv')
    # df_products.show()

    df_customers = spark.read.format('com.crealytics.spark.excel')\
                        .excel('/Volumes/workspace/default/raw_data/Customer.xlsx', headerRows=1)
    # df_customers.show()

    p_df = pd.read_json('/Volumes/workspace/default/raw_data/Orders.json', dtype=str)
    df_orders = spark.createDataFrame(p_df)
    # df_orders.show()

    def clean_colname(x):
        return re.sub(r'[^a-zA-Z0-9]', '_', x.lower().strip())

    for col in df_customers.columns:
        df_customers = df_customers.withColumnRenamed(col, clean_colname(col))

    for col in df_orders.columns:
        df_orders = df_orders.withColumnRenamed(col, clean_colname(col))

    for col in df_products.columns:
        df_products = df_products.withColumnRenamed(col, clean_colname(col))
    
    return df_orders, df_products, df_customers

    


def create_schemas(spark):
    spark.sql('create schema if not exists raw_data;')
    spark.sql('create schema if not exists dq_good')
    spark.sql('create schema if not exists dq_bad')
    spark.sql('Create schema if not exists enriched_layer')
    spark.sql('Create schema if not exists aggregate_layer')

def run_dq(spark, rules, df):
    good_columns = df.columns

    rules = rules.withColumn('rule_name', F.lower(F.concat_ws('_', F.col('column'), F.col('rule_type'))))

    df_bad = df.limit(0).withColumn('rule_name',F.lit(None))

    good_rules = []
    for rule in rules.collect():
        print('*'*10)
        print(rule)
        rule_type = rule['rule_type']
        if rule_type=='NULL_CHECK':
            cond = F.col(rule['column']).isNull()
            df_bad = df_bad.union(df.filter(cond).select(*good_columns).withColumn('rule_name', F.lit(rule['rule_name'])))
        elif rule_type=='UNIQUENESS_CHECK':
            column = rule['column']
            df_unique = df.select(column).groupBy(column).count()\
                    .withColumnRenamed("count", column+"_unique_status")\
                    .withColumnRenamed(column, column+"_key")\
                    .filter(F.col(column+'_unique_status')>1)

            df = df.alias("a").join(df_unique.alias("b"), df[column] == df_unique[column+"_key"], "left") \
                    .select("a.*","b."+column+"_unique_status")
            cond = (F.col(column+"_unique_status").isNotNull()) & (F.col(column).isNotNull())
            df_bad = df_bad.union(df.filter(cond).drop(column+"_unique_status").select(*good_columns).withColumn('rule_name', F.lit(rule['rule_name'])))
        good_rules.append(cond)
    
    for r in good_rules:
        df = df.filter(~r)

    df_good = df.select(*good_columns)

    return df_good, df_bad

def create_enriched_orders(df_orders, df_products, df_customers):
    df_en_orders = df_orders.alias('o')\
                .join(df_products.alias('p'), df_orders.product_id==df_products.product_id, 'inner')\
                .join(df_customers.alias('c'), df_customers.customer_id == df_orders.customer_id, 'inner')\
                .withColumn('order_date', F.to_date(F.col('order_date'), 'd/M/yyyy'))\
                .withColumn('ship_date', F.to_date(F.col('ship_date'), 'd/M/yyyy'))\
                .withColumn('profit', F.format_number(F.col('profit').cast('decimal(10,5)'), 2))\
                .selectExpr('o.order_id', 'order_date', 'ship_date', 'ship_mode', 'o.customer_id', 'o.product_id', 'quantity', 'price', 'discount', 'profit',\
                             'c.customer_name', 'c.country as customer_country', 'p.category as product_category', 'p.sub_category as product_sub_category')
    return df_en_orders

def get_enriched_products_customers(df_orders, df_products, df_customers):
    df_en_p_c = df_orders.select('customer_id','product_id').alias('o')\
                .join(df_products.alias('p'), df_orders.product_id==df_products.product_id, 'inner')\
                .join(df_customers.alias('c'), df_customers.customer_id == df_orders.customer_id, 'inner')\
                .selectExpr('c.customer_id', 'c.customer_name', 'c.email', 'c.phone', 'c.address', 'c.segment as customer_segment', 'c.country as customer_country', \
                            'c.city as customer_city', 'c.state as customer_state', 'c.postal_code as customer_postal_code', 'c.region as customer_region',  \
                            'p.product_id', 'p.category as product_category', 'p.sub_category as product_sub_category', 'p.product_name', 'p.state as product_state',\
                            'p.price_per_product')\
                .withColumn('customer_postal_code', F.lpad(F.col('customer_postal_code'), 5, '0'))
    return df_en_p_c

def aggregate(df_en_orders):
    df_agg = df_en_orders.withColumn('order_year', F.year('order_date'))\
                .withColumn('profit', F.regexp_replace('profit', '[^0-9-.]', '').cast('decimal(10,2)'))\
                .groupBy('order_year', 'product_category', 'product_sub_category', 'customer_id')\
                .agg(F.sum('profit').alias('total_profit'))
    return df_agg


# COMMAND ----------

def main(spark):

    create_schemas(spark)
    df_orders, df_products, df_customers =load_raw_data(spark)

    df_customers.write.format('delta').mode('overwrite').saveAsTable('raw_data.customers')
    df_products.write.format('delta').mode('overwrite').saveAsTable('raw_data.products')
    df_orders.write.format('delta').mode('overwrite').saveAsTable('raw_data.orders')

    df_customers = spark.read.table('raw_data.customers')
    df_orders = spark.read.table('raw_data.orders')
    df_products = spark.read.table('raw_data.products')
    dataset_df_mapping = {
        'customers' : df_customers,
        'orders' : df_orders,
        'products' : df_products
    }

    df_rules = spark.read.format('csv').option('header', True).load('/Volumes/workspace/default/mappings/dq_rules.csv')
    

    datasets = df_rules.select('dataset').distinct().collect()

    df_rules.show()

    for d in datasets:
        dataset_name = d.dataset
        print('-'*20)
        print(dataset_name)
        df = dataset_df_mapping[dataset_name]
        df_good, df_bad = run_dq(spark, df_rules.filter(F.col('dataset')==dataset_name), df)
        df_good.write.format('delta').mode('overwrite').saveAsTable(f'dq_good.{dataset_name}')
        df_bad.write.format('delta').mode('overwrite').saveAsTable(f'dq_bad.{dataset_name}')

    df_customers = spark.read.table('dq_good.customers')
    df_orders = spark.read.table('dq_good.orders')
    df_products = spark.read.table('dq_good.products')

    df_q2 = get_enriched_products_customers(df_orders, df_products, df_customers)
    df_q2.write.format('delta').mode('overwrite').saveAsTable('enriched_layer.customer_product_data')

    df_en_orders = create_enriched_orders(df_orders, df_products, df_customers)
    df_en_orders.write.format('delta').mode('overwrite').saveAsTable('enriched_layer.orders_data')
        
    df_agg = aggregate(df_en_orders)
    df_agg.show()
    df_agg.write.format('delta').mode('overwrite').saveAsTable('aggregate_layer.profit_by_year_category_subcategory_customer')

if __name__=='__main__':
    main(spark)


# COMMAND ----------

# MAGIC %sql
# MAGIC select order_year, sum(total_profit) as year_wise_profit
# MAGIC from aggregate_layer.profit_by_year_category_subcategory_customer
# MAGIC group by order_year
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select order_year, product_category, sum(total_profit) as year_by_product_wise_profit
# MAGIC from aggregate_layer.profit_by_year_category_subcategory_customer
# MAGIC group by order_year, product_category

# COMMAND ----------

# MAGIC %sql
# MAGIC select customer_id, sum(total_profit) as customer_wise_profit
# MAGIC from aggregate_layer.profit_by_year_category_subcategory_customer
# MAGIC group by customer_id

# COMMAND ----------

# MAGIC %sql
# MAGIC select customer_id, order_year, sum(total_profit) as customer_by_year_wise_profit
# MAGIC from aggregate_layer.profit_by_year_category_subcategory_customer
# MAGIC group by customer_id, order_year