import pytest
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import re
from pei_assessment import run_dq, get_enriched_products_customers, create_enriched_orders, aggregate

@pytest.fixture(scope="session")
def spark_session():
    spark = (SparkSession.builder
             .master("local[1]")
             .appName("Pytest-Spark-Local")
             .getOrCreate())
    yield spark
    spark.stop()


def clean_colname(x):
    return re.sub(r'[^a-zA-Z0-9]', '_', x.lower().strip())

def test_dq_customers(spark_session):
    df_customers = spark_session.createDataFrame(pd.read_excel('test_files/Customer.xlsx'))
    for col in df_customers.columns:
        df_customers = df_customers.withColumnRenamed(col, clean_colname(col))
    df_rules = spark_session.read.format('csv').option('header', True).load('test_files/dq_rules.csv').filter(F.col('dataset')=='customers')
    df_good, df_bad = run_dq(spark_session, df_rules, df_customers)
    df_good.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('good_data/customers/')
    df_bad.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('bad_data/customers/')


    assert df_good.filter(F.col('customer_id').isNull()).count() == 0, "Null values found in customer_id column of customers dataset in good records"
    assert df_good.groupBy('customer_id').count().filter(F.col('count')>1).count() == 0, "Duplicate customer_id found in customers dataset in good records"

    assert df_customers.filter(F.col('customer_id').isNull()).count()==df_bad.filter(F.col('rule_name')=='customer_id_null_check').count(), "Mismatch in null check rule for customer_id column in customers dataset in bad records"
    
    df1 = df_customers.filter(F.col('customer_id').isNotNull()).groupBy('customer_id').count().filter(F.col('count')>1)
    df2 = df_bad.filter(F.col('rule_name')=='customer_id_uniqueness_check').groupBy('customer_id').count().filter(F.col('count')>1)
    flag = (df1.subtract(df2).isEmpty() and df2.subtract(df1).isEmpty())
    
    assert flag, "Mismatch in uniqueness check rule for customer_id column in customers dataset in bad records"

def test_df_orders(spark_session):
    df_orders = spark_session.createDataFrame(pd.read_json('test_files/Orders.json', dtype=str))
    for col in df_orders.columns:
        df_orders = df_orders.withColumnRenamed(col, clean_colname(col))

    df_rules = spark_session.read.format('csv').option('header', True).load('test_files/dq_rules.csv').filter(F.col('dataset')=='orders')
    df_good, df_bad = run_dq(spark_session, df_rules, df_orders)
    
    df_good.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('good_data/orders/')
    df_bad.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('bad_data/orders/')

    assert df_good.filter(F.col('order_id').isNull()).count() == 0, "Null values found in order_id column of orders dataset in good records"
    assert df_good.groupBy('order_id').count().filter(F.col('count')>1).count() == 0, "Duplicate order_id found in orders dataset in good records"

    assert df_orders.filter(F.col('order_id').isNull()).count()==df_bad.filter(F.col('rule_name')=='order_id_null_check').count(), "Mismatch in null check rule for order_id column in orders dataset in bad records"

    df1 = df_orders.filter(F.col('order_id').isNotNull()).groupBy('order_id').count().filter(F.col('count')>1)
    df2 = df_bad.filter(F.col('rule_name')=='order_id_uniqueness_check').groupBy('order_id').count().filter(F.col('count')>1)
    flag = (df1.subtract(df2).isEmpty() and df2.subtract(df1).isEmpty())
    assert flag, "Mismatch in uniqueness check rule for order_id column in orders dataset in bad records"

def test_df_products(spark_session):
    df_products = spark_session.read.format('csv')\
                    .option('header','true')\
                    .option('inferSchema','false')\
                    .option('quote','"')\
                    .option('escape', '"')\
                    .load('test_files/Products.csv')
    for col in df_products.columns:
        df_products = df_products.withColumnRenamed(col, clean_colname(col))
    df_rules = spark_session.read.format('csv').option('header', True).load('test_files/dq_rules.csv').filter(F.col('dataset')=='products')
    df_good, df_bad = run_dq(spark_session, df_rules, df_products)

    df_good.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('good_data/products/')
    df_bad.repartition(1).write.format('csv').option('header', True).mode('overwrite').save('bad_data/products/')

    assert df_good.filter(F.col('product_id').isNull()).count() == 0, "Null values found in product_id column of products dataset in good records"
    assert df_good.groupBy('product_id').count().filter(F.col('count')>1).count() == 0, "Duplicate product_id found in products dataset in good records"

    assert df_products.filter(F.col('product_id').isNull()).count()==df_bad.filter(F.col('rule_name')=='product_id_null_check').count(), "Mismatch in null check rule for product_id column in products dataset in bad records"

    df1 = df_products.filter(F.col('product_id').isNotNull()).groupBy('product_id').count().filter(F.col('count')>1)
    df2 = df_bad.filter(F.col('rule_name')=='product_id_uniqueness_check').groupBy('product_id').count().filter(F.col('count')>1)
    df1.show()
    df2.show()
    flag = (df1.subtract(df2).isEmpty() and df2.subtract(df1).isEmpty())

    assert flag, "Mismatch in uniqueness check rule for product_id column in products dataset in bad records"

def test_enriched_products_customers(spark_session):
    df_orders = spark_session.read.format('csv').option('header', True).load('good_data/orders/')
    df_products = spark_session.read.format('csv').option('header', True).load('good_data/products/')
    df_customers = spark_session.read.format('csv').option('header', True).load('good_data/customers/')

    df_orders.createOrReplaceTempView("orders")
    df_products.createOrReplaceTempView("products")
    df_customers.createOrReplaceTempView("customers")

    df_enriched = get_enriched_products_customers(df_orders, df_products, df_customers)

    df_test = spark_session.sql(""" 
    select c.customer_id, c.customer_name, c.email, c.phone, c.address, c.segment as customer_segment, 
        c.country as customer_country, c.city as customer_city, c.state as customer_state, 
        lpad(c.postal_code, 5, '0') as customer_postal_code, c.region as customer_region, 
        p.product_id, p.category as product_category, p.sub_category as product_sub_category, 
        p.product_name, p.state as product_state, p.price_per_product
    from (select customer_id, product_id from orders) as o
    join customers c on o.customer_id = c.customer_id
    join products p on o.product_id = p.product_id
    """)

    assert df_enriched.subtract(df_test).count() == 0 and df_test.subtract(df_enriched).count() == 0, "Data mismatch in enriched products customers dataset"

def test_enriched_orders(spark_session):
    df_orders = spark_session.read.format('csv').option('header', True).load('good_data/orders/')
    df_products = spark_session.read.format('csv').option('header', True).load('good_data/products/')
    df_customers = spark_session.read.format('csv').option('header', True).load('good_data/customers/')

    df_orders.createOrReplaceTempView("orders")
    df_products.createOrReplaceTempView("products")
    df_customers.createOrReplaceTempView("customers")

    df_enriched = create_enriched_orders(df_orders, df_products, df_customers)

    

    df_test = spark_session.sql(""" 
    select o.order_id, 
            to_date(o.order_date, 'd/M/yyyy') as order_date, 
            to_date(o.ship_date, 'd/M/yyyy') as ship_date, 
            o.ship_mode, o.customer_id, o.product_id, quantity, price, discount, 
            format_number(cast(o.profit AS decimal(10,5)), 2) as profit,
            c.customer_name, c.country as customer_country, p.category as product_category, 
            p.sub_category as product_sub_category
    from orders as o
    join customers c on o.customer_id = c.customer_id
    join products p on o.product_id = p.product_id
    """)

    assert df_enriched.subtract(df_test).count() == 0 and df_test.subtract(df_enriched).count() == 0, "Data mismatch in enriched orders dataset"


def test_aggregate(spark_session):
    df_orders = spark_session.read.format('csv').option('header', True).load('good_data/orders/')
    df_products = spark_session.read.format('csv').option('header', True).load('good_data/products/')
    df_customers = spark_session.read.format('csv').option('header', True).load('good_data/customers/')

    # df_orders.createOrReplaceTempView("orders")
    # df_products.createOrReplaceTempView("products")
    # df_customers.createOrReplaceTempView("customers")

    df_enriched = create_enriched_orders(df_orders, df_products, df_customers)
    df_agg = aggregate(df_enriched)

    df_enriched.createOrReplaceTempView("enriched_orders")

    df_test = spark_session.sql("""
                with cte as 
                (
                    select *, 
                    year(order_date) as order_year, 
                    cast(regexp_replace(profit, '[^0-9-.]', '') as decimal(10,2)) as profit_val 
                    from enriched_orders
                )
                
                select order_year, product_category, product_sub_category, customer_id, 
                        sum(profit_val) as total_profit
                from cte
                group by order_year, product_category, product_sub_category, customer_id
                """)
    
    assert df_agg.subtract(df_test).count() == 0 and df_test.subtract(df_agg).count() == 0, "Data mismatch in aggregated dataset"
                