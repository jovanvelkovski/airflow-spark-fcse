from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
# from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from pyspark.conf import SparkConf  

spark_master = "spark://spark:7077"
conf = {
    "spark.master" : spark_master,
    "spark.submit.deployMode" : "client",
    "spark.network.timeout" : "300s"
}
spark_app_name = "Spark Hello World"
mini_sparkify_event_data = "/usr/local/spark/resources/data/mini_sparkify_event_data.json"

args = {
    'owner': 'Airflow',
}

with DAG(
    dag_id='test_spark_submit_operator',
    default_args=args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['test'],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    load_and_clean_dataset = SparkSubmitOperator(
        task_id="load_and_clean_dataset",
        application="/usr/local/spark/app/sparkify_1.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        application_args=[mini_sparkify_event_data],
        dag=dag
    )

    exploratory_data_analysis = SparkSubmitOperator(
        task_id="exploratory_data_analysis",
        application="/usr/local/spark/app/sparkify_2.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    end = DummyOperator(task_id="end", dag=dag)

    start  >> \
    load_and_clean_dataset >> \
    exploratory_data_analysis >> \
    end
    