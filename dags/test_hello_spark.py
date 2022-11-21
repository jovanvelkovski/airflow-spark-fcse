from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

conf = {
    "spark.master" : "spark://spark:7077",
    "spark.network.timeout" : "300s",
    "spark.executor.memory" : "512m",
    "spark.driver.memory" : "512m",
    "spark.executor.memoryOverhead" : "1g",
    "spark.executor.cores" : 1,
    "spark.scheduler.mode" : "FAIR"
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

    waiting_for_file = FileSensor(
        task_id="waiting_for_file",
        poke_interval=30,
        timeout=60 * 5,
        mode="reschedule",
        filepath="/usr/local/airflow/spark-data/sparkify_events.parquet"
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

    feature_engineering = SparkSubmitOperator(
        task_id="feature_engineering",
        application="/usr/local/spark/app/sparkify_3.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    # modelling group
    modelling_random_forest = SparkSubmitOperator(
        task_id="modelling_random_forest",
        application="/usr/local/spark/app/sparkify_4.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    modelling_logistic_regression = SparkSubmitOperator(
        task_id="modelling_logistic_regression",
        application="/usr/local/spark/app/sparkify_5.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    modelling_linear_svm = SparkSubmitOperator(
        task_id="modelling_linear_svm",
        application="/usr/local/spark/app/sparkify_6.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    
    feature_engineering_done = DummyOperator(task_id="feature_engineering_done", dag=dag)
    end = DummyOperator(task_id="end", dag=dag)
    temp_end = DummyOperator(task_id="temp_end", dag=dag)

    start  >> \
    load_and_clean_dataset >> \
    waiting_for_file >> \
    exploratory_data_analysis >> \
    temp_end

    waiting_for_file >> \
    feature_engineering >> \
    feature_engineering_done >> \
    modelling_random_forest >> \
    end

    feature_engineering_done >> \
    modelling_logistic_regression >> \
    end
    
    feature_engineering_done >> \
    modelling_linear_svm >> \
    end