from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

conf = {
    "spark.master" : "spark://spark:7077",
    "spark.network.timeout" : "300s",
    # "spark.executor.memory" : "512m",
    # "spark.driver.memory" : "512m",
    "spark.executor.memoryOverhead" : "1g",
    "spark.executor.cores" : 1,
    "spark.scheduler.mode" : "FAIR"
}
spark_app_name = "Spark Hello World"
args = {
    'owner': 'Airflow',
}

with DAG(
    dag_id='sparkify_churn_prediction',
    default_args=args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['test'],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    load_and_clean_dataset = SparkSubmitOperator(
        task_id="load_and_clean_dataset",
        application="/usr/local/spark/app/load_and_clean_dataset.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    check_dataset = FileSensor(
        task_id="check_dataset",
        poke_interval=30,
        timeout=60 * 5,
        mode="reschedule",
        filepath="/usr/local/airflow/spark-data/sparkify_events.parquet"
    )

    exploratory_data_analysis = SparkSubmitOperator(
        task_id="exploratory_data_analysis",
        application="/usr/local/spark/app/exploratory_data_analysis.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    feature_engineering = SparkSubmitOperator(
        task_id="feature_engineering",
        application="/usr/local/spark/app/feature_engineering.py",
        name=spark_app_name,
        conn_id="spark_default",
        verbose=1,
        conf=conf,
        dag=dag
    )

    with TaskGroup(group_id="modelling") as modelling:
        modelling_random_forest = SparkSubmitOperator(
            task_id="modelling_random_forest",
            application="/usr/local/spark/app/modelling_random_forest.py",
            name=spark_app_name,
            conn_id="spark_default",
            verbose=1,
            conf=conf,
            dag=dag
        )

        modelling_logistic_regression = SparkSubmitOperator(
            task_id="modelling_logistic_regression",
            application="/usr/local/spark/app/modelling_logistic_regression.py",
            name=spark_app_name,
            conn_id="spark_default",
            verbose=1,
            conf=conf,
            dag=dag
        )

        modelling_linear_svm = SparkSubmitOperator(
            task_id="modelling_linear_svm",
            application="/usr/local/spark/app/modelling_linear_svm.py",
            name=spark_app_name,
            conn_id="spark_default",
            verbose=1,
            conf=conf,
            dag=dag
        )

    with TaskGroup(group_id="check_features") as check_features:
        check_no_last_week_features = FileSensor(
            task_id="check_no_last_week_features",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/no_lw_features.parquet"
        )

        check_all_features = FileSensor(
            task_id="check_all_features",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/features.parquet"
        )

    with TaskGroup(group_id="check_insights") as check_insights:
        check_user_count_by_registration_date = FileSensor(
            task_id="check_user_count_by_registration_date",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart1_UserCountByRegistrationDate.png"
        )

        check_user_count_by_last_activity_date = FileSensor(
            task_id="check_user_count_by_last_activity_date",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart2_UserCountByLastActivityDate.png"
        )

        check_mean_user_age_by_activity_date = FileSensor(
            task_id="check_mean_user_age_by_activity_date",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart3_MeanUserAgeByActivityDate.png"
        )

        check_user_count_by_user_age = FileSensor(
            task_id="check_user_count_by_user_age",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart4_UserCountByUserAge.png"
        )

        check_user_count_by_user_mean_age = FileSensor(
            task_id="check_user_count_by_user_mean_age",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart5_UserCountByUserMeanAge.png"
        )

        check_user_age_when_an_event_happened = FileSensor(
            task_id="check_user_age_when_an_event_happened",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart6_UserAgeWhenAnEventHappened.png"
        )

        check_songs_played_and_page_interactions_by_activity_date = FileSensor(
            task_id="check_songs_played_and_page_interactions_by_activity_date",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart7_SongsPlayedAndPageInteractionsByActivityDate.png"
        )

        check_interactions_and_session_time_by_activity_day = FileSensor(
            task_id="check_interactions_and_session_time_by_activity_day",
            poke_interval=30,
            timeout=60 * 5,
            mode="reschedule",
            filepath="/usr/local/airflow/spark-data/chart8_InteractionsAndSessionTimeByActivityDay.png"
        )

        

    end = DummyOperator(task_id="end", dag=dag)
    
    start  >> \
    load_and_clean_dataset >> \
    check_dataset >> \
    exploratory_data_analysis >> \
    check_insights

    check_dataset >> \
    feature_engineering >> \
    check_features >> \
    modelling >> \
    end
