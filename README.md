# docker-spark-fcse

1. Clone project
- mkdir airflow-spark-fcse
- cd airflow-spark-fcse
- git clone https://github.com/jovanvelkovski/airflow-spark-fcse

2. Build airflow docker
- cd airflow-spark-fcse/airflow
- docker build --rm -t docker-airflow2:latest .

3. Setup the sandbox
- cd ..
- cp -R sandbox/. ../sandbox

4. Launch containers
- docker-compose -f docker-compose.yml up -d

5. Create a test user for airflow
- docker-compose run airflow-webserver airflow users create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin

6. Edit connection from Airflow to Spark
Go to Airflow UI > Admin > Edit connections
Edit spark_default entry:
    Connection Type: Spark
    Host: spark://spark
    Port: 7077