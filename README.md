# Orchestration of processing tasks using Apache Airflow and Apache Spark

Docker with Airflow + Postgres + Spark cluster + JDK (spark-submit support)

## 📦 Containers

* **airflow-webserver**: Airflow webserver and scheduler, with spark-submit support.
    * image: docker-airflow2:latest (Airflow version 2.4.1)
        * Based on python:3.8, [pyjaime/docker-airflow-spark](https://github.com/pyjaime/docker-airflow-spark), [puckel/docker-airflow](https://github.com/puckel/docker-airflow) and [cordon-thiago/airflow-spark](https://github.com/cordon-thiago/airflow-spark/)
    * port: 8080

* **postgres**: Postgres database, used by Airflow.
    * image: postgres:13.6
    * port: 5432

* **spark-master**: Spark Master.
    * image: bitnami/spark:3.3.1
    * port: 8081

* **spark-worker[1]** and **spark-worker[2]**: Spark workers
    * image: bitnami/spark:3.3.1

## 🛠 Setup

### Clone project
	
	$ mkdir airflow-spark-fcse
    $ cd airflow-spark-fcse
    $ git clone https://github.com/jovanvelkovski/airflow-spark-fcse
   
### Build airflow Docker

    $ cd airflow-spark-fcse/airflow
    $ docker build --rm -t docker-airflow2:latest .

### Launch containers

    $ cd ..
    $ docker-compose -f docker-compose.yml up -d
    $ docker exec -it airflow-spark-fcse-spark-worker-1-1 /bin/bash
    $ pip install numpy && exit
    $ docker exec -it airflow-spark-fcse-spark-worker-2-1 /bin/bash
    $ pip install numpy && exit

### Check accesses

* Airflow: http://localhost:8080
* Spark Master: http://localhost:8081

### Enable Airflow user
  
    $ docker-compose run airflow-webserver airflow users create --role Admin --username admin \
      --email admin --firstname admin --lastname admin --password admin

### Enable Spark connection

* Go to Airflow UI > Admin > Edit connections
* Edit spark_default entry:
  * Connection Type: Spark
  * Host: spark://spark
  * Port: 7077 
