docker-compose -f docker-compose.yml up -d
docker exec -it airflow-spark-fcse-spark-worker-1-1 pip install numpy
docker exec -it airflow-spark-fcse-spark-worker-2-1 pip install numpy
docker exec -it airflow-spark-fcse-spark-worker-3-1 pip install numpy