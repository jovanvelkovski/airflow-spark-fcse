import sys
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("Hello World").getOrCreate()

csv_file = sys.argv[1]

df = spark.read.csv(csv_file, header=True, inferSchema= True)

df.printSchema()