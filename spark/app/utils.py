from pyspark.sql import SparkSession

from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.functions import vector_to_array

from pyspark.mllib.linalg.distributed import CoordinateMatrix

spark = SparkSession.builder.appName("Sparkify").getOrCreate()

sparkify_events_data = "/usr/local/airflow/spark-data/sparkify_events.parquet"
last_week_events_data = "/usr/local/airflow/spark-data/last_week_events.parquet"

sparkify_events_df = spark.read.parquet(sparkify_events_data)
last_week_df = spark.read.parquet(last_week_events_data)

sparkify_events_df.createOrReplaceTempView("sparkify_events")
last_week_df.createOrReplaceTempView("last_week_events")


def prefix_column(dataframe, prefix):
    """Prefix the columns of a dataframe

    Args:
        dataframe (object): A Spark Dataframe with a userIdIndex column, and several other columns
        prefix (str): The string to prefix the other columns

    Returns:
        object: A Spark Dataframe with renamed columns

    """
    cols = dataframe.columns.copy()
    cols.remove("userIdIndex")

    for col in cols:
        dataframe = dataframe.withColumnRenamed(col, f"{prefix}_{col}")

    return dataframe


# Create an artist rating View for each user
# For each user the rating is the number of songs played for that artist divided by the most song counts of an artist
# Range between 0-1
def create_ratings(table_name):
    """Create a ratings dataframe for a given table with events

    Args:
        table_name (str): The name of SQL table
    """
    rating_df = spark.sql(
        f"""
    SELECT userIdIndex, artistIndex, COUNT(song) / MAX(COUNT(song)) OVER(PARTITION BY userIdIndex) AS rating
    FROM {table_name}
    WHERE artist IS NOT NULL
    GROUP BY userIdIndex, artistIndex
    """
    )

    rating_df.createOrReplaceTempView(f"{table_name}_rating")


def get_als_features(ratings, factors, num_factors=20, output_col="vector"):
    """Get features for each user given the ratings dataframe and the item factor matrix.

    Args:
        ratings (object): A Spark Dataframe with the columns userIdIndex, artistIndex and rating
        factors (IndexedRowMatrix): A Matrix with Item Factors from a trained ALS model
        output_col (str): The name of the output column features

    Returns:
        object: A Spark Dataframe with columns userIdIndex and several output_col in the number of number
        of factors

    """
    mat = CoordinateMatrix(
        ratings.select("userIdIndex", "artistIndex", "rating").rdd.map(tuple)
    )
    mat = mat.toIndexedRowMatrix()

    # Dot multiplication between user-artist ratings matrix and artist factors
    dataframe = mat.multiply(factors).rows.toDF()

    # Transform array to columns
    dataframe = dataframe.withColumn(
        "userIdIndex", dataframe["index"].cast(DoubleType())
    )
    dataframe = dataframe.withColumn(output_col, vector_to_array("vector"))
    dataframe = dataframe.select(
        ["userIdIndex"] + [dataframe[output_col][i] for i in range(num_factors)]
    )

    return dataframe


## Percentage of access for each page for each user
# Remove Cancel and Cancellation Confirmation
def create_page_percentage(table_name):
    """For each user calculate the percentage of a
    specific page interaction over all pages interactions.

    Remove page `Cancel` and `Cancellation Confirmation`
    because they are used to predict churn, not to serve as features.

    Args:
        table_name (str): A Spark SQL table name

    Returns:
        object: A Spark Dataframe

    """

    page_df = spark.sql(
        f"""
    SELECT userIdIndex, page, COUNT(page) / SUM(COUNT(page)) OVER(PARTITION BY userIdIndex) AS page_perct
    FROM {table_name}
    WHERE page IS NOT NULL AND page NOT IN ('Cancel', 'Cancellation Confirmation')
    GROUP BY userIdIndex, page
    """
    )

    return page_df


# Total page access
def create_page_counts(table_name):
    """Calculate the number of pages access for each user

    Remove page `Cancel` and `Cancellation Confirmation`
    because they are used to predict churn, not to serve as features.

    Args:
        table_name (str): A Spark SQL table name

    Returns:
        object: A Spark Dataframe
    """

    page_df = spark.sql(
        f"""
    SELECT userIdIndex, page, COUNT(page) AS page_count
    FROM {table_name}
    WHERE page IS NOT NULL AND page NOT IN ('Cancel', 'Cancellation Confirmation')
    GROUP BY userIdIndex, page
    """
    )

    return page_df


# Level percentage
# Drop free (because paid = 1 - free, no necessity to have both as they are linearly dependent, thus highly correlated)
def create_level_percentage(table_name):
    """Calculate the level percentage for each user by using the number of events
    that happened in each level

    Args:
        table_name (str): A Spark SQL table name

    Returns:
        object: A Spark Dataframe
    """

    level_df = spark.sql(
        f"""
    SELECT userIdIndex, level, COUNT(level) / SUM(COUNT(level)) OVER(PARTITION BY userIdIndex) AS level_perct
    FROM {table_name}
    WHERE level IS NOT NULL
    GROUP BY userIdIndex, level
    """
    )

    return level_df


def assembler(features_df):
    """Transform the features columns into a single feature column as a vector

    Args:
        features_df (object): A Spark Dataframe with columns: userIdIndex, several features, label

    """
    cols = features_df.columns.copy()
    cols.remove("userIdIndex")
    cols.remove("label")

    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    features_df = assembler.transform(features_df).select(
        "userIdIndex", "features", "label"
    )

    return features_df
