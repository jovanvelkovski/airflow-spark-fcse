import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("Load and clean dataset").getOrCreate()

sparkify_event_data = sys.argv[1]
directory_path = sys.argv[2]

df = spark.read.json(sparkify_event_data)
df.createOrReplaceTempView("sparkify_events_raw")

df = spark.sql(
    """
    SELECT
        userId, 
        gender, 
        location, 
        level, 
        sessionId, 
        page, 
        artist, 
        song, 
        length, 
        userAgent, 
        registration, 
        ts
    FROM sparkify_events_raw
    WHERE userId != '' 
        AND userId IS NOT NULL 
        AND sessionId IS NOT NULL
    ORDER BY userId, ts
"""
)

user_indexer = StringIndexer(inputCol="userId", outputCol="userIdIndex")
artist_indexer = StringIndexer(
    inputCol="artist", outputCol="artistIndex", handleInvalid="keep"
)

user_indexer_model = user_indexer.fit(df)
artist_indexer_model = artist_indexer.fit(df)

df = user_indexer_model.transform(df)
df = artist_indexer_model.transform(df)

df = df.withColumn("elapsed", df.ts - df.registration)
df = df.withColumn("timestamp", F.to_timestamp(F.from_unixtime(df.ts / 1000)))
df = df.withColumn(
    "registration_date", F.to_timestamp(F.from_unixtime(df.registration / 1000))
)

df = df.withColumn(
    "elapsed_days", F.round(df.elapsed / (24 * 60 * 60 * 1000), 0).cast("integer")
)

df = df.filter(df.elapsed_days >= 0)
df.createOrReplaceTempView("sparkify_events")

df.write.mode("overwrite").parquet(f"{directory_path}/sparkify_events.parquet")

last_week_df = spark.sql(
    """
WITH cte AS (
    SELECT *, MAX(ts) OVER(PARTITION BY userId) AS max_ts
    FROM sparkify_events
)
SELECT
    userId, 
    gender, 
    location, 
    level, 
    sessionId, 
    page, 
    artist, 
    song, 
    length, 
    userAgent, 
    registration, 
    ts, 
    userIdIndex, 
    artistIndex, 
    registration_date, 
    timestamp, 
    elapsed, 
    elapsed_days
FROM cte
WHERE ts >= max_ts - 7*24*60*60*1000
"""
)

last_week_df.write.mode("overwrite").parquet(
    f"{directory_path}/last_week_events.parquet"
)

churn = spark.sql(
    """
SELECT 
    userIdIndex, 
    CAST(SUM(IF(page = 'Cancellation Confirmation', 1, 0)) >= 1 AS INT) AS label 
FROM sparkify_events
GROUP BY userIdIndex
"""
)
churn.write.mode("overwrite").parquet(f"{directory_path}/churn.parquet")
