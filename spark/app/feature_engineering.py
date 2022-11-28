import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from utils import (
    prefix_column,
    create_ratings,
    get_als_features,
    create_page_percentage,
    create_page_counts,
    create_level_percentage,
    assembler,
)

directory_path = sys.argv[1]
spark = SparkSession.builder.appName("Feature Engineering").getOrCreate()

sparkify_events_data = f"{directory_path}/sparkify_events.parquet"
last_week_data = f"{directory_path}/last_week_events.parquet"
churn_data = f"{directory_path}/churn.parquet"

df = spark.read.parquet(sparkify_events_data)
last_week_df = spark.read.parquet(last_week_data)
churn = spark.read.parquet(churn_data)

df.createOrReplaceTempView("sparkify_events")
last_week_df.createOrReplaceTempView("last_week_events")


# Create ratings for both total and last week events
create_ratings(spark, "sparkify_events")
create_ratings(spark, "last_week_events")

# Get rating dataframe for whole dataset
rating_df = spark.sql("SELECT * FROM sparkify_events_rating")
rating_last_week_df = spark.sql(
    """
SELECT
    t.userIdIndex, 
    t.artistIndex, 
    COALESCE(lw.rating, 0) AS rating 
FROM sparkify_events_rating AS t 
LEFT JOIN last_week_events_rating AS lw
    ON t.userIdIndex = lw.userIdIndex
    AND t.artistIndex = lw.artistIndex
"""
)


# Create ALS model and train (Reduce to 20 features)
als = ALS(
    rank=20,
    regParam=0.01,
    userCol="userIdIndex",
    itemCol="artistIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
)

model = als.fit(rating_df)

# Get item factors matrix from ALS model
factors = (
    IndexedRowMatrix(model.itemFactors.rdd.map(tuple)).toBlockMatrix().toLocalMatrix()
)

# All events
als_features_all = get_als_features(rating_df, factors, output_col="als_features_all")

# Last week events
rating_last_week_df = spark.sql(
    """
SELECT
    t.userIdIndex, 
    t.artistIndex, 
    COALESCE(lw.rating, 0) AS rating 
FROM sparkify_events_rating AS t 
LEFT JOIN last_week_events_rating AS lw
    ON t.userIdIndex = lw.userIdIndex AND t.artistIndex = lw.artistIndex
"""
)

als_features_lw = get_als_features(
    rating_last_week_df, factors, output_col="als_features_lw"
)

# All events
page_pct_all = (
    create_page_percentage(spark, "sparkify_events")
    .groupBy("userIdIndex")
    .pivot("page")
    .sum("page_perct")
    .na.fill(0.0)
)
page_pct_all = prefix_column(page_pct_all, prefix="page_pct_all")

# Last week events
page_pct_lw = (
    create_page_percentage(spark, "last_week_events")
    .groupBy("userIdIndex")
    .pivot("page")
    .sum("page_perct")
    .na.fill(0.0)
)
page_pct_lw = prefix_column(page_pct_lw, prefix="page_pct_lw")

# All events
page_count_all = (
    create_page_counts(spark, "sparkify_events")
    .groupby("userIdIndex")
    .pivot("page")
    .sum("page_count")
    .na.fill(0)
)
page_count_all = prefix_column(page_count_all, prefix="page_count_all")

# Last week events
page_count_lw = (
    create_page_counts(spark, "last_week_events")
    .groupby("userIdIndex")
    .pivot("page")
    .sum("page_count")
    .na.fill(0)
)
page_count_lw = prefix_column(page_count_lw, prefix="page_count_lw")

# Level of each user by getting its last reported level
level_indexer = StringIndexer(inputCol="level", outputCol="levelIndex")
level_encoder = OneHotEncoder(inputCols=["levelIndex"], outputCols=["l"])

level = spark.sql(
    """
WITH last_level AS 
(
    SELECT userIdIndex, level, ROW_NUMBER() OVER (PARTITION BY userIdIndex ORDER BY ts DESC) AS number
    FROM sparkify_events
)
SELECT userIdIndex, level FROM last_level WHERE number = 1
"""
)

level = level_indexer.fit(level).transform(level)
level = level_encoder.fit(level).transform(level).select("userIdIndex", "l")
level = level.withColumnRenamed("l", "level")

# All events
level_pct_all = (
    create_level_percentage(spark, "sparkify_events")
    .groupBy("userIdIndex")
    .pivot("level")
    .sum("level_perct")
    .na.fill(0.0)
    .drop("free")
)
level_pct_all = prefix_column(level_pct_all, prefix="level_pct_all")

# Last week events
level_pct_lw = (
    create_level_percentage(spark, "last_week_events")
    .groupBy("userIdIndex")
    .pivot("level")
    .sum("level_perct")
    .na.fill(0.0)
    .drop("free")
)
level_pct_lw = prefix_column(level_pct_lw, prefix="level_pct_lw")

# Gender
gender_encoder = OneHotEncoder(inputCols=["genderIndex"], outputCols=["g"])
gender_indexer = StringIndexer(inputCol="gender", outputCol="genderIndex")

gender = df.groupBy("userIdIndex").agg(F.first("gender").alias("gender"))
gender = gender_indexer.fit(gender).transform(gender)
gender = gender_encoder.fit(gender).transform(gender).select("userIdIndex", "g")
gender = gender.withColumnRenamed("g", "gender")

# Song length
song_length_all = (
    df.groupBy("userIdIndex")
    .agg(F.mean("length").alias("mean"), F.stddev("length").alias("std"))
    .na.fill(0.0)
)
song_length_all = prefix_column(song_length_all, prefix="song_length_all")

song_length_lw = (
    last_week_df.groupBy("userIdIndex")
    .agg(F.mean("length").alias("mean"), F.stddev("length").alias("std"))
    .na.fill(0.0)
)
song_length_lw = prefix_column(song_length_lw, prefix="song_length_lw")

# Hourly Song Counts
hourly_song_counts_all = (
    df.select("userIdIndex", F.hour(F.from_unixtime(df.ts / 1000)).alias("hour"))
    .groupBy("userIdIndex", "hour")
    .count()
    .groupBy("userIdIndex")
    .pivot("hour")
    .sum("count")
    .na.fill(0)
)
hourly_song_counts_all = prefix_column(
    hourly_song_counts_all, prefix="hourly_song_counts_all"
)

hourly_song_counts_lw = (
    last_week_df.select(
        "userIdIndex", F.hour(F.from_unixtime(last_week_df.ts / 1000)).alias("hour")
    )
    .groupBy("userIdIndex", "hour")
    .count()
    .groupBy("userIdIndex")
    .pivot("hour")
    .sum("count")
    .na.fill(0)
)
hourly_song_counts_lw = prefix_column(
    hourly_song_counts_lw, prefix="hourly_song_counts_lw"
)

# Hourly Mean Song Length
hourly_mean_song_length_all = (
    df.select(
        "userIdIndex", "length", F.hour(F.from_unixtime(df.ts / 1000)).alias("hour")
    )
    .groupBy("userIdIndex", "hour")
    .mean("length")
    .groupBy("userIdIndex")
    .pivot("hour")
    .mean("avg(length)")
    .na.fill(0)
)
hourly_mean_song_length_all = prefix_column(
    hourly_mean_song_length_all, prefix="hourly_mean_song_length_all"
)

hourly_mean_song_length_lw = (
    last_week_df.select(
        "userIdIndex",
        "length",
        F.hour(F.from_unixtime(last_week_df.ts / 1000)).alias("hour"),
    )
    .groupBy("userIdIndex", "hour")
    .mean("length")
    .groupBy("userIdIndex")
    .pivot("hour")
    .mean("avg(length)")
    .na.fill(0)
)
hourly_mean_song_length_lw = prefix_column(
    hourly_mean_song_length_lw, prefix="hourly_mean_song_length_lw"
)

elapsed_days = spark.sql(
    """
SELECT userIdIndex, MAX(elapsed_days) AS age, AVG(elapsed_days) AS avg_age
FROM sparkify_events
GROUP BY userIdIndex
"""
)

sessions = spark.sql(
    """
SELECT userIdIndex, 
       COUNT(sessionId) AS sessions_count, 
       AVG(pages_accessed) AS sessions_avg_pages_accessed,
       AVG(count_artists) AS sessions_avg_count_artists,
       AVG(avg_song_length) AS sessions_avg_song_length,
       AVG(session_time) AS sessions_avg_session_time,
       STD(pages_accessed) AS sessions_std_pages_accessed,
       STD(count_artists) AS sessions_std_count_artists,
       STD(avg_song_length) AS sessions_std_song_length,
       STD(session_time) AS sessions_std_session_time
FROM
(
    SELECT userIdIndex, sessionId, COUNT(song) AS songs_played, COUNT(page) AS pages_accessed, 
           COUNT(DISTINCT artist) AS count_artists, AVG(length) AS avg_song_length, 
           (MAX(ts) - MIN(ts)) / 1000 AS session_time       
    FROM sparkify_events
    GROUP BY userIdIndex, sessionId
)
GROUP BY userIdIndex
"""
)
# Fill std NAs with 0
sessions = sessions.fillna(0.0)

# Features Dataframe with no last week
no_lw_features = (
    als_features_all.join(page_pct_all, on=["userIdIndex"])
    .join(page_count_all, on=["userIdIndex"])
    .join(level_pct_all, on=["userIdIndex"])
    .join(song_length_all, on=["userIdIndex"])
    .join(hourly_song_counts_all, on=["userIdIndex"])
    .join(hourly_mean_song_length_all, on=["userIdIndex"])
    .join(elapsed_days, on=["userIdIndex"])
    .join(sessions, on=["userIdIndex"])
    .join(churn, on=["userIdIndex"])
)

# Features Dataframe with last week
a1 = als_features_all.join(als_features_lw, on=["userIdIndex"])
p1 = page_pct_all.join(page_pct_lw, on=["userIdIndex"])
p2 = page_count_all.join(page_count_lw, on=["userIdIndex"])
l1 = level_pct_all.join(level_pct_lw, on=["userIdIndex"])
s1 = song_length_all.join(song_length_lw, on=["userIdIndex"])
h1 = hourly_song_counts_all.join(hourly_song_counts_lw, on=["userIdIndex"])
h2 = hourly_mean_song_length_all.join(hourly_mean_song_length_lw, on=["userIdIndex"])

features = (
    a1.join(p1, on=["userIdIndex"])
    .join(p2, on=["userIdIndex"])
    .join(l1, on=["userIdIndex"])
    .join(s1, on=["userIdIndex"])
    .join(h1, on=["userIdIndex"])
    .join(h2, on=["userIdIndex"])
    .join(elapsed_days, on=["userIdIndex"])
    .join(sessions, on=["userIdIndex"])
    .join(churn, on=["userIdIndex"])
)

# Assemble features dataframes
no_lw_features = assembler(no_lw_features)
features = assembler(features)

no_lw_features.write.mode("overwrite").parquet(
    f"{directory_path}/no_lw_features.parquet"
)
features.write.mode("overwrite").parquet(f"{directory_path}/features.parquet")
