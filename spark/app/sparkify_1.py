from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

###########

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

spark = SparkSession.builder.appName("Sparkify").getOrCreate()

# load and clean the dataset
mini_sparkify_event_data = "/usr/local/spark/resources/data/mini_sparkify_event_data.json"

df = spark.read.json(mini_sparkify_event_data)
df.createOrReplaceTempView("sparkify_events")

# clean the dataset
# remove userId, sessionId null
df = spark.sql("""
    SELECT userId, gender, location, level, sessionId, page, artist, song, length, userAgent, registration, ts
    FROM sparkify_events
    WHERE userId != '' 
        AND userId IS NOT NULL 
        AND sessionId IS NOT NULL
    ORDER BY userId, ts
""")
df.createOrReplaceTempView("sparkify_events_1")

# Create Indexer for userId and artist
user_indexer = StringIndexer(inputCol="userId", outputCol="userIdIndex")
artist_indexer = StringIndexer(inputCol="artist", outputCol="artistIndex", handleInvalid="keep")

# Create labels for user and artist
user_indexer_model = user_indexer.fit(df)
artist_indexer_model = artist_indexer.fit(df)

# Transform dataframe
df = user_indexer_model.transform(df)
df = artist_indexer_model.transform(df)

# Convert timestamp
df = df.withColumn("elapsed", df.ts - df.registration)
df = df.withColumn("timestamp", F.to_timestamp(F.from_unixtime(df.ts/1000)))
df = df.withColumn("registration_date", F.to_timestamp(F.from_unixtime(df.registration/1000)))

# Elapsed Time in days (Days between registration and activity)
df = df.withColumn("elapsed_days", F.round(df.elapsed/(24*60*60*1000), 0).cast("integer"))

# Remove negative elapsed_days
print(df.filter(df.elapsed_days < 0).count())
df = df.filter(df.elapsed_days >= 0)
df.createOrReplaceTempView("sparkify_events_2")

# We will use this afterwards
# Create a View with each userId last week activity
last_week_df = spark.sql("""
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
FROM
(
    SELECT *, MAX(ts) OVER(PARTITION BY userId) AS max_ts
    FROM sparkify_events_2
)
WHERE ts >= max_ts - 7*24*60*60*1000
""")

last_week_df.createOrReplaceTempView("last_week_events")
