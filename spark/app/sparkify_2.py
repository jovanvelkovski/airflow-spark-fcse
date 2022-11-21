from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

spark = SparkSession.builder.appName("Exploratory Data Analysis").getOrCreate()

sparkify_events_data = "/usr/local/airflow/spark-data/sparkify_events.parquet"
churn_data = "/usr/local/airflow/spark-data/churn.parquet"

df = spark.read.parquet(sparkify_events_data)
churn = spark.read.parquet(churn_data)

df.createOrReplaceTempView("sparkify_events")
churn.createOrReplaceTempView("churn")

date_ranges = spark.sql("""
SELECT
    MIN(date(registration_date)) AS min_registration_date, 
    MAX(date(registration_date)) AS max_registration_date,
    MIN(date(timestamp)) AS min_activity_date, 
    MAX(date(timestamp)) AS max_activity_date
FROM sparkify_events
""").toPandas()

registration_date = spark.sql("""
SELECT registration, label, COUNT(userIdIndex) AS count FROM
(
    SELECT 
        events.userIdIndex, 
        events.registration, 
        churn.label 
    FROM
    (
        SELECT 
            userIdIndex, 
            MIN(date(registration_date)) AS registration 
        FROM sparkify_events
        GROUP BY userIdIndex
    ) AS events
    INNER JOIN churn 
        ON events.userIdIndex = churn.userIdIndex
)
GROUP BY registration, label
""")

reg_df = registration_date.toPandas()
reg_df = reg_df.pivot(index="registration", columns="label").fillna(0)
reg_df.columns = ["Not Cancelled", "Cancelled"]
reg_df_cumsum = reg_df.cumsum()

fig, axes = plt.subplots(figsize=(20, 8), nrows=1, ncols=2)

reg_df.plot(ax = axes[0], xlabel = "Registration Date", ylabel = "User Count")
reg_df_cumsum.plot(ax = axes[1], xlabel = "Registration Date", ylabel = "Cumulative User Count")
plt.savefig("/usr/local/airflow/spark-data/chart1_UserCountByRegistrationDate.png")

# Last Activity date
last_activity_date = spark.sql("""
SELECT
    ts, 
    label, 
    COUNT(userIdIndex) AS count
FROM
(
    SELECT
        events.userIdIndex, 
        events.ts, 
        churn.label 
    FROM
    (
        SELECT 
            userIdIndex, 
            MAX(date(timestamp)) AS ts 
        FROM sparkify_events
        GROUP BY userIdIndex
    ) AS events
    INNER JOIN churn
        ON events.userIdIndex = churn.userIdIndex
)
GROUP BY ts, label
""")

lad_df = last_activity_date.toPandas()
lad_df = lad_df.pivot(index = "ts", columns = "label").fillna(0)
lad_df.columns = ["Not Cancelled", "Cancelled"]
lad_df_cumsum = lad_df.cumsum()

fig, axes = plt.subplots(figsize = (20, 8), nrows=1, ncols=2)

lad_df.plot(ax = axes[0], xlabel="Last Activity Date", ylabel="User Count")
lad_df_cumsum.plot(ax = axes[1], xlabel="Last Activity Date", ylabel="Cumulative User Count")
plt.savefig("/usr/local/airflow/spark-data/chart2_UserCountByLastActivityDate.png")

mean_user_age = spark.sql("""
SELECT 
    ts, 
    label, 
    COUNT(userIdIndex) AS users, 
    AVG(age) AS age FROM
(
    SELECT 
        events.userIdIndex, 
        events.ts, 
        events.age, 
        churn.label 
    FROM
    (
        SELECT 
            userIdIndex, 
            date(timestamp) AS ts, 
            AVG(elapsed_days) AS age 
        FROM sparkify_events
        GROUP BY userIdIndex, date(timestamp)
    ) AS events
    INNER JOIN churn 
        ON events.userIdIndex = churn.userIdIndex
)
GROUP BY ts, label
""")

mean_user_age_df = mean_user_age.toPandas()

mua_df = mean_user_age_df.pivot(index="ts", columns="label", values="age").fillna(0)
mua_df.columns = ["Not Cancelled", "Cancelled"]

user_count_df = mean_user_age_df.pivot(index="ts", columns="label", values="users").fillna(0)
user_count_df.columns = ["Not Cancelled", "Cancelled"]

fig, axes = plt.subplots(figsize=(20, 8), nrows=1, ncols=2)

mua_df.plot(ax=axes[0], xlabel="Activity Date", ylabel="Mean User Age")
user_count_df.plot(ax=axes[1], xlabel="Activity Date", ylabel="User Count")
plt.savefig("/usr/local/airflow/spark-data/chart3_MeanUserAgeByActivityDate.png")

user_age = spark.sql("""
SELECT events.userIdIndex, age, avg_age, label FROM 
(
    SELECT userIdIndex, MAX(elapsed_days) AS age, AVG(elapsed_days) AS avg_age
    FROM sparkify_events
    GROUP BY userIdIndex
) AS events
INNER JOIN churn ON churn.userIdIndex = events.userIdIndex
""")

user_age_df = user_age.toPandas()

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

user_age_df.loc[user_age_df["label"] == 0, "age"].plot(ax=axes, kind="hist")
user_age_df.loc[user_age_df["label"] == 1, "age"].plot(ax=axes, kind="hist")

axes.set_xlabel("User Age")
axes.set_ylabel("User Count")

plt.savefig("/usr/local/airflow/spark-data/chart4_UserCountByUserAge.png")


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

user_age_df.loc[user_age_df["label"] == 0, "avg_age"].plot(ax=axes, kind="hist")
user_age_df.loc[user_age_df["label"] == 1, "avg_age"].plot(ax=axes, kind="hist")

axes.set_xlabel("User Mean Age")
axes.set_ylabel("User Count")

plt.savefig("/usr/local/airflow/spark-data/chart5_UserCountByUserMeanAge.png")

song_played_per_user_per_day = spark.sql("""
SELECT user_age_when_played, label, 
       COUNT(userIdIndex) AS users_count, 
       SUM(songs_played) AS songs_played, 
       AVG(songs_played) AS avg_songs_played, 
       STD(songs_played) AS std_songs_played,
       SUM(page_interactions) AS page_interactions,
       AVG(page_interactions) AS avg_page_interactions,
       STD(page_interactions) AS std_page_interactions
FROM 
(
    SELECT events.userIdIndex, events.user_age_when_played, events.songs_played, events.page_interactions, churn.label FROM 
    (
        SELECT userIdIndex, elapsed_days AS user_age_when_played, SUM(IF(page = 'NextSong', 1, 0)) AS songs_played, SUM(IF(page = 'NextSong', 0, 1)) AS page_interactions 
        FROM sparkify_events
        GROUP BY userIdIndex, elapsed_days
    ) AS events
    INNER JOIN churn
    ON events.userIdIndex = churn.userIdIndex
) AS table2
GROUP BY label, user_age_when_played
ORDER BY user_age_when_played, label
""")

sppupd_df = song_played_per_user_per_day.toPandas()
sppupd_df = sppupd_df.pivot(index="user_age_when_played", columns="label")

uc = sppupd_df["users_count"].fillna(0)
uc.columns = ["Not Cancelled", "Cancelled"]

plt.figure(figsize=(8, 6))
uc.plot(xlabel="User Age When Event Happened", ylabel="User Count")

plt.savefig("/usr/local/airflow/spark-data/chart6_UserAgeWhenAnEventHappened.png")

asp_df = sppupd_df["avg_songs_played"].fillna(0)
asp_df.columns = ["Not Cancelled", "Cancelled"]

api_df = sppupd_df["avg_page_interactions"].fillna(0)
api_df.columns = ["Not Cancelled", "Cancelled"]

ssp_df = sppupd_df["std_songs_played"].fillna(0)
ssp_df.columns = ["Not Cancelled", "Cancelled"]

spi_df = sppupd_df["std_page_interactions"].fillna(0)
spi_df.columns = ["Not Cancelled", "Cancelled"]

fig, axes = plt.subplots(figsize=(20, 16), nrows=2, ncols=2, sharex=True, sharey=True)

asp_df.plot(ax=axes[0, 0], xlabel="Activity Date", ylabel="Songs Played", title="Avg Songs Played")
ssp_df.plot(ax=axes[0, 1], xlabel="Activity Date", ylabel="Songs Played", title="Std Songs Played")

asp_df.plot(ax=axes[1, 0], xlabel="Activity Date", ylabel="Page Interactions", title="Avg Page Interactions")
ssp_df.plot(ax=axes[1, 1], xlabel="Activity Date", ylabel="Page Interactions", title="Std Page Interactions")

plt.savefig("/usr/local/airflow/spark-data/chart7_SongsPlayedAndPageInteractionsByActivityDate.png")

sessions = spark.sql("""
SELECT activity_time, label,
       COUNT(sessionId) AS sessions, 
       AVG(interactions) AS interactions, 
       AVG(session_time) AS session_time
FROM
(
    SELECT events.userIdIndex, events.interactions, events.sessionId, events.session_time, events.activity_time, churn.label FROM 
    (
        SELECT userIdIndex, sessionId, COUNT(sessionId) AS interactions, MIN(elapsed_days) AS activity_time, (MAX(ts) - MIN(ts))/1000 AS session_time 
        FROM sparkify_events
        GROUP BY userIdIndex, sessionId
    ) AS events
    INNER JOIN churn
    ON events.userIdIndex = churn.userIdIndex
)
GROUP BY activity_time, label
ORDER BY activity_time, label
""")


sessions_df = sessions.toPandas()
sessions_df = sessions_df.pivot(index="activity_time", columns="label").fillna(0)

si_df = sessions_df["interactions"].fillna(0)
si_df.columns = ["Not Cancelled", "Cancelled"]

sst_df = sessions_df["session_time"].fillna(0)
sst_df.columns = ["Not Cancelled", "Cancelled"]

fig, axes = plt.subplots(figsize=(20, 8), nrows=1, ncols=2)

si_df.plot(ax=axes[0], xlabel="Activity Day", ylabel="Mean Interactions", title="Mean Interactions per Activity Day")
sst_df.plot(ax=axes[1], xlabel="Activity Day", ylabel="Mean Session Time (s)", title="Mean Session Time per Activity Day")

plt.savefig("/usr/local/airflow/spark-data/chart8_InteractionsAndSessionTimeByActivityDay.png")
