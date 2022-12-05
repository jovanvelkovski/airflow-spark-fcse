import sys
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from utils import train_test_model


plt.rcParams["figure.dpi"] = 140
directory_path = sys.argv[1]

spark = SparkSession.builder.appName("Modelling - Random Forest").getOrCreate()

no_lw_features_data = f"{directory_path}/no_lw_features.parquet"
churn_data = f"{directory_path}/churn.parquet"
features_data = f"{directory_path}/features.parquet"

no_lw_features = spark.read.parquet(no_lw_features_data)
churn = spark.read.parquet(churn_data)
features = spark.read.parquet(features_data)

churn_rate = churn.filter(churn.label == 1).count() / churn.count()

(training, test) = no_lw_features.randomSplit([0.75, 0.25], seed=42)
train_test_model(training, test, "Random Forest", directory=directory_path)

# Weight by label ratio
calc_weights = F.udf(
    lambda x: churn_rate if x == 0 else (1.0 - churn_rate), DoubleType()
)
features = features.withColumn("weights", calc_weights("label"))

# Train with weights
(training, test) = features.randomSplit([0.75, 0.25], seed=42)
model, prediction, fscore = train_test_model(
    training, test, "Random Forest", weights=True, directory=directory_path
)

# Metrics
pred = prediction.select("label", "prediction").toPandas()
y_true = pred["label"]
y_pred = pred["prediction"]
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print(cm)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"])
disp.plot(cmap=plt.cm.Blues)

plt.savefig(f"{directory_path}/chart9_RandomForest_ConfMatrix.png")
