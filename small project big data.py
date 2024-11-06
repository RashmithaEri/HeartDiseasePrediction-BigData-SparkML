Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, isnan, isnull
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType  # Import data types for casting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("HeartDiseasePrediction").getOrCreate()

# Load the dataset
df = spark.read.csv("/content/framingham_heart_disease.csv", header=True, inferSchema=True)

# Display first few rows and data summary
df.show(5)
df.describe().show()
df.printSchema()

# Convert specific columns to numeric types
for column in ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']:
    df = df.withColumn(column, df[column].cast(DoubleType()))

# Fill missing values with the mean of each column
df = df.na.fill({col: df.select(mean(col)).first()[0] for col in df.columns if df.select(mean(col)).first()[0] is not None})

# Verify that there are no remaining null values
df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Convert Spark DataFrame to Pandas for correlation matrix and heatmap visualization
pandas_df = df.toPandas()

# Replace 'NA' with pandas-compatible NaN values
pandas_df.replace('NA', pd.NA, inplace=True)  # Replace 'NA' with pandas-compatible NaN

# Ensure all object-type columns are converted to numeric, coercing errors to NaN
for column in pandas_df.select_dtypes(include=['object']).columns:
    pandas_df[column] = pd.to_numeric(pandas_df[column], errors='coerce')

# Fill remaining NaN values in pandas_df with column means for consistency
pandas_df.fillna(pandas_df.mean(), inplace=True)

# Visualization: Correlation heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(pandas_df.corr(numeric_only=True), annot=True)
plt.show()

# Feature engineering: Prepare the features and label columns
assembler = VectorAssembler(inputCols=[col for col in df.columns if col != 'TenYearCHD'], outputCol="features")
df = assembler.transform(df).select("features", "TenYearCHD")

# Split data into training and test sets
train, test = df.randomSplit([0.7, 0.3], seed=42)

# Initialize and train Logistic Regression model
lr = LogisticRegression(labelCol="TenYearCHD", featuresCol="features")
lr_model = lr.fit(train)

# Make predictions on the test set
predictions = lr_model.transform(test)

# Evaluate metrics
# 1. AUC
binary_evaluator = BinaryClassificationEvaluator(labelCol="TenYearCHD", metricName="areaUnderROC")
auc = binary_evaluator.evaluate(predictions)

# 2. Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

# 3. Precision and Recall
precision_evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="weightedRecall")
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Display the results
print(f"Model Accuracy: {accuracy}")
print(f"Model AUC: {auc}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")

# Stop Spark session
spark.stop()
