# Heart Disease Prediction Using Statistical ML with Spark for Big Data Analysis

## Project Overview
This project uses the Framingham Heart Disease dataset to predict the likelihood of cardiovascular disease using Apache Spark and logistic regression. The objective is to leverage Spark's distributed computing capabilities to preprocess and analyze large-scale health data, identify key risk factors, and assess model performance. The analysis was conducted on a two-VM configuration to optimize computational efficiency and demonstrate the scalability of Spark for big data analytics.

## Dataset
The dataset used is the [Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) from Kaggle, containing around 5,000 records. It includes features such as:
- Age
- Cholesterol
- Blood pressure
- Smoking habits
- Glucose levels

These factors are used to predict the presence of cardiovascular disease.

## Virtual Machine Configuration
To perform distributed computing, I set up two virtual machines (VMs) as follows:
- **IP Addresses:** 192.168.13.113 and 192.168.13.114
- **DNS:** 192.168.13
- **Gateway:** 192.168.13.1
- **Subnet Mask:** 255.255.255.0
- **Cores per VM:** 8
- **Spark Master URL:** `spark://hadoop1:7077`

The Spark Master node was initialized on `hadoop1`, with both VMs configured as worker nodes for data processing tasks. Each VM had Spark pre-installed, and configuration scripts were run to initialize master and worker services under `/opt/spark/sbin/`.

## Data Preprocessing
The following steps were undertaken to prepare the dataset for analysis:
1. **Handling Missing Values:** Missing values in columns such as `cigsPerDay`, `BPMeds`, `totChol`, `BMI`, `heartRate`, and `glucose` were filled with column mean values to ensure consistency and reduce biases.
2. **Data Type Conversion:** Columns with incorrect data types (e.g., strings for numerical features) were cast to `DoubleType` using Spark's `withColumn` method for ML compatibility.
3. **Standardizing Data (Scaling):** MinMax scaling was applied to continuous variables (e.g., age, BMI, blood pressure) to normalize data and prevent larger-scale features from dominating the model.
4. **Data Splitting:** The dataset was divided into training (70%) and test (30%) sets, with the training set used to fit the model and the test set for performance evaluation.

## Statistical Analysis
A correlation matrix was generated to analyze feature relationships. By converting the Spark DataFrame to a Pandas DataFrame, I visualized correlations using a heatmap. Notably, age and systolic blood pressure were positively correlated, aligning with known risk factors for cardiovascular disease.

## Machine Learning Model
The logistic regression model was implemented as follows:
1. **Feature Engineering:** All features, except the target variable (`TenYearCHD`), were combined into a single vector column using Spark's `VectorAssembler`.
2. **Model Training:** A logistic regression model was trained with `TenYearCHD` as the label and the vectorized features as the input. Logistic regression is well-suited for binary classification tasks and effectively estimated the probability of heart disease.

## Model Evaluation
Model performance was assessed using the following metrics:
- **Accuracy:** 86.9%
- **AUC (Area Under ROC):** 0.707
- **Precision and Recall:** Precision was 0.83, and recall was 0.87 for predicting heart disease.

## Performance Comparison
The program's computational efficiency was tested under two configurations:
1. **Single VM Configuration:** Processing time was approximately 2.8 minutes.
2. **Two VM Configuration:** Execution time reduced significantly to 28 seconds, demonstrating the benefits of a distributed setup for big data processing.

## Conclusion
This project demonstrates the potential of using Apache Spark and logistic regression for scalable health data analysis, specifically in predicting heart disease risk. The correlation analysis provided insights into key cardiovascular indicators, while logistic regression achieved high accuracy. Future exploration could include advanced models like Random Forests or Deep Neural Networks for improved predictive performance. This project highlights the role of big data analytics in healthcare, with applications in clinical decision support and personalized patient care.

## Python Packages Used
- `pyspark` for data processing and machine learning in Spark
- `matplotlib` and `seaborn` for data visualization
- `pandas` for data manipulation

## Dataset Link
[Framingham Heart Study Dataset on Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

