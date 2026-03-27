
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv("data/Fitness_Health_Tracking_Dataset.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:")
print(numeric_columns)

print("\nMissing values before preprocessing:")
print(
    df.groupby("User_ID")[[
        "Hours_Slept",
        "Water_Intake (Liters)",
        "Active_Minutes",
        "Heart_Rate (bpm)"
    ]].apply(lambda x: x.isnull().sum())
)

numeric_cols = [
    "Hours_Slept",
    "Water_Intake (Liters)",
    "Active_Minutes",
    "Heart_Rate (bpm)"
]

df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
    lambda x: x.interpolate(method="linear")
)

df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
    lambda x: x.ffill().bfill()
)

df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")


# PROFESSIONAL EDA PIPELINE


print("\nDate Range:")
print("Start Date:", df["Date"].min())
print("End Date:", df["Date"].max())

numeric_cols = [
    "Steps_Taken",
    "Calories_Burned",
    "Hours_Slept",
    "Active_Minutes",
    "Heart_Rate (bpm)",
    "Stress_Level (1-10)"
]
# 4. OUTLIER DETECTION (BOXPLOTS)

plt.figure(figsize=(15,10))

for i, col in enumerate(numeric_cols):
    plt.subplot(3,2,i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.close()

# CORRELATION HEATMAP


corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.close()


#  TIME-SERIES TREND (ONE USER)


user_id = df["User_ID"].iloc[0]
user_data = df[df["User_ID"] == user_id].sort_values("Date")

plt.figure(figsize=(12,5))
plt.plot(user_data["Date"], user_data["Heart_Rate (bpm)"])
plt.title(f"Heart Rate Trend - User {user_id}")
plt.xticks(rotation=45)
plt.close()

 #. USER-LEVEL AGGREGATION


user_summary = df.groupby("User_ID")[numeric_cols].mean()

print("\nUser-Level Average Summary:\n")
print(user_summary.head())

plt.figure(figsize=(8,5))
sns.histplot(user_summary["Heart_Rate (bpm)"], kde=True)
plt.title("Average Heart Rate Distribution Across Users")
plt.close()

workout_counts = df["Workout_Type"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(workout_counts,
        labels=workout_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Workout Type Distribution")
plt.close()

print("\nFirst 5 rows after preprocessing:")
print(df.head())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

ts_df = df[["User_ID", "Date", "Heart_Rate (bpm)"]].copy().head(100)
ts_df = ts_df.dropna()
ts_df = ts_df.sort_values(["User_ID", "Date"])

ts_df = ts_df.rename(columns={
    "User_ID": "id",
    "Date": "time",
    "Heart_Rate (bpm)": "value"
})

print("Extracting TSFresh features...")

features = extract_features(
    ts_df,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters=MinimalFCParameters(),
    disable_progressbar=True,
    n_jobs=1
)

features = features.dropna(axis=1, how="all")

print("TSFresh Done:", features.shape)

features.to_csv("tsfresh_features.csv")
print("TSFresh features saved ✅")

# -------------------------------
# PROPHET FORECAST (HEART RATE)
# -------------------------------

# from prophet import Prophet

# # Prepare data
# prophet_df = df.groupby("Date")["Heart_Rate (bpm)"].mean().reset_index()
# prophet_df.columns = ["ds", "y"]

# prophet_df = prophet_df.dropna()
# prophet_df = prophet_df.sort_values("ds")

# print("Prophet input shape:", prophet_df.shape)

# # Train model
# model = Prophet()
# model.fit(prophet_df)

# # Future prediction
# future = model.make_future_dataframe(periods=30)
# forecast = model.predict(future)

# # Save plot (IMPORTANT: no plt.show)
# model.plot(forecast)
# plt.title("Heart Rate Forecast")
# plt.savefig("prophet_heart_rate.png")
# plt.close()

# print("Prophet graph saved ✅")

# -------------------------------
# CLUSTERING (KMEANS)
# -------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

cluster_cols = [
    "Steps_Taken",
    "Calories_Burned",
    "Hours_Slept",
    "Active_Minutes",
    "Heart_Rate (bpm)",
    "Stress_Level (1-10)"
]

cluster_df = df[cluster_cols].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_data)

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(6,5))
plt.scatter(pca_data[:,0], pca_data[:,1], c=labels)
plt.title("KMeans Clusters")
plt.savefig("kmeans_clusters.png")
plt.close()

print("Clustering done ✅")



