
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.figure(figsize=(15,10))

for i, col in enumerate(numeric_cols):
    plt.subplot(3,2,i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# CORRELATION HEATMAP


corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#  TIME-SERIES TREND (ONE USER)


user_id = df["User_ID"].iloc[0]
user_data = df[df["User_ID"] == user_id].sort_values("Date")

plt.figure(figsize=(12,5))
plt.plot(user_data["Date"], user_data["Heart_Rate (bpm)"])
plt.title(f"Heart Rate Trend - User {user_id}")
plt.xticks(rotation=45)
plt.show()

 #. USER-LEVEL AGGREGATION


user_summary = df.groupby("User_ID")[numeric_cols].mean()

print("\nUser-Level Average Summary:\n")
print(user_summary.head())

plt.figure(figsize=(8,5))
sns.histplot(user_summary["Heart_Rate (bpm)"], kde=True)
plt.title("Average Heart Rate Distribution Across Users")
plt.show()

workout_counts = df["Workout_Type"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(workout_counts,
        labels=workout_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Workout Type Distribution")
plt.show()

print("\nFirst 5 rows after preprocessing:")
print(df.head())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


