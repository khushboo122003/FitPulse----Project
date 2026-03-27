
#import streamlit as st

#def main():
    #st.title("My First Streamlit App")
   # st.write("Hi Students")

#if __name__ == "__main__":
    #main()

# import streamlit as st

# def main():
#     st.title("Streamlit – Step 2")
#     st.write("Hi Students")

#     name = st.text_input("Enter your name:")
#     st.write("You typed:", name)

# if __name__ == "__main__":
#      main()

# import streamlit as st

# def main():
#      st.title("Streamlit – Step 3")
#      st.write("Hi Students")

#      name = st.text_input("Enter your name:")

#      if st.button("Greet"):
#         st.write("Hello", name)

# if __name__ == "__main__":
#     main()

# import streamlit as st

# def main():
#  st.title("Streamlit – Step 4")
# st.write("Hi Students")

# name = st.text_input("Enter your name:")

# st.write("Enter two numbers to add:")

# a = st.number_input("First number:", value=0)
# b = st.number_input("Second number:", value=0)

# if st.button("Calculate Sum"):
#          total = a + b
#          st.write("Hello", name)
#          st.write("Sum of the two numbers is:", total)

# if __name__ == "__main__":
#     main()

# import streamlit as st

# def main():
#      st.title("Streamlit – Step 5")
#      st.write("Hi Students")
#      name = st.text_input("Enter your name:")

#      a = st.number_input("First number:", value=0)
#      b = st.number_input("Second number:", value=0)

# operation = st.selectbox(
#          "Choose operation:",
#          ["Add", "Subtract", "Multiply"]
#      )
# if st.button("Calculate"):
#          if operation == "Add":
#             result = a + b
#          elif operation == "Subtract":
#             result = a - b
#          else:
#             result = a * b

#          st.write("Hello", name)
#          st.write("Operation:", operation)
#          st.write("Result:", result)

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🏃 FitPulse Health Dashboard")

# Load data
df = pd.read_csv("data/Fitness_Health_Tracking_Dataset.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Select user
user_id = st.selectbox("Select User", df["User_ID"].unique())

user_data = df[df["User_ID"] == user_id]

# Heart rate trend
st.subheader("Heart Rate Trend")

plt.figure(figsize=(10,4))
plt.plot(user_data["Date"], user_data["Heart_Rate (bpm)"])
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution
st.subheader("Steps Distribution")

plt.figure()
sns.histplot(df["Steps_Taken"], kde=True)
st.pyplot(plt)

# Show TSFresh features
try:
    features = pd.read_csv("data/tsfresh_features.csv")
    st.subheader("TSFresh Features")
    st.write(features.head())
except:
    st.warning("TSFresh file not found")

st.subheader("🧠 User Clustering (KMeans)")

try:
    st.image("kmeans_clusters.png")
    st.success("Users grouped into fitness clusters")
except:
    st.warning("KMeans image not found")

st.subheader("💡 Health Insights")

avg_hr = df["Heart_Rate (bpm)"].mean()
avg_steps = df["Steps_Taken"].mean()

if avg_hr > 90:
    st.error("⚠️ High heart rate detected")
elif avg_hr < 60:
    st.warning("⚠️ Low heart rate detected")
else:
    st.success("✅ Heart rate is normal")

if avg_steps < 5000:
    st.warning("⚠️ Low activity detected")
else:
    st.success("✅ Good daily activity")

st.subheader("👤 User Summary")

selected_user = st.selectbox(
    "Select User",
    df["User_ID"].unique(),
    key="user_summary"
)

user_data = df[df["User_ID"] == selected_user]

st.write(user_data.describe())