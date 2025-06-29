import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#here i added dataset from iris dataset 
try:
    df = pd.read_csv(r"C:\Users\Shraddha Saxena\OneDrive\Desktop\internship tasks\IRIS.csv")
except FileNotFoundError:
    st.error("❌ File 'IRIS.csv' not found in the current folder.")
    st.stop()
X = df.drop("species", axis=1)
y = df["species"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
#this is the basic ui for my task
st.markdown("### 🌼 Iris Flower Species Overview")
col1, col2 = st.columns([1, 3])
with col1:
 st.image("https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg", width=100, caption="Iris-setosa")
with col2:
 st.markdown("""
    🔹 Iris-setosa  
    _ Smallest petals among the three  
    - Typically grows in colder regions  
    - Sepals are broad and petals short  
    - Usually purple or blue in color  
    - Known for its symmetrical shape  
    """)
col3, col4 = st.columns([1, 3])
with col3:
 st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", width=100, caption="Iris-versicolor")
with col4:
 st.markdown("""
    🔹 Iris-versicolor
    - Medium-sized petals  
    - Grows in wetlands or swamps  
    - Varies from blue to violet shades  
    - Sepals droop slightly outward  
    - Often used in gardens and floral art  
    """)
col5, col6 = st.columns([1, 3])
with col5:
 st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", width=100, caption="Iris-virginica")
with col6:
 st.markdown("""
    🔹 Iris-virginica 
    - Largest petals of the three  
    - Thrives in moist, warm habitats  
    - Often deep violet to bluish-purple  
    - Sepals curve backward  
    - Considered showy and elegant  
    """)
#user input by sliders 
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)
#prediction button 
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max() * 100
    st.success(f"✅ Predicted Species: **{prediction}**")
    st.info(f"🧠 Model Confidence: **{confidence:.2f}%**")
#if any category matched it would show the iris image in results also
    species_images = {
    'Iris-setosa': "https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg",
    'Iris-versicolor': "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    'Iris-virginica': "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
    image_url = species_images.get(prediction)
    if image_url:
     st.image(image_url, caption=f"🌼 Predicted: **{prediction}**", use_container_width=True)
st.markdown("---")
st.subheader("📊 Feature Importance")

# Define feature meanings (explanations)
feature_explanations = {
    'sepal_length': "🟦 Helps distinguish species, but less informative than petals.",
    'sepal_width': "🟪 Slightly useful for setosa; often overlaps in other species.",
    'petal_length': "🟩 Highly important; longer petals often mean virginica.",
    'petal_width': "🟥 Most significant; wide petals are strong indicators."
}

# Calculate importances
feature_names = df.columns[:-1]
importances = model.feature_importances_

# Add explanations
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Score': importances,
    'Explanation': [feature_explanations.get(name, "") for name in feature_names]
}).sort_values(by="Importance Score", ascending=False)

# Show as styled table
st.dataframe(feature_df, use_container_width=True)

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(feature_df["Feature"], feature_df["Importance Score"], color="skyblue")
ax.set_xlabel("Importance Score")
ax.set_title("🔬 Feature Importance (Random Forest)")
ax.invert_yaxis()
st.pyplot(fig)

st.markdown("---")
st.subheader("🌺 Real-World Iris Examples")

st.markdown("""
<style>
.iris-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: center;
    margin-top: 20px;
}
.iris-card {
    background-color: #fdfdfd;
    border-radius: 12px;
    padding: 12px;
    width: 220px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
}
.iris-card:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.iris-card img {
    width: 100%;
    height: 160px;
    object-fit: cover;
    border-radius: 10px;
}
.iris-caption {
    font-weight: 600;
    margin-top: 10px;
}
.tooltip {
    font-size: 13px;
    color: #555;
    margin-top: 6px;
}
a.iris-link {
    text-decoration: none;
    color: inherit;
}
</style>

<div class="iris-grid">
  <a class="iris-link" href="https://en.wikipedia.org/wiki/Iris_setosa" target="_blank">
    <div class="iris-card">
      <img src="https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg" alt="Iris Setosa">
      <div class="iris-caption">Iris Setosa</div>
      <div class="tooltip">Small, purple blooms found in cool climates like Alaska & Canada.</div>
    </div>
  </a>

  <a class="iris-link" href="https://en.wikipedia.org/wiki/Iris_versicolor" target="_blank">
    <div class="iris-card">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/640px-Iris_versicolor_3.jpg" alt="Iris Versicolor">
      <div class="iris-caption">Iris Versicolor</div>
      <div class="tooltip">The Blue Flag Iris—seen near wetlands and ponds in North America.</div>
    </div>
  </a>

  <a class="iris-link" href="https://en.wikipedia.org/wiki/Iris_virginica" target="_blank">
    <div class="iris-card">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/640px-Iris_virginica.jpg" alt="Iris Virginica">
      <div class="iris-caption">Iris Virginica</div>
      <div class="tooltip">Large violet petals; prefers humid areas in the southeastern U.S.</div>
    </div>
  </a>
</div>
""", unsafe_allow_html=True)

# Caption
st.caption("""
📌 **Note:**  
Each feature contributes differently to species prediction.  
Petal features (especially **width** and **length**) are the strongest indicators,  
while sepal measurements play a smaller supporting role.
""")
