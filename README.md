# CaliforniaHousing
EndToEndMLProject
California Housing Price Prediction 🏡💰
This project is an End-to-End Machine Learning pipeline to predict median house values in California using Scikit-Learn, Pandas, Matplotlib, and Machine Learning models.

📌 Project Overview
This project follows an end-to-end ML workflow, covering: ✅ Data loading & exploration
✅ Geospatial visualization of housing prices
✅ Handling missing values & categorical attributes
✅ Feature engineering & transformations
✅ Training & evaluating ML models (Decision Tree, Random Forest)
✅ Fine-tuning models using Cross-Validation

1️⃣ Dataset 📂
The dataset used is the California Housing Dataset, available in Scikit-learn.

Features include:

🏠 longitude, latitude: Geographic location
👨‍👩‍👧‍👦 population: Number of people in the area
🏡 median_house_value: Target variable (house price)
💵 median_income: Average income of residents
To load the dataset:

python
Copy
Edit
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.head()
2️⃣ Project Structure 🏗️
bash
Copy
Edit
📂 california-housing-ml
│── 📂 data                   # Raw dataset (if applicable)
│── 📂 notebooks              # Jupyter Notebooks
│── 📂 models                 # Trained models
│── 📂 src                    # ML pipeline scripts
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│── main.py                   # Run the ML pipeline
│── setup.py                  # Package setup
3️⃣ Steps in the Project 🔍
📊 Data Exploration & Visualization
Display dataset statistics
Plot scatter maps of houses using Matplotlib
python
Copy
Edit
import matplotlib.pyplot as plt
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="Population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()
🔍 Correlation Matrix
Understanding feature relationships
python
Copy
Edit
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
⚡ Preprocessing & Feature Engineering
Handling missing values
One-hot encoding categorical features
Scaling numerical data
python
Copy
Edit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop("median_house_value", axis=1))
🏆 Model Training & Evaluation
Decision Tree Model
python
Copy
Edit
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
Random Forest Model
python
Copy
Edit
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
🔧 Fine-Tuning with Cross-Validation
python
Copy
Edit
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
