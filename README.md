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

