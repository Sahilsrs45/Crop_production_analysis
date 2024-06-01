# Pune Weather and Crop Data Analysis

## Overview
This project aims to analyze the weather data in Pune, India, and its correlation with crop production. It merges weather data with crop production data to identify patterns and relationships, visualizes the data, and builds machine learning models for crop prediction.

## Data Sources
- Pune weather data: CSV file (`pune.csv`)
- Crop production data: CSV file (`Monthly_data_cmo.csv`)

## Data Preprocessing
- Load weather and crop data into Pandas DataFrames.
- Filter and merge the data based on the date and district (Pune).
- Handle missing values and normalize the data.
  
## Exploratory Data Analysis (EDA)
- Explore correlations between weather variables and crop production.
- Visualize temperature variation, humidity, cloud cover, crop arrivals, etc.
- Analyze crop production trends over time.

## Machine Learning Models
- **K-Nearest Neighbors (KNN) Classifier:** Predict crop types based on weather features.
- **Decision Tree Classifier:** Predict crop types using decision trees.
- **Random Forest Classifier:** Ensemble learning method for crop prediction.

## Model Evaluation
- Evaluate model performance using accuracy scores, confusion matrices, and classification reports.
- Compare the performance of different models.

## Results
- KNN, Decision Tree, and Random Forest models are trained and evaluated for crop prediction.
- Random Forest Classifier outperforms other models with the highest accuracy.

## Conclusion
- Weather features significantly impact crop production in Pune.
- Machine learning models can effectively predict crop types based on weather data.
- Further optimization and fine-tuning of models can improve prediction accuracy.

## Repository Structure
- `data/`: Directory containing the dataset files.
- `notebooks/`: Jupyter notebooks for data analysis and model training.
- `results/`: Directory to save model evaluation results and visualizations.
- `scripts/`: Python scripts for data preprocessing and model training.
- `README.md`: Project overview and instructions.
- `requirements.txt`: File specifying project dependencies.

## Getting Started
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Explore the Jupyter notebooks in the `notebooks/` directory.
4. Run the scripts for data preprocessing and model training.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- Weather data sourced from [source](link).
- Crop production data sourced from [source](link).
- This project utilizes libraries such as Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, XGBoost, and Yellowbrick.

