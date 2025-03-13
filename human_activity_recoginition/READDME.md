# Human Activity Recognition Using Logistic Regression

## 📂 Project Structure

```
Human_Activity_Recognition
│
├── data_for_project
│   ├── UCI HAR Dataset
│   └── new_sensor_data.csv
│
├── models
│   └── logistic_model.pkl
│
├── results
│   └── predictions.csv
│
├── src
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── main.py
│
└── README.md
```

## 🎯 Project Aim

This project focuses on **Human Activity Recognition (HAR)** using sensor data from the **UCI HAR Dataset**. A **Logistic Regression Model** is trained and used to predict human activities based on new sensor data.

## 🛠️ Steps Followed

1. **Data Preprocessing**: Cleaning and structuring data.
2. **Model Training**: Logistic Regression model training and saving.
3. **Prediction Pipeline**: Predicting activity on new sensor data.
4. **Results Storage**: Saving the predictions in `results/predictions.csv`.

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/username/Human_Activity_Recognition.git
```

2. Navigate to the project folder

```bash
cd Human_Activity_Recognition
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the main script for prediction

```bash
python main.py
```

## 📈 Results

Predicted activities will be saved in the `results/predictions.csv` file.

## 📊 Output Files

- **Model File**: `models/logistic_model.pkl`
- **Predictions**: `results/predictions.csv`
- **Confusion Matrix and ROC Curve** (optional for evaluation): `results/metrics.png`

## 🛑 Dependencies

- Python 3.8+
- Pandas
- Scikit-learn
- Numpy

## ✅ Future Scope

- Adding Deep Learning Models
- Real-time sensor data integration
- Visualizing the activity patterns

## 🌟 Credits

Dataset Source: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---




