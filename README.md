# 🚲 Time Series Forecasting – LSTM Bike Sharing Demand Prediction

## 📌 Project Overview
This project focuses on forecasting **hourly bike-sharing demand** using an **LSTM (Long Short-Term Memory) deep learning model**.  
The objective is to learn **temporal patterns, seasonality, and demand trends** from historical bike rental data and make accurate future predictions.

---

## 🎯 Problem Statement
Accurately predicting bike-sharing demand helps organizations:
- Optimize bike availability
- Improve operational efficiency
- Reduce shortages during peak hours

Given historical hourly rental data, the task is to **predict the next hour’s bike demand**.

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📂 Dataset
- **Source:** UCI Machine Learning Repository  
- **Type:** Hourly time-series data  
- **Target Variable:** `cnt` (total bike rentals per hour)

---

## 🔄 Project Workflow

### 1️⃣ Data Preprocessing
- Loaded and sorted data chronologically
- Selected `cnt` as the target time series
- Applied **MinMaxScaler**
- Scaler fitted only on training data to avoid data leakage

---

### 2️⃣ Sequence Creation
- Converted time-series data into supervised learning format
- Used a **24-hour sliding window**
  - Past 24 hours → Predict next hour

---

### 3️⃣ Train–Test Split
- Time-based split (80% train, 20% test)
- **No random shuffling** to preserve temporal order

---

### 4️⃣ Model Architecture
- Stacked LSTM model:
  - LSTM (64 units) + Dropout
  - LSTM (32 units) + Dropout
  - Dense output layer
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- EarlyStopping used to prevent overfitting

---

### 5️⃣ Model Training
- Epochs: 50  
- Batch size: 32  
- Validation split: 20%  
- EarlyStopping monitored validation loss  

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| RMSE | ≈ 58 |
| MAE | ≈ 36 |
| Normal Demand MAE | ≈ 30 |
| Peak Demand MAE | ≈ 95 |

✔ Strong performance during normal demand  
✔ Expected higher error during peak demand spikes  

---

## 📈 Visualizations
- Training vs Validation Loss
- Actual vs Predicted Demand
- Error Distribution
- Hour-wise Error Analysis
- Season-wise Error Analysis
- Error Trend Over Time
- Peak vs Normal Demand Comparison

---

## 🔍 Advanced Error Analysis
- **Hour-wise Error:** Higher errors during peak commute hours  
- **Season-wise Error:** Better performance in summer, higher error in winter  
- **Error Trend:** Stable over time (no performance degradation)  
- **Peak Demand:** More difficult due to sudden demand spikes  

---

## 💾 Model Saving
The trained model is saved using the **native Keras format**:

```python
model.save("lstm_bike_sharing_model.keras")
