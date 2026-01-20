# Sampling Techniques on Imbalanced Credit Card Dataset

## 1. Introduction
This project focuses on understanding the role of sampling techniques in handling imbalanced datasets and analyzing how different sampling strategies affect the performance of various machine learning models. A credit card fraud detection dataset is used, where fraudulent transactions are significantly fewer than non-fraudulent ones.

## 2. Dataset Description
The dataset contains credit card transaction records with the following characteristics:
- Class = 0 represents non-fraudulent transactions
- Class = 1 represents fraudulent transactions

The dataset is highly imbalanced, which can negatively affect machine learning model performance if not handled properly.

## 3. Handling Class Imbalance
To address the imbalance in the dataset:
- The dataset is divided into majority and minority classes
- The minority class is upsampled using random sampling with replacement
- A balanced dataset is created with equal representation of both classes

This step ensures that models learn patterns from both classes without bias.

## 4. Sampling Techniques
After balancing the dataset, five sampling strategies are applied to generate training and testing datasets:

1. Random Holdout Sampling  
2. Stratified Holdout Sampling  
3. Bootstrap Sampling  
4. K-Fold Sampling (Single Fold)  
5. Stratified K-Fold Sampling (Single Fold)  

These techniques help evaluate how different data-splitting methods influence model performance.

## 5. Machine Learning Models
The following machine learning models are trained and evaluated using each sampling technique:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## 6. Model Training and Evaluation
For each sampling technique:
- The dataset is split into training and testing sets
- Feature scaling is applied using StandardScaler
- Models are trained using the training data
- Predictions are made on the testing data
- Accuracy is calculated to evaluate performance

The accuracy values are stored in a result table for comparison.

## 7. Result Table
The result table displays the accuracy (%) of each model under different sampling techniques. Rows represent machine learning models, and columns represent sampling strategies.

## 8. Result Graph
A bar graph is generated from the result table to visually compare the performance of models across sampling techniques. This visualization helps in understanding performance trends.

## 9. Best Sampling Analysis
The analysis identifies:
- The best sampling technique for each model
- The overall best sampling technique based on average accuracy
- The best model and sampling technique combination

## 10. Conclusion
This assignment demonstrates that proper handling of imbalanced datasets is essential for reliable machine learning results. The choice of sampling technique has a significant impact on model performance.

## 11. How to Run
pip install -r requirements.txt  
python sampling_assignment.py

## 12. Author
Course Assignment – Sampling Techniques in Machine Learning
