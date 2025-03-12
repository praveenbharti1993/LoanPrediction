{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9e631d8e-5e88-45ed-9080-0e03524ec455",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "import warnings \n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5f75e42a-eff3-4f8f-a17a-6e9ea3ba139e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Load Data ==========\n",
    "train_df = pd.read_csv('train_ctrUa4K.csv')\n",
    "test_df = pd.read_csv('test_lAUu6dG.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a8b77882-ec76-4170-ba7a-85cdf1ec1c8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Data Cleaning ==========\n",
    "# Impute missing values\n",
    "imputer_cat = SimpleImputer(strategy=\"most_frequent\")\n",
    "imputer_num = SimpleImputer(strategy=\"mean\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "76718647-46f1-4803-8801-86790a85de31",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Categorical\n",
    "train_df[['Gender', 'Married', 'Dependents', 'Self_Employed']] = imputer_cat.fit_transform(train_df[['Gender', 'Married', 'Dependents', 'Self_Employed']])\n",
    "test_df[['Gender', 'Married', 'Dependents', 'Self_Employed']] = imputer_cat.transform(test_df[['Gender', 'Married', 'Dependents', 'Self_Employed']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c2587b70-af1c-43b6-96c2-667b0642d886",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Numerical\n",
    "train_df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = imputer_num.fit_transform(\n",
    "    train_df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])\n",
    "test_df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = imputer_num.transform(\n",
    "    test_df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c1eaeef9-3749-4249-a354-ccfb58423dfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Feature Engineering ==========\n",
    "train_df['TotalIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']\n",
    "train_df['Debt_Income_Ratio'] = train_df['LoanAmount'] / train_df['TotalIncome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "436f99b6-f778-4e55-a649-1c2d069a0b73",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']\n",
    "test_df['Debt_Income_Ratio'] = test_df['LoanAmount'] / test_df['TotalIncome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "bc1e3363-188d-4b42-95b5-1e0c2c3c3828",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Encoding ==========\n",
    "label_encoder = LabelEncoder()\n",
    "train_df['Loan_Status'] = label_encoder.fit_transform(train_df['Loan_Status'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4b5b3a3e-7f64-43ef-b97c-a53ac9a7eab5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Combine for consistent encoding\n",
    "combined = pd.concat([train_df.drop('Loan_Status', axis=1), test_df])\n",
    "combined = pd.get_dummies(combined, drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "325dec97-4f56-4081-b135-171f62fab01a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split back\n",
    "train_encoded = combined.iloc[:len(train_df), :]\n",
    "test_encoded = combined.iloc[len(train_df):, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "937aefd9-a055-4bbe-8c13-eea3db1d5b3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Scaling ==========\n",
    "scaler = StandardScaler()\n",
    "train_encoded[['TotalIncome', 'Debt_Income_Ratio', 'LoanAmount']] = scaler.fit_transform(\n",
    "    train_encoded[['TotalIncome', 'Debt_Income_Ratio', 'LoanAmount']])\n",
    "test_encoded[['TotalIncome', 'Debt_Income_Ratio', 'LoanAmount']] = scaler.transform(\n",
    "    test_encoded[['TotalIncome', 'Debt_Income_Ratio', 'LoanAmount']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "76fe3fd1-7614-4f01-93b7-fee8d374b204",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Model Training and Evaluation ==========\n",
    "X = train_encoded\n",
    "y = train_df['Loan_Status']\n",
    "\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "adcf6e29-6104-421c-85ed-72fffc0202c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Performance:\n",
      "0.7967479674796748\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.49      0.63        43\n",
      "           1       0.78      0.96      0.86        80\n",
      "\n",
      "    accuracy                           0.80       123\n",
      "   macro avg       0.83      0.73      0.74       123\n",
      "weighted avg       0.81      0.80      0.78       123\n",
      "\n",
      "[[21 22]\n",
      " [ 3 77]]\n"
     ]
    }
   ],
   "source": [
    "# Random Forest\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "rf_preds = rf.predict(X_val)\n",
    "\n",
    "print(\"Random Forest Performance:\")\n",
    "print(accuracy_score(y_val, rf_preds))\n",
    "print(classification_report(y_val, rf_preds))\n",
    "print(confusion_matrix(y_val, rf_preds))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "90f15e91-9e15-4927-9ca2-0c671a78ccb4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Performance:\n",
      "0.7479674796747967\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.68      0.53      0.60        43\n",
      "           1       0.78      0.86      0.82        80\n",
      "\n",
      "    accuracy                           0.75       123\n",
      "   macro avg       0.73      0.70      0.71       123\n",
      "weighted avg       0.74      0.75      0.74       123\n",
      "\n",
      "[[23 20]\n",
      " [11 69]]\n"
     ]
    }
   ],
   "source": [
    "# XGBoost\n",
    "xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')\n",
    "xgb.fit(X_train, y_train)\n",
    "xgb_preds = xgb.predict(X_val)\n",
    "\n",
    "print(\"XGBoost Performance:\")\n",
    "print(accuracy_score(y_val, xgb_preds))\n",
    "print(classification_report(y_val, xgb_preds))\n",
    "print(confusion_matrix(y_val, xgb_preds))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "989c23cf-6cfd-470c-abc8-566eb115f749",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ======== Feature Engineering ========\n",
    "# Total Income\n",
    "train_encoded['TotalIncome'] = train_encoded['ApplicantIncome'] + train_encoded['CoapplicantIncome']\n",
    "test_encoded['TotalIncome'] = test_encoded['ApplicantIncome'] + test_encoded['CoapplicantIncome']\n",
    "\n",
    "# Debt-to-Income Ratio\n",
    "train_encoded['Debt_Income_Ratio'] = train_encoded['LoanAmount'] / train_encoded['TotalIncome']\n",
    "test_encoded['Debt_Income_Ratio'] = test_encoded['LoanAmount'] / test_encoded['TotalIncome']\n",
    "\n",
    "# EMI (Equated Monthly Installment)\n",
    "train_encoded['EMI'] = train_encoded['LoanAmount'] / train_encoded['Loan_Amount_Term']\n",
    "test_encoded['EMI'] = test_encoded['LoanAmount'] / test_encoded['Loan_Amount_Term']\n",
    "\n",
    "# Income Per Person\n",
    "train_encoded['Income_Per_Person'] = train_encoded['TotalIncome'] / (train_encoded['Dependents'] + 1)\n",
    "test_encoded['Income_Per_Person'] = test_encoded['TotalIncome'] / (test_encoded['Dependents'] + 1)\n",
    "\n",
    "# Loan-to-Income Ratio\n",
    "train_encoded['Loan_to_Income'] = train_encoded['LoanAmount'] / train_encoded['TotalIncome']\n",
    "test_encoded['Loan_to_Income'] = test_encoded['LoanAmount'] / test_encoded['TotalIncome']\n",
    "\n",
    "# Credit History Interaction\n",
    "train_encoded['Credit_Income'] = train_encoded['Credit_History'] * train_encoded['TotalIncome']\n",
    "test_encoded['Credit_Income'] = test_encoded['Credit_History'] * test_encoded['TotalIncome']\n",
    "\n",
    "# ======== Scaling ========\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "scaler = StandardScaler()\n",
    "scaled_features = ['TotalIncome', 'Debt_Income_Ratio', 'EMI', 'Income_Per_Person', 'Loan_to_Income', 'Credit_Income']\n",
    "train_encoded[scaled_features] = scaler.fit_transform(train_encoded[scaled_features])\n",
    "test_encoded[scaled_features] = scaler.transform(test_encoded[scaled_features])\n",
    "\n",
    "# ======== Train Model ========\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "# Train-Test Split\n",
    "X = train_encoded\n",
    "y = train_df['Loan_Status']\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Random Forest Classifier\n",
    "rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,\n",
    "                            min_samples_leaf=2, max_features='sqrt', random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "rf_preds = rf.predict(X_val)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Accuracy:\", accuracy_score(y_val, rf_preds))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_val, rf_preds))\n",
    "print(\"\\nConfusion Matrix:\\n\", confusion_matrix(y_val, rf_preds))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1f3ba77-36f2-469a-b4d6-cad8523a66c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ========== Prediction and Submission ==========\n",
    "test_df['Loan_Status'] = label_encoder.inverse_transform(xgb.predict(test_encoded))\n",
    "\n",
    "submission = test_df[['Loan_ID', 'Loan_Status']]\n",
    "submission.to_csv('submission.csv', index=False)\n",
    "submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc4c632e-bae3-4f90-893e-cd9c8cd6b018",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
