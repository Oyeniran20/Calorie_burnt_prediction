# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    Args:
        file_path (str): Path to the CSV dataset.
    Returns:
        train_inputs (DataFrame): Processed training inputs.
        test_inputs (DataFrame): Processed testing inputs.
        train_targets (Series): Training targets.
        test_targets (Series): Testing targets.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(columns=["User_ID"])
    
    # Encode categorical columns
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
    # IQR capping for non-normally distributed columns
    for col in ['Age', 'Body_Temp']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Z-score capping for normally distributed columns
    from scipy.stats import zscore
    for col in ['Height', 'Weight', 'Duration', 'Heart_Rate']:
        z_scores = zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        z_threshold = 3
        df[col] = np.where(abs_z_scores > z_threshold, df[col].mean() + z_threshold * df[col].std(), df[col])
        df[col] = np.where(abs_z_scores > z_threshold, df[col].mean() - z_threshold * df[col].std(), df[col])
    
    # Feature engineering
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    age_bins = [0, 30, 60, 80]
    age_labels = ['Young', 'Middle-aged', 'Old']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    df['Calories_sqrt'] = np.sqrt(df['Calories'])
    
    # Split data into features and target
    features = df.drop(columns=['Calories', 'Calories_sqrt'])
    target = df['Calories_sqrt']
    
    # One-hot encode categorical features
    ohe = OneHotEncoder(drop='first')
    age_group_encoded = ohe.fit_transform(features[['Age_Group']])
    age_group_encoded_df = pd.DataFrame(age_group_encoded, columns=ohe.get_feature_names_out(['Age_Group']))
    features = pd.concat([features.drop(columns=['Age_Group']), age_group_encoded_df], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Standard scaling
    scaler = StandardScaler()
    columns_to_scale = ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'BMI']
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    # PCA for correlated groups
    pca_physical = PCA(n_components=1)
    X_train['Physical_Factor'] = pca_physical.fit_transform(X_train[['Height', 'Weight']])
    X_test['Physical_Factor'] = pca_physical.transform(X_test[['Height', 'Weight']])
    
    pca_exertion = PCA(n_components=1)
    X_train['Exertion_Factor'] = pca_exertion.fit_transform(X_train[['Duration', 'Heart_Rate', 'Body_Temp']])
    X_test['Exertion_Factor'] = pca_exertion.transform(X_test[['Duration', 'Heart_Rate', 'Body_Temp']])
    
    # Drop original columns used for PCA
    X_train = X_train.drop(columns=['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    X_test = X_test.drop(columns=['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the CatBoost model.
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training targets.
    Returns:
        model (CatBoostRegressor): Trained model.
    """
    model = RandomForestRegressor(random_seed=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    Args:
        model (CatBoostRegressor): Trained model.
        X_test (DataFrame): Testing features.
        y_test (Series): Testing targets.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")

def save_model(model, file_name):
    """
    Save the trained model to a file.
    Args:
        model (CatBoostRegressor): Trained model.
        file_name (str): File name to save the model.
    """
    with open(file_name, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_name}")

if __name__ == "__main__":
    # File paths
    dataset_path = "Exercise.csv"  # Replace with your dataset file path
    model_file_name = "model.pkl"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model, model_file_name)
