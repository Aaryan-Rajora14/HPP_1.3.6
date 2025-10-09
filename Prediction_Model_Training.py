import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_csv('Housing.csv')
    
    # Convert yes/no columns to 1/0
    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea']
    
    for col in binary_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Encode furnishingstatus
    le = LabelEncoder()
    df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])
    
    return df

# Feature engineering
def feature_engineering(df):
    # Create new features
    df['area_per_bedroom'] = df['area'] / df['bedrooms']
    df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['has_parking'] = (df['parking'] > 0).astype(int)
    
    return df

# Build and tune gradient boosting model
def build_gradient_boosting_model(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        gb, param_grid, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return best_model, y_train_pred, y_test_pred

# Main execution
def main():
    # Load and prepare data
    df = load_and_prepare_data()
    df = feature_engineering(df)
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Dataset Shape:", df.shape)
    print("Features:", X.columns.tolist())
    
    # Build model
    model, y_train_pred, y_test_pred = build_gradient_boosting_model(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Save model and scaler
    with open('HPP_Model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print("\nModel saved as 'HPP_Model.pkl'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()