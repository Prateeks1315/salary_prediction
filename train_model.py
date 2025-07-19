import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class SalaryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = ['years_experience', 'education_level', 'job_title', 'company_size', 'location']
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['education_level', 'job_title', 'company_size', 'location']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            data[column] = self.label_encoders[column].fit_transform(data[column])
        
        return data
    
    def train(self, df):
        """Train the salary prediction model"""
        # Prepare data
        data = self.prepare_data(df)
        
        # Split features and target
        X = data[self.feature_columns]
        y = data['salary']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Mean Squared Error: ${mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
    
    def predict(self, years_experience, education_level, job_title, company_size, location):
        """Make salary prediction for new data"""
        # Create DataFrame with input
        input_data = pd.DataFrame({
            'years_experience': [years_experience],
            'education_level': [education_level],
            'job_title': [job_title],
            'company_size': [company_size],
            'location': [location]
        })
        
        # Encode categorical variables
        categorical_columns = ['education_level', 'job_title', 'company_size', 'location']
        
        for column in categorical_columns:
            if column in self.label_encoders:
                # Handle unseen categories
                try:
                    input_data[column] = self.label_encoders[column].transform(input_data[column])
                except ValueError:
                    # If category not seen during training, use most frequent category
                    input_data[column] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_data[self.feature_columns])
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        return prediction
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the trained model and encoders"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

def main():
    """Train and save the salary prediction model"""
    # Load data
    df = pd.read_csv('data/salary_data.csv')
    print("Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Initialize and train model
    predictor = SalaryPredictor()
    metrics = predictor.train(df)
    
    # Save model
    predictor.save_model('models/salary_model.pkl')
    
    # Test prediction
    test_prediction = predictor.predict(5, 'Bachelor', 'Software Engineer', 'Medium', 'San Francisco')
    print(f"\nTest prediction: ${test_prediction:.2f}")

if __name__ == "__main__":
    main()
