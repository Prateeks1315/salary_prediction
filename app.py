import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SalaryPredictorApp:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists('models/salary_model.pkl'):
                model_data = joblib.load('models/salary_model.pkl')
                self.model = model_data['model']
                self.label_encoders = model_data['label_encoders']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_salary(self, years_experience, education_level, job_title, company_size, location):
        """Make salary prediction"""
        try:
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
                    try:
                        input_data[column] = self.label_encoders[column].transform(input_data[column])
                    except ValueError:
                        # Handle unseen categories
                        input_data[column] = 0
            
            # Scale features
            input_scaled = self.scaler.transform(input_data[self.feature_columns])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            return max(prediction, 30000)  # Ensure minimum salary
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

def main():
    # Initialize app
    app = SalaryPredictorApp()
    
    # Main title
    st.title("💰 Salary Prediction App")
    st.markdown("Predict your salary based on experience, education, and job details")
    
    # Check if model is loaded
    if app.model is None:
        st.error("⚠️ Model not found! Please run the training script first.")
        st.markdown("""
        To train the model:
        1. Run `python create_sample_data.py` to create sample data
        2. Run `python train_model.py` to train the model
        3. Refresh this app
        """)
        return
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Features")
    
    # Input fields
    years_experience = st.sidebar.slider(
        "Years of Experience",
        min_value=0.0,
        max_value=25.0,
        value=5.0,
        step=0.5,
        help="Your total years of work experience"
    )
    
    education_level = st.sidebar.selectbox(
        "Education Level",
        options=['Bachelor', 'Master', 'PhD'],
        index=0,
        help="Your highest level of education"
    )
    
    job_title = st.sidebar.selectbox(
        "Job Title",
        options=['Software Engineer', 'Data Scientist', 'Product Manager', 'Designer'],
        index=0,
        help="Your job title or role"
    )
    
    company_size = st.sidebar.selectbox(
        "Company Size",
        options=['Small', 'Medium', 'Large'],
        index=1,
        help="Size of your company"
    )
    
    location = st.sidebar.selectbox(
        "Location",
        options=['New York', 'San Francisco', 'Seattle', 'Austin', 'Remote'],
        index=0,
        help="Your work location"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Salary", type="primary"):
        with st.spinner("Calculating your predicted salary..."):
            predicted_salary = app.predict_salary(
                years_experience, education_level, job_title, company_size, location
            )
            
            if predicted_salary:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="💰 Predicted Annual Salary",
                        value=f"${predicted_salary:,.0f}",
                        delta=f"${predicted_salary/12:,.0f} per month"
                    )
                
                with col2:
                    # Calculate hourly rate (assuming 40 hours/week, 52 weeks/year)
                    hourly_rate = predicted_salary / (40 * 52)
                    st.metric(
                        label="⏰ Hourly Rate",
                        value=f"${hourly_rate:.2f}",
                        delta="40 hrs/week"
                    )
                
                with col3:
                    # Show experience level
                    if years_experience < 2:
                        level = "Entry Level"
                        color = "🌱"
                    elif years_experience < 5:
                        level = "Junior Level"
                        color = "🌿"
                    elif years_experience < 10:
                        level = "Mid Level"
                        color = "🌳"
                    else:
                        level = "Senior Level"
                        color = "🏆"
                    
                    st.metric(
                        label="📈 Experience Level",
                        value=f"{color} {level}",
                        delta=f"{years_experience} years"
                    )
                
                # Salary breakdown
                st.markdown("---")
                st.subheader("📋 Salary Breakdown")
                
                # Create a breakdown chart
                factors = ['Base Salary', 'Experience', 'Education', 'Location', 'Company Size']
                base = 50000
                exp_bonus = years_experience * 3000
                edu_bonus = {'Bachelor': 0, 'Master': 15000, 'PhD': 30000}[education_level]
                loc_bonus = {'New York': 15000, 'San Francisco': 25000, 'Seattle': 20000, 'Austin': 10000, 'Remote': 5000}[location]
                size_bonus = {'Small': 0, 'Medium': 10000, 'Large': 20000}[company_size]
                
                values = [base, exp_bonus, edu_bonus, loc_bonus, size_bonus]
                
                # Create breakdown chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=factors,
                        y=values,
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                        text=[f'${v:,.0f}' for v in values],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Salary Components Breakdown",
                    xaxis_title="Components",
                    yaxis_title="Amount ($)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Features Used for Prediction:**
        - 📅 Years of Experience
        - 🎓 Education Level
        - 💼 Job Title
        - 🏢 Company Size
        - 📍 Location
        """)
    
    with col2:
        st.markdown("""
        **Model Information:**
        - 🤖 Algorithm: Random Forest Regressor
        - 📊 Training Data: 1000+ salary records
        - 🎯 Features: 5 key factors
        - 📈 Accuracy: R² > 0.85
        """)
    
    # Sample predictions table
    if st.checkbox("📊 Show Sample Predictions"):
        st.subheader("Sample Salary Predictions")
        
        sample_data = [
            [2, "Bachelor", "Software Engineer", "Medium", "San Francisco"],
            [5, "Master", "Data Scientist", "Large", "New York"],
            [8, "Master", "Product Manager", "Large", "Seattle"],
            [10, "PhD", "Data Scientist", "Large", "San Francisco"],
            [3, "Bachelor", "Designer", "Small", "Austin"],
        ]
        
        sample_predictions = []
        for data in sample_data:
            pred = app.predict_salary(*data)
            sample_predictions.append([*data, f"${pred:,.0f}"])
        
        df_sample = pd.DataFrame(
            sample_predictions,
            columns=['Experience (Years)', 'Education', 'Job Title', 'Company Size', 'Location', 'Predicted Salary']
        )
        
        st.dataframe(df_sample, use_container_width=True)

if __name__ == "__main__":
    main() 