import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create sample salary dataset
np.random.seed(42)

# Generate sample data
n_samples = 1000

# Features
years_experience = np.random.uniform(0, 20, n_samples)
education_level = np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.5, 0.35, 0.15])
job_title = np.random.choice(['Software Engineer', 'Data Scientist', 'Product Manager', 'Designer'], n_samples)
company_size = np.random.choice(['Small', 'Medium', 'Large'], n_samples)
location = np.random.choice(['New York', 'San Francisco', 'Seattle', 'Austin', 'Remote'], n_samples)

# Create salary based on features (with some realistic correlations)
base_salary = 50000
experience_bonus = years_experience * 3000
education_bonus = {'Bachelor': 0, 'Master': 15000, 'PhD': 30000}
job_bonus = {'Software Engineer': 20000, 'Data Scientist': 25000, 'Product Manager': 30000, 'Designer': 15000}
size_bonus = {'Small': 0, 'Medium': 10000, 'Large': 20000}
location_bonus = {'New York': 15000, 'San Francisco': 25000, 'Seattle': 20000, 'Austin': 10000, 'Remote': 5000}

salary = (base_salary + 
          experience_bonus + 
          [education_bonus[ed] for ed in education_level] + 
          [job_bonus[job] for job in job_title] + 
          [size_bonus[size] for size in company_size] + 
          [location_bonus[loc] for loc in location] + 
          np.random.normal(0, 5000, n_samples))  # Add some noise

# Create DataFrame
df = pd.DataFrame({
    'years_experience': years_experience,
    'education_level': education_level,
    'job_title': job_title,
    'company_size': company_size,
    'location': location,
    'salary': np.maximum(salary, 40000)  # Ensure minimum salary
})

# Create data directory if it doesn't exist
import os
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/salary_data.csv', index=False)
print("Sample data created successfully!")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Salary range: ${df['salary'].min():.0f} - ${df['salary'].max():.0f}") 