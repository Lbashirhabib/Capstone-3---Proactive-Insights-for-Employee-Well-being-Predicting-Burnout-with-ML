import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load pre-trained model (replace with your actual model)
# model = joblib.load('burnout_model.pkl')

# Mock model for demonstration
class MockModel:
    def predict(self, X):
        return np.clip(0.2*X['Mental Fatigue Score'] + 0.15*X['Resource Allocation'] + 
                      np.random.normal(0, 0.05, len(X)), 0, 1)

model = MockModel()

# App title and description
st.title('NeuroWell Analytics - Employee Burnout Prediction')
st.markdown("""
Predict and prevent employee burnout using our AI-powered tool.
""")

# Navigation sidebar
page = st.sidebar.selectbox("Choose a page", 
                           ["Home", "Data Insights", "Burnout Predictor", "Prevention Tips"])

if page == "Home":
    st.header("About This Tool")
    st.image("https://images.unsplash.com/photo-1521791136064-7986c2920216", width=600)
    st.write("""
    This application helps HR teams:
    - Identify employees at risk of burnout
    - Understand key burnout drivers
    - Implement preventive measures
    
    **How it works:**
    1. Explore our data insights
    2. Predict burnout for individual employees
    3. Get customized recommendations
    """)

elif page == "Data Insights":
    st.header("Burnout Data Analysis")
    
    # Generate sample data for visualization
    np.random.seed(42)
    data = pd.DataFrame({
        'Burn Rate': np.random.beta(2, 5, 1000),
        'Mental Fatigue Score': np.random.randint(1, 10, 1000),
        'Resource Allocation': np.random.uniform(4, 12, 1000),
        'WFH Setup': np.random.choice(['Yes', 'No'], 1000)
    })
    
    # Burn Rate Distribution
    st.subheader("Burnout Rate Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(data['Burn Rate'], bins=20, kde=True, color='salmon')
    plt.xlabel('Burn Rate')
    st.pyplot(fig1)
    
    # Key Factors
    st.subheader("Top Burnout Drivers")
    factors = pd.DataFrame({
        'Factor': ['Mental Fatigue', 'Work Hours', 'WFH Availability', 'Tenure'],
        'Impact': [38, 29, 15, 18]
    })
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Impact', y='Factor', data=factors, palette='Reds_r')
    plt.title('Feature Importance (%)')
    st.pyplot(fig2)

elif page == "Burnout Predictor":
    st.header("Employee Burnout Risk Assessment")
    
    with st.form("employee_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["Male", "Female", "Other"])
            company_type = st.selectbox("Company Type", ["Service", "Product"])
            wfh = st.radio("WFH Setup Available", ["Yes", "No"])
            
        with col2:
            designation = st.slider("Designation Level", 1, 10, 5)
            hours = st.slider("Daily Work Hours", 4, 12, 8)
            fatigue = st.slider("Mental Fatigue Score (1-10)", 1, 10, 5)
        
        submitted = st.form_submit_button("Predict Burnout Risk")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'Gender': [1 if gender == "Male" else 0],
                'Company Type': [1 if company_type == "Service" else 0],
                'WFH Setup Available': [1 if wfh == "Yes" else 0],
                'Designation': [designation],
                'Resource Allocation': [hours],
                'Mental Fatigue Score': [fatigue]
            })
            
            # Make prediction
            burnout_risk = model.predict(input_data)[0]
            risk_percent = int(burnout_risk * 100)
            
            # Display result
            st.subheader("Prediction Result")
            st.metric("Burnout Risk Score", f"{risk_percent}%")
            
            # Risk level interpretation
            if burnout_risk < 0.3:
                st.success("Low Risk - Employee is doing well!")
            elif burnout_risk < 0.6:
                st.warning("Moderate Risk - Monitor and check-in")
            else:
                st.error("High Risk - Immediate intervention recommended")

elif page == "Prevention Tips":
    st.header("Burnout Prevention Strategies")
    
    st.subheader("For High-Risk Employees")
    st.write("""
    - **Workload Adjustment:** Reduce hours to ≤8/day
    - **Mandatory Breaks:** 10 minutes every 2 hours
    - **Mental Health Support:** Company-paid counseling
    """)
    
    st.subheader("For All Employees")
    st.write("""
    - **Flexible Work:** Minimum 2 WFH days/week
    - **Wellness Programs:** Meditation sessions
    - **Clear Boundaries:** No after-hours emails
    """)
    
    st.subheader("For Managers")
    st.write("""
    - **Training:** Burnout recognition workshops
    - **1:1 Check-ins:** Monthly well-being meetings
    - **Team Culture:** Encourage vacation usage
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**NeuroWell Analytics**  
Employee Well-being Solutions  
Habib Bashir Lawal""")