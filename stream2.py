import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import joblib
from acb import ccsc,aprc,aprdrgc
import matplotlib.pyplot as plt
loded_model=joblib.load('lr.pkl')
dt=pd.read_csv('Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv')
age_string_index = {'0 to 17': 1, '18 to 29': 2, '30 to 49': 3, '50 to 69': 4, '70 or Older': 5}
gender_string_index = {'F':1,'M':2,'U':3}
admission_string_index = {'Emergency': 1, 'Urgent': 2, 'Elective': 3, 'Not Available': 4, 'Trauma': 5,'Newborn':6}
mortality_string_index = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
illness_string_index={'Minor':1,'Moderate':2,'Major':3,'Extreme':4}
# Impute missing values in 'APR Risk of Mortality' column based on the proportion of occurrence
facility_id_counts = dt['APR Risk of Mortality'].value_counts(normalize=True)
missing_indices = dt['APR Risk of Mortality'].isnull()
dt.loc[missing_indices, 'APR Risk of Mortality'] = np.random.choice(facility_id_counts.index, size=missing_indices.sum(), p=facility_id_counts.values)
# Impute missing values in 'Facility Id' column based on the proportion of occurrence
facility_id_counts = dt['Facility Id'].value_counts(normalize=True)
missing_indices = dt['Facility Id'].isnull()
dt.loc[missing_indices, 'Facility Id'] = np.random.choice(facility_id_counts.index, size=missing_indices.sum(), p=facility_id_counts.values)





class streamlit:
    def __init__(self):
        self.model=loded_model
        self.CCS_code_str=None
        self.age_group=None
        self.APR_code_str=None
        self.MDC_code_str=None
    '''def train_data(self):
        self.model.fit(x_train,y_train)
        return self.model'''
    def construct_slider(self):
        st.sidebar.markdown('<p style="font-size: 25px; color: black; font-weight: bold;" class="header-style">Length Of Stay</p>', unsafe_allow_html=True)
        facility_id = st.sidebar.selectbox(
            f"**Select {'Facility Id'}**",
            sorted(dt['Facility Id'].unique())
        )
        age_group=st.sidebar.selectbox(
            f"**Select {'Age Group'}**",
            sorted(dt['Age Group'].unique())
        )
        self.age_group=age_group
        age_group = age_string_index[age_group]
        gender_str=st.sidebar.selectbox(
            f"**Select {'Gender'}**",
            sorted(dt['Gender'].unique())
        )
        gender = gender_string_index[gender_str]
        type_of_admission_str=st.sidebar.selectbox(
            f"**Select {'Type of Admission'}**",
            sorted(dt['Type of Admission'].unique())
        )
        type_of_admission = admission_string_index[type_of_admission_str]
        CCS_code_str=st.sidebar.selectbox(
            f"**Select {'CCS Diagnosis Description'}**",
            sorted(dt['CCS Diagnosis Description'].unique())
        )
        self.CCS_code_str=CCS_code_str
        CCS_code=ccsc[CCS_code_str]
        APR_code_str=st.sidebar.selectbox(
            f"**Select {'APR DRG Description'}**",
            sorted(dt['APR DRG Description'].unique())
        )
        self.APR_code_str=APR_code_str
        APR_code=aprdrgc[APR_code_str]
        MDC_code_str=st.sidebar.selectbox(
            f"**Select {'APR MDC Description'}**",
            sorted(dt['APR MDC Description'].unique())
        )
        self.MDC_code_str=MDC_code_str
        MDC_code=aprc[MDC_code_str]
        illness_code_str=st.sidebar.selectbox(
            f"**Select {'APR Severity of Illness Description'}**",
            sorted(dt['APR Severity of Illness Description'].astype(str).unique())
        )
        illness_code=illness_string_index[illness_code_str]
        mortality_code_str=st.sidebar.selectbox(
            f"**Select {'APR Risk of Mortality'}**",
            sorted(dt['APR Risk of Mortality'].astype(str).unique())
        )
        mortality_code = mortality_string_index[mortality_code_str]
        total_charges_input = st.sidebar.text_input("**Enter Total Charges**", value="0")

        # Convert the input to float
        total_charges = float(total_charges_input)

        values=[facility_id,age_group,gender,type_of_admission,CCS_code,APR_code,MDC_code,illness_code,mortality_code,total_charges]
        return values
    
    def construct_app(self):
        #self.train_data()
        values=self.construct_slider()
        prediction=self.model.predict([values])
        page_bg_img = """
        <style>
        [class="main st-emotion-cache-uf99v8 ea3mdgi8"]{
           background-color: #e5e5f7;
opacity: 1;
background-image:  radial-gradient(#444cf7 1.4500000000000002px, transparent 1.4500000000000002px), radial-gradient(#444cf7 1.4500000000000002px, #e5e5f7 1.4500000000000002px);
background-size: 58px 58px;
background-position: 0 0,29px 29px;
        }
        </style>"""
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.markdown(
    '<div style="text-align: center;"><p class="header-style" style="font-size: 40px;color: black; font-weight: bold;">Inpatient Length Of Stay</p></div>',
    unsafe_allow_html=True
)
        column_1= st.columns(1)
        with column_1[0]:
            st.markdown(
                f'<p class="font-style" style="font-size: 30px;color: black; font-weight: bold;">Prediction </p>',
                unsafe_allow_html=True
            )
            st.markdown(
    f'<span style="color:black; font-size: 20px; font-weight: bold;">{str(round(prediction[0], 1))} Days</span>',
    unsafe_allow_html=True
)

        

# Display the HTML/CSS code using Markdown

        
        #Plot1
        st.markdown(
    '<div style="text-align: center;"><p class="header-style" style="font-size: 20px;color: black; font-weight: bold;">CCS Diagnosis Description</p></div>',
    unsafe_allow_html=True
)
        filtered_data1 = dt.copy()
        filtered_data1 = filtered_data1[filtered_data1['CCS Diagnosis Description'] == self.CCS_code_str]
        filtered_data1 = filtered_data1[filtered_data1['Age Group'] == self.age_group]

        # Check if filtered data is empty
        if filtered_data1.empty:
            st.write(f"No data available for {self.CCS_code_str} ({self.age_group})")
        else:
        # Convert 'Length of Stay' column to numeric data type, coerce errors to NaN
            filtered_data1['Length of Stay'] = pd.to_numeric(filtered_data1['Length of Stay'], errors='coerce')

            # Drop rows with NaN values in 'Length of Stay' column
            filtered_data1 = filtered_data1.dropna(subset=['Length of Stay'])

            # Calculate average length of stay by gender
            avg_length_of_stay_by_gender = filtered_data1.groupby('Gender')['Length of Stay'].mean()

            # Plot bar chart with data labels
            fig, ax = plt.subplots(figsize=(8,7))
            bars = avg_length_of_stay_by_gender.plot(kind='bar', ax=ax,color='lightblue')
            ax.set_xlabel('Gender',fontsize=22)
            ax.set_ylabel('Average Length of Stay',fontsize=22)
            ax.set_title(f'Average Length of Stay by Gender for {self.CCS_code_str} ({self.age_group})',fontsize=22)

            # Add data labels to each bar
            for bar in bars.patches:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom',fontsize=22)

            st.pyplot(fig)
        

            #Plot2
        st.markdown(
    '<div style="text-align: center;"><p class="header-style" style="font-size: 20px;color: black; font-weight: bold;">APR DRG Description</p></div>',
    unsafe_allow_html=True
)
        filtered_data2 = dt.copy()
        filtered_data2 = filtered_data2[filtered_data2['APR DRG Description'] == self.APR_code_str]
        filtered_data2 = filtered_data2[filtered_data2['Age Group'] == self.age_group]

        # Check if filtered data is empty
        if filtered_data2.empty:
            st.write(f"No data available for {self.APR_code_str} ({self.age_group})")
        else:
        # Convert 'Length of Stay' column to numeric data type, coerce errors to NaN
            filtered_data2['Length of Stay'] = pd.to_numeric(filtered_data2['Length of Stay'], errors='coerce')

            # Drop rows with NaN values in 'Length of Stay' column
            filtered_data2 = filtered_data2.dropna(subset=['Length of Stay'])

            # Calculate average length of stay by gender
            avg_length_of_stay_by_gender = filtered_data2.groupby('Gender')['Length of Stay'].mean()

            # Plot bar chart with data labels
            fig, ax = plt.subplots(figsize=(8,7))
            bars = avg_length_of_stay_by_gender.plot(kind='bar', ax=ax,color='lightblue')
            ax.set_xlabel('Gender',fontsize=22)
            ax.set_ylabel('Average Length of Stay',fontsize=22)
            ax.set_title(f'Average Length of Stay by Gender for {self.APR_code_str} ({self.age_group})',fontsize=22)

                # Add data labels to each bar
            for bar in bars.patches:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom',fontsize=22)

            st.pyplot(fig)
        


        #Plot3
        st.markdown(
    '<div style="text-align: center;"><p class="header-style" style="font-size: 20px;color: black; font-weight: bold;">APR MDC Description</p></div>',
    unsafe_allow_html=True
)
        filtered_data3 = dt.copy()
        filtered_data3 = filtered_data3[filtered_data3['APR MDC Description'] == self.MDC_code_str]
        filtered_data3 = filtered_data3[filtered_data3['Age Group'] == self.age_group]

        # Check if filtered data is empty
        if filtered_data3.empty:
            st.write(f"No data available for {self.MDC_code_str} ({self.age_group})")
        else:
        # Convert 'Length of Stay' column to numeric data type, coerce errors to NaN
            filtered_data3['Length of Stay'] = pd.to_numeric(filtered_data3['Length of Stay'], errors='coerce')

            # Drop rows with NaN values in 'Length of Stay' column
            filtered_data3 = filtered_data3.dropna(subset=['Length of Stay'])

            # Calculate average length of stay by gender
            avg_length_of_stay_by_gender = filtered_data3.groupby('Gender')['Length of Stay'].mean()

                # Plot bar chart with data labels
            fig, ax = plt.subplots(figsize=(8,7))
            bars = avg_length_of_stay_by_gender.plot(kind='bar', ax=ax,color='lightblue')
            ax.set_xlabel('Gender',fontsize=22)
            ax.set_ylabel('Average Length of Stay',fontsize=22)
            ax.set_title(f'Average Length of Stay by Gender for {self.MDC_code_str} ({self.age_group})',fontsize=22)

            # Add data labels to each bar
            for bar in bars.patches:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom',fontsize=22)

            st.pyplot(fig)
        

        return self


sa=streamlit()
sa.construct_app()
