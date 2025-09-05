import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
decisiontree_model=joblib.load("decisiontree_model.joblib")
linear_regression_model=joblib.load("linearregression_model.joblib")
randomforest_model=joblib.load("randomforest_model.joblib")
svr_model=joblib.load("svr_model.joblib")
st.title("Student Exam Score Predictor")
Hours_Studied=st.number_input('Hours Studied Per Week',min_value=3.0,max_value=40.0,value=3.0)
Attendance =st.number_input('Attendance Percentage of Classes',min_value=50.0,max_value=100.0,value=50.0)
Parental_Involvement=st.selectbox("Parental Involvement",["Low","Medium","High"])
Access_to_Resources=st.selectbox("Access to Resources",["Low","Medium","High"])
Extracurricular_Activities=st.selectbox("Extracurricular Activities",["Yes","No"])
Sleep_Hours =st.number_input('Average Sleep Hours Per Night',min_value=3.0,max_value=12.0,value=3.0)
Previous_Scores =st.number_input('Previous Scores',min_value=40.0,max_value=100.0,value=40.0)
Motivation_Level =st.selectbox("Motivation Level",["Low","Medium","High"])
Internet_Access =st.selectbox("Internet Access",["Yes","No"])
Tutoring_Sessions  =st.number_input('Tutoring Sessions Attended Per Month',min_value=0,max_value=5,value=0)
Family_Income =st.selectbox("Family Income",["Low","Medium","High"])
Teacher_Quality =st.selectbox("Teacher Quality",["Low","Medium","High"])
School_Type  =st.selectbox("School Type",['Public', 'Private'])
Peer_Influence =st.selectbox("Peer Influence",['Positive','Negative','Neutral'])
Physical_Activity=st.number_input('Average Number of hours of physical activity per week.',min_value=0.0,max_value=8.0,value=0.0)
Learning_Disabilities=st.selectbox("Learning Disabilities",["Yes","No"])
Parental_Education_Level=st.selectbox("Parental Education Level",['High School','College','Postgraduate'])
Distance_from_Home =st.selectbox("Distance from Home ",['Near','Moderate','Far'])
Gender  =st.selectbox("Gender",['Male','Female'])



PI_encoded=0
if Parental_Involvement=="Low":
    PI_encoded=1
elif Parental_Involvement=="Medium":
    PI_encoded=2

ATR_encoded=0
if Access_to_Resources=="Low":
    ATR_encoded=1
elif Access_to_Resources=="Medium":
    ATR_encoded=2

EA_encoded=0
if Extracurricular_Activities=="Yes":
    EA_encoded=1


ML_encoded=0
if  Motivation_Level=="Low":
    ML_encoded=1
elif Motivation_Level=="Medium":
    ML_encoded=2


IA_encoded=0
if Internet_Access=="Yes":
    IA_encoded=1



FA_encoded=0
if  Family_Income=="Low":
    FA_encoded=1
elif Family_Income=="Medium":
    FA_encoded=2

TQ_encoded=0
if  Teacher_Quality=="Low":
    TQ_encoded=1
elif Teacher_Quality=="Medium":
    TQ_encoded=2

ST_encoded=0
if  School_Type=="Public":
    ST_encoded=1

PEI_encoded=0
if Peer_Influence=="Neutral":
    PEI_encoded=1
elif Peer_Influence=="Positive":
    PEI_encoded=2


LD_encoded=0
if Learning_Disabilities=="Yes":
    LD_encoded=1

PEL_encoded=0
if Parental_Education_Level=="High School":
    PEL_encoded=1
elif Parental_Education_Level=="Postgraduate":
    PEL_encoded=2



DFH_encoded=0
if Distance_from_Home=="Moderate":
    DFH_encoded=1
elif Distance_from_Home=="Near":
    DFH_encoded=2

gender_encoded=0
if Gender=="Male":
    gender_encoded=1

if st.button("Predict Exam Score"):
    input_data = np.array([[Hours_Studied, Attendance, PI_encoded, ATR_encoded, EA_encoded, Sleep_Hours, Previous_Scores, ML_encoded, IA_encoded, Tutoring_Sessions, FA_encoded, TQ_encoded, ST_encoded, PEI_encoded, Physical_Activity, LD_encoded, PEL_encoded, DFH_encoded, gender_encoded]])
    lr_pred = linear_regression_model.predict(input_data)
    dt_pred = decisiontree_model.predict(input_data)
    rf_pred = randomforest_model.predict(input_data)
    svr_pred = svr_model.predict(input_data)
    lr_pred = max(0, min(100, lr_pred[0]))
    dt_pred = max(0, min(100, dt_pred[0]))
    rf_pred = max(0, min(100, rf_pred[0]))
    svr_pred = max(0, min(100, svr_pred[0]))
    st.success(f'Linear Regression Predicted Exam Score: {lr_pred:.2f}')
    st.success(f'Decision Tree Predicted Exam Score: {dt_pred:.2f}')
    st.success(f'Random Forest Predicted Exam Score: {rf_pred:.2f}')
    st.success(f'SVR Predicted Exam Score: {svr_pred:.2f}')



