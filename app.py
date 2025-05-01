import pandas as pd
import streamlit as st
import pickle

st.set_page_config(layout="wide")

df = pd.read_csv("Symptom-severity.csv")
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")

df["Symptom"] = df["Symptom"].str.replace("_", " ").str.strip()
desc_df["Disease"] = desc_df["Disease"].str.replace("_", " ").str.strip()
prec_df["Disease"] = prec_df["Disease"].str.replace("_", " ").str.strip()

symptom_to_weight = dict(zip(df["Symptom"], df["weight"]))

all_symptoms = sorted(symptom_to_weight.keys())

model = pickle.load(open("model.pkl", "rb"))


st.title("🩺 Disease Prediction Application 🚑")

st.markdown("### 💡 About this Application")
st.markdown("##### Welcome to the Disease Prediction App, an intelligent tool designed to help predict possible diseases based on the symptoms you experience." 
" This application leverages a machine learning model trained with one hundred and thirty two (132) medical symptoms to provide real-time health insights.")

st.markdown("### ❗ Important Notes\n")
st.markdown(''' 
    <ul>
        <li> This tool is designed for educational and informational purposes only.</li> 
        <li>It is <b>not a subsitute for professional medical advice.</b></li>
        <li>Predictions are based solely on symptom severity and may not consider other factors such as age, history, or coexisting conditions.</li>
        <li>Always <b>consult a healthcare provider for diagnosis and treatment decisions.</b></li>
    </ul>        
            ''', 
unsafe_allow_html=True)

selected_symptoms = st.multiselect("Enter the Symptoms", all_symptoms)

if st.button("Predict"):
    if len(selected_symptoms) > 17:
        st.error("Please Select **Not More than 17 Symptoms**.")
        st.stop()
    
    input_vector = [symptom_to_weight.get(symptom,0) for symptom in selected_symptoms]

    input_vector += [0] * (17 - len(input_vector))

    prediction = model.predict([input_vector])

    st.success(f"### 🚨 Predicted Diseases: **{prediction[0]}**")

    description = desc_df.loc[desc_df["Disease"] == prediction[0], "Description"].values

    if description.size > 0:
        st.markdown("### ⚕️ Disease Description")
        st.info(f"#### {description[0]}")
    
    precautions = prec_df.loc[prec_df["Disease"] == prediction[0]]
    if not precautions.empty:
        st.markdown("### ⚠️ Recommended Precautions")
        for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            if pd.notna(precautions.iloc[0][col]):
                st.markdown(f"#### 🔹 {precautions.iloc[0][col]}")



