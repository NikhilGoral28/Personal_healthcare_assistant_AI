
import pickle

import numpy as np
import pandas as pd

import google.generativeai as genai


'''

# Load the model and data
model = pickle.load(open('E:\Personal_healthcare_assistant_AI\Personal_healthcare_assistant_AI\models/model.pkl', 'rb'))

label_encoder = pickle.load(open('E:\Personal_healthcare_assistant_AI\Personal_healthcare_assistant_AI\models/label_encoder.pkl', 'rb'))
symptoms_index = pickle.load(open('E:\Personal_healthcare_assistant_AI\Personal_healthcare_assistant_AI\models/symptom_index.pkl', 'rb'))


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_index))
    for item in patient_symptoms:
        input_vector[symptoms_index[item]] = 1
    
    prediction = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    return predicted_disease
'''


import openai

openai.api_key = "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"
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful healthcare assistant."},
            {"role": "user", "content": "I have cough, cold, and fever. What disease could it be?"}
        ],
        temperature=0.5,
        max_tokens=50,
    )

    print(response['choices'][0]['message']['content'])
except Exception as e:
    print(f"Error: {e}")


'''
if __name__ == "__main__":
    patient_symptoms = ['vomiting']
    pre_disease = get_predicted_value(patient_symptoms)
    print("\n=== Prediction Result ===")
    print(f"Input Symptoms: {patient_symptoms}")
    print(f"Predicted Disease: {pre_disease}")
    '''