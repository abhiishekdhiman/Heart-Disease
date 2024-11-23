#%%writefile app_pred.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Dieases Prediction", layout="wide")

df = pd.read_csv('dataset.csv')
y = df['target']
x = df.drop(['target'], axis=1)
model = LogisticRegression()
model.fit(x,y)

tab1, tab2, = st.tabs(["Info", "Model"])
tab1.title("About Website")
tab1.image("main1.png")
tab1.divider()
tab1.header(":page_facing_up: Dataset")
tab1.write(df)
tab1.write(
  "- :link: Kaggle Link  `https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset`"
)
tab1.divider()
tab1.header(":scroll: Variables or features explanations:")
tab1.markdown("""
- age (Age in years)
- sex : (1 = male, 0 = female)
- cp (Chest Pain Type): [ 0: asymptomatic, 1: atypical angina, 2: non-anginal pain, 3: typical angina]
- trestbps (Resting Blood Pressure in mm/hg )
- chol (Serum Cholesterol in mg/dl)
- fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]
- restecg (Resting ECG): [0: showing probable or definite left ventricular hypertrophy by Estes’ criteria, 1: normal, 2: having ST-T wave abnormality]
- thalach (maximum heart rate achieved)
- exang (Exercise Induced Angina): [1 = yes, 0 = no]
- oldpeak (ST depression induced by exercise relative to rest)
- slope (the slope of the peak exercise ST segment): [0: downsloping; 1: flat; 2: upsloping]
- ca [number of major vessels (0–3)
- thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]
- target: [0 = disease, 1 = no disease]
""")
tab1.divider()
tab1.header("Why these parameters: :question:")
tab1.markdown("""
- **Age:** Age is the most important risk factor in developing cardiovascular or heart diseases, with approximately a tripling of risk with each decade of life. Coronary fatty streaks can begin to form in adolescence. It is estimated that 82 percent of people who die of coronary heart disease are 65 and older. Simultaneously, the risk of stroke doubles every decade after age 55.
- **Sex:** Men are at greater risk of heart disease than pre-menopausal women. Once past menopause, it has been argued that a woman’s risk is similar to a man’s although more recent data from the WHO and UN disputes this. If a female has diabetes, she is more likely to develop heart disease than a male with diabetes.
- **Angina (Chest Pain):** Angina is chest pain or discomfort caused when your heart muscle doesn’t get enough oxygen-rich blood. It may feel like pressure or squeezing in your chest. The discomfort also can occur in your shoulders, arms, neck, jaw, or back. Angina pain may even feel like indigestion.
- **Resting Blood Pressure:** Over time, high blood pressure can damage arteries that feed your heart. High blood pressure that occurs with other conditions, such as obesity, high cholesterol or diabetes, increases your risk even more.
- **Serum Cholesterol:** A high level of low-density lipoprotein (LDL) cholesterol (the “bad” cholesterol) is most likely to narrow arteries. A high level of triglycerides, a type of blood fat related to your diet, also ups your risk of a heart attack. However, a high level of high-density lipoprotein (HDL) cholesterol (the “good” cholesterol) lowers your risk of a heart attack.
- **Fasting Blood Sugar:** Not producing enough of a hormone secreted by your pancreas (insulin) or not responding to insulin properly causes your body’s blood sugar levels to rise, increasing your risk of a heart attack.
- **Resting ECG:** For people at low risk of cardiovascular disease, the USPSTF concludes with moderate certainty that the potential harms of screening with resting or exercise ECG equal or exceed the potential benefits. For people at intermediate to high risk, current evidence is insufficient to assess the balance of benefits and harms of screening.
- **Max heart rate achieved:** The increase in cardiovascular risk, associated with the acceleration of heart rate, was comparable to the increase in risk observed with high blood pressure. It has been shown that an increase in heart rate by 10 beats per minute was associated with an increase in the risk of cardiac death by at least 20%, and this increase in the risk is similar to the one observed with an increase in systolic blood pressure by 10 mm Hg.
- **Exercise induced angina:** The pain or discomfort associated with angina usually feels tight, gripping or squeezing, and can vary from mild to severe. Angina is usually felt in the center of your chest but may spread to either or both of your shoulders, or your back, neck, jaw or arm. It can even be felt in your hands. o Types of Angina a. Stable Angina / Angina Pectoris b. Unstable Angina c. Variant (Prinzmetal) Angina d. Microvascular Angina.
- **Peak exercise ST segment:** A treadmill ECG stress test is considered abnormal when there is a horizontal or down-sloping ST-segment depression ≥ 1 mm at 60–80 ms after the J point. Exercise ECGs with up-sloping ST-segment depressions are typically reported as an ‘equivocal’ test. In general, the occurrence of horizontal or down-sloping ST-segment depression at a lower workload (calculated in METs) or heart rate indicates a worse prognosis and higher likelihood of multi-vessel disease. The duration of ST-segment depression is also important, as prolonged recovery after peak stress is consistent with a positive treadmill ECG stress test. Another finding that is highly indicative of significant CAD is the occurrence of ST-segment elevation > 1 mm (often suggesting transmural ischemia); these patients are frequently referred urgently for coronary angiography.
""")
tab1.divider()
tab1.header(":bar_chart: Data Analysis")
tab1.write(
  "Let us look at the people’s age who are suffering from the disease or not.Here, target = 1 implies that the person is suffering from heart disease and target = 0 implies the person is not suffering."
)
tab1.image("data_analyse.webp")
tab1.divider()
tab1.header(
  " :technologist: Industrial Scope of Heart Disease Prediction System")
tab1.markdown("""
- Machine learning (ML) proves to be effective in assisting in making decisions and predictions from the large quantity of data produced by the healthcare industry.
- This makes heart disease a major concern to be dealt with. But it is difficult to identify heart disease because of several contributory risk factors such as diabetes, high blood pressure, high cholesterol, abnormal pulse rate, and many other factors. Due to such constraints, scientists have turned towards modern approaches like Data Mining and Machine Learning for predicting the disease.
""")
tab2.title("Heart Dieases Prediction System")
tab2.image("main1.png")
tab2.markdown("""
**Credits**
- App built in `Python` + `Streamlit` by Abhishek :coffee:
- Model used `LogisticRegression`
- Github:link: `https://github.com/abhiishekdhiman/Heart-Disease`
""")
tab2.divider()
st.sidebar.header('User Input Paramter')

st_age = st.sidebar.slider('Age', 0, 100, 23)

st_sex = st.sidebar.radio('Sex', ['Male', 'Female'])
if st_sex == 'Male':
  st_sex = int(1)
else:
  st_sex = int(0)

st_cp = st.sidebar.selectbox(
  'Chest Pain Type',
  ['Typical Angina', 'Atypical Angina', 'Non — Anginal Pain', 'Asymptotic'])
if st_cp == 'Typical Angina':
  st_cp = int(0)
elif st_cp == 'Atypical Angina':
  st_cp = int(1)
elif st_cp == 'Non — Anginal Pain':
  st_cp = int(2)
elif st_cp == 'Asymptotic':
  st_cp = int(3)

st_trestbps = st.sidebar.slider('Resting Blood Pressure [ mmHg (unit) ] ', 94,
                                200, 120)

st_chol = st.sidebar.slider('Serum Cholestrol [ mg/dl (unit) ]', 126, 564, 200)

st_fbs = st.sidebar.radio('Fasting Blood Sugar  > 120 mg/dl',
                          ['True', 'False'])
if st_fbs == 'True':
  st_fbs = int(1)
else:
  st_fbs = int(0)

st_ecg = st.sidebar.selectbox(
  'Resting ECG Result',
  ['Normal', 'Having ST-T wave Abnormality', 'Left ventricular Hyperthrophy'])
if st_ecg == 'Normal':
  st_ecg = int(0)
elif st_ecg == 'Having ST-T wave Abnormality':
  st_ecg = int(1)
elif st_ecg == 'Left ventricular Hyperthrophy':
  st_ecg = int(2)

st_thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 102)

st_exang = st.sidebar.radio('Exercise Induced Angina', ['True', 'False'])
if st_exang == 'True':
  st_exang = int(1)
else:
  st_exang = int(0)

st_oldpeak = st.sidebar.slider(
  'ST depression Induced by exercise relative to rest', 0.0, 7.0, 2.0)

st_slope = st.sidebar.selectbox('Peak Exercise ST Segment',
                                ['Upsloping', 'Flat', 'Downsloping'])
if st_slope == 'Upsloping':
  st_slope = int(0)
elif st_slope == 'Flat':
  st_slope = int(1)
elif st_slope == 'Downsloping':
  st_slope = int(2)

st_ca = st.sidebar.slider(
  'Number of major vessels (1–4) colored by flourosopy', 0, 4, 2)

st_thal = st.sidebar.selectbox('Thalassemia',
                               ['Normal', 'Fixed Defect', 'Reversible Defect'])
if st_thal == 'Normal':
  st_thal = int(1)
elif st_thal == 'Fixed Defect':
  st_thal = int(2)
if st_thal == 'Reversible Defect':
  st_thal = int(3)

input_user_DF = pd.DataFrame(
  {
    'age': [st_age],
    'sex': [st_sex],
    'cp': [st_cp],
    'trestbps': [st_trestbps],
    'chol': [st_chol],
    'fbs': [st_fbs],
    'restecg': [st_ecg],
    'thalach': [st_thalach],
    'exang': [st_exang],
    'oldpeak': [st_oldpeak],
    'slope': [st_slope],
    'ca': [st_ca],
    'thal': [st_thal]
  },
  index=[0])


@st.cache_resource
def prediction(input_user):
  return model.predict(input_user)


if tab2.button('Show User Input { DataFrame }'):
  st.write(input_user_DF)
if tab2.button('Show Result'):
  pred_value = prediction(input_user_DF)
  if pred_value == 0:
    tab2.success('The Person does not have a Heart Disease')
  else:
    tab2.warning('The Person has Heart Disease')
