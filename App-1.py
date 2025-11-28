



import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# بيانات الطلاب
Students = ["Ali","Mosa","Mona","Mohamed","Marawan","sohila","ghada","russa","Yossef","Ibrahim"]
grades_math = [56,65,87,90,57,43,98,99,53,89]
grades_scince = [66,78,87,27,69,59,88,97,88,77]
grades_english = [75,40,50,68,47,98,55,86,47,76]
behavior = ["bad","good","bad","good","bad","good","good","good","bad","bad"]
absences = [1,2,8,6,7,5,2,4,9,8]
issues = [0,5,7,1,6,6,4,8,9,1]

# إنشاء DataFrame
df = pd.DataFrame({
    "Students": Students,
    "math": grades_math,
    "scince": grades_scince,
    "english": grades_english,
    "behavior": behavior,
    "absences": absences,
    "issues": issues
})

# ترميز السلوك
le = LabelEncoder()
df['behavior_Numiric'] = le.fit_transform(df['behavior'])

# تقسيم البيانات
X = df[["math","scince","english","absences","issues"]]
y = df['behavior_Numiric']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# نموذج الانحدار اللوجستي
model = LogisticRegression()
model.fit(Xtrain, Ytrain)

# --- Streamlit App ---
st.title("تنبؤ سلوك الطالب")

st.write("ادخل بيانات الطالب الجديد:")

math = st.number_input("درجة الرياضيات", min_value=0, max_value=100, value=90)
scince = st.number_input("درجة العلوم", min_value=0, max_value=100, value=90)
english = st.number_input("درجة اللغة الإنجليزية", min_value=0, max_value=100, value=90)
absences = st.number_input("عدد الغيابات", min_value=0, max_value=20, value=0)
issues = st.number_input("عدد المشكلات", min_value=0, max_value=20, value=0)

if st.button("تنبؤ بالسلوك"):
    New_Student = pd.DataFrame([{
        "math": math,
        "scince": scince,
        "english": english,
        "absences": absences,
        "issues": issues
    }])
    
    Predict_New_Student = model.predict(New_Student)
    convert = le.inverse_transform(Predict_New_Student)
    
    st.success(f"السلوك المتوقع للطالب الجديد: {convert[0]}")

