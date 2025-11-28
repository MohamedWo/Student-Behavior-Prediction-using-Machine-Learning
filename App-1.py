import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from  sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# data={
#     "Students":["Ali","Mosa","Mona","Mohamed","Marawan","sohila","ghada","russa","Yossef","Ibrahim"],
#     "grades_math":[56,65,87,90,57,43,98,99,53,89],
#     "grades_scince":[66,78,87,27,69,59,88,97,88,77],
#     "grades_english":[75,40,50,68,47,98,55,86,47,76],
#     "behavior":["bad","good","bad","good","bad","good","good","good","bad","bad"],
#     "absences":[1,2,8,6,7,5,2,4,9,8],
#     "issues":[0,5,7,1,6,6,4,8,9,1],
# }
Students=["Ali","Mosa","Mona","Mohamed","Marawan","sohila","ghada","russa","Yossef","Ibrahim"]
grades_math=[56,65,87,90,57,43,98,99,53,89]
grades_scince=[66,78,87,27,69,59,88,97,88,77]
grades_english=[75,40,50,68,47,98,55,86,47,76]
behavior=["bad","good","bad","good","bad","good","good","good","bad","bad"]
absences=[1,2,8,6,7,5,2,4,9,8]
issues=[0,5,7,1,6,6,4,8,9,1]


# newdata=pd.DataFrame(data)


df=pd.DataFrame({
    "Students":Students,
    "math":grades_math,
    "scince":grades_scince,
    "english":grades_english,
    "behavior":behavior,
    "absences":absences,
    "issues":issues
})

# df.to_csv("studebts_data.csv",index=False)

le=LabelEncoder()
df['behavior_Numiric']=le.fit_transform(df['behavior'])
df['behavior_Numiric']


x=df[["math","scince","english","absences","issues"]]
y=df['behavior_Numiric']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.3,random_state=42)

model=LogisticRegression()
model.fit(Xtrain,Ytrain)
Y_pre=model.predict(Xtest)



New_Student={
    "math": 95 ,
    "scince":95 ,
    "english": 91,
    "absences":0 ,
    "issues":0
}
New_Student=pd.DataFrame([New_Student])

Predict_New_Student=model.predict(New_Student)

convert=le.inverse_transform(Predict_New_Student)


print(f"behavior new students : {convert[0]}")


