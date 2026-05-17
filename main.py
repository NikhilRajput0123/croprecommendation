import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("Crop_recommendation.csv")
x=df.drop("label",axis=1)
y=df["label"]
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pred=model.predict(x_test)
score=accuracy_score(y_test,pred)
print(f"Model Accuracy:{score:.2f}")


