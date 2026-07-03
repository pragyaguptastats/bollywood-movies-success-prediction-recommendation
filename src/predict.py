import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load(
    "models/best_regressor.pkl"
)
movie = {}

movie["Release_Period"] = input("Release Period : ")
movie["Whether_Remake"] = input("Remake (Yes/No) : ")
movie["Whether_Franchise"] = input("Franchise (Yes/No) : ")
movie["Genre"] = input("Genre : ")
movie["New_Actor"] = input("New Actor (Yes/No) : ")
movie["New_Director"] = input("New Director (Yes/No) : ")
movie["New_Music_Director"] = input("New Music Director (Yes/No) : ")
movie["Lead_Star"] = input("Lead Star : ")
movie["Director"] = input("Director : ")
movie["Music_Director"] = input("Music Director : ")

movie["Number_of_Screens"] = int(
    input("Number of Screens : ")
)

movie["Budget(INR)"] = int(
    input("Budget (INR) : ")
)

df = pd.DataFrame([movie])

for col in df.columns:

    if df[col].dtype == object:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

prediction = model.predict(df)

print("\n")
print("PREDICTED MOVIE REVENUE")

print(f"Estimated Revenue : ₹ {prediction[0]:,.0f}")