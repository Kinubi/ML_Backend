from src import ML_Backend
import pandas as pd
import re
import numpy as np

df = pd.read_csv("data/train_data.csv", names=["id", "property_description", "area_jhb", "listing_agency", "rental_amount", "street_address", "occupation_date", "Bedrooms_1", "Bedrooms_2", "bathrooms", "area", "Feature_1", "Feature_2", "additional"])
df = df[df["id"] != "id"]
df = df[~df["property_description"].str.contains("POA")]

df["Extracted Price"] = df["property_description"].str.extract(r"R\s+(.*?\s\d{3})")

dfperday = df[df["property_description"].str.contains("per day")]

df = df[~df["property_description"].str.contains("per day")]
df = df.drop_duplicates(subset=["property_description"])



dfperday["Extracted Price"] = dfperday["property_description"].str.extract(r"R\s+(.*)\sper\sday")
dfperday["Extracted Price"] = dfperday["Extracted Price"] .str.replace("\s", "").astype(int) * 30

df["Extracted Price"] = df["Extracted Price"].str.replace("\s", "").astype(int)
df = pd.concat([df, dfperday])

df["Extracted area from description"] = df["property_description"].str.extract("Size:\s(.*)\smB2")
df["Extracted area from area"] = df["area"].str.extract("(\d*)\smB2")
df["Extracted area from description"] = df["Extracted area from description"].str.replace("\s", "0").replace(np.nan, 0)
df["Extracted area from area"] = df["Extracted area from area"].str.replace("\s", "0").replace(np.nan, 0)


dfa = df[(df["Extracted area from description"].astype(float) >= df["Extracted area from area"].astype(float))]
dfb = df[(df["Extracted area from description"].astype(float) < df["Extracted area from area"].astype(float))]

dfa["Extracted area"] = dfa["Extracted area from description"].astype(float)
dfb["Extracted area"] = dfa["Extracted area from area"].astype(float)

df= pd.concat([dfa, dfb, df])

df = df.drop_duplicates(subset=["property_description", "id"],keep='first')


df["Extracted bedrooms from Bedrooms_1"] = df[(df["listing_agency"] == " Bathrooms") | (df["listing_agency"] == " Beds")]["Bedrooms_1"]
df["Extracted bedrooms from Bedrooms_1"] = df["Extracted bedrooms from Bedrooms_1"].replace(np.nan, "0").str.replace("\s", "")

dfsplit = df[df["Extracted bedrooms from Bedrooms_1"].str.contains("\|")]

dfsplit["Extracted bedrooms from Bedrooms_1"] = dfsplit["Extracted bedrooms from Bedrooms_1"].str.split("|").str[0].astype(float)
df =  df[~df["Extracted bedrooms from Bedrooms_1"].str.contains(r"\|")]
df = pd.concat([dfsplit, df])


df["Extracted Bedrooms"] = df[df["Bedrooms_1"]=="Beds"]["Bedrooms_2"]



# df.loc[pd.to_numeric(df["Extracted bedrooms from Bedrooms_1"], errors="coerce").isnull(), "Extracted bedrooms from Bedrooms_1"] = 0
# df.loc[pd.to_numeric(df["Extracted bedrooms from Bedrooms_2"], errors="coerce").isnull(), "Extracted bedrooms from Bedrooms_2"] = 0

# dfa = df[(df["Extracted bedrooms from Bedrooms_1"].astype(float) >= df["Extracted bedrooms from Bedrooms_2"].astype(float))]
# dfb = df[(df["Extracted bedrooms from Bedrooms_1"].astype(float) < df["Extracted bedrooms from Bedrooms_2"].astype(float))]

# dfa["Extracted Bedrooms"] = dfa["Extracted bedrooms from Bedrooms_1"].astype(float)
# dfb["Extracted Bedrooms"] = dfa["Extracted bedrooms from Bedrooms_2"].astype(float)
# df= pd.concat([dfa, dfb, df])

# df = df.drop_duplicates(subset=["property_description", "id"],keep='first')
#df.loc[pd.to_numeric(df["Extracted bedrooms from Bedrooms_1"], errors="coerce").isnull(), "Extracted bedrooms from Bedrooms_1"] = 0
#df.loc[pd.to_numeric(df["Extracted bedrooms from Bedrooms_2"], errors="coerce").isnull(), "Extracted bedrooms from Bedrooms_2"] = 0
# print(df[df["Extracted bedrooms from Bedrooms_2"].astype(float)>0])
# dfbd1 = df[(df["Extracted bedrooms from Bedrooms_1"].astype(float) > 0)]
# dfbd2 = df[(df["Extracted bedrooms from Bedrooms_2"].astype(float) > 0)]

# dfrest = df[(df["Extracted bedrooms from Bedrooms_2"].astype(float) == 0) & (df["Extracted bedrooms from Bedrooms_1"].astype(float) == 0)]
# dfbd1["Extracted Bedrooms"] = dfbd1['Extracted bedrooms from Bedrooms_1'].copy()
# print(dfbd1[dfbd1["Extracted Bedrooms"].astype(float)> 0])
# dfbd2["Extracted Bedrooms"] = dfbd1['Extracted bedrooms from Bedrooms_2'].copy()
# print(dfbd2[dfbd2["Extracted Bedrooms"].astype(float)> 0])

# dfrest["Extracted Bedrooms"] = np.nan
# df = pd.concat([dfbd1, dfbd2, dfrest], ignore_index=True)
print(df[df["Extracted Bedrooms"].astype(float) > 0])


df["Extracted Bathrooms"] = df[df["listing_agency"] != " Bathrooms"]["bathrooms"]
df["Extracted Bathrooms"] = df["Extracted Bathrooms"].replace(np.nan, "0").str.replace("\s", "")
dfsplit = df[df["Extracted Bathrooms"].str.contains("\|")]
dfsplit["Extracted Bathrooms"] = dfsplit["Extracted Bathrooms"].str.split("|").str[0].astype(float)
df =  df[~df["Extracted Bathrooms"].str.contains("\|")]
df = pd.concat([df, dfsplit])

dfbeds = df[(df["listing_agency"] == " Beds") & (df["bathrooms"].isna())]
df = df[df["listing_agency"] != " Beds"]

dfbeds["Extracted Bathrooms"] = dfbeds["Bedrooms_1"].astype(float)
df["Extracted Bathrooms"] = df["Extracted Bathrooms"].replace(np.nan, "0").str.replace("\s", "")
dfsplit = df[df["Extracted Bathrooms"].replace(np.nan, "0").str.contains("\|")]
dfsplit["Extracted Bathrooms"] = dfsplit["Extracted Bathrooms"].str.split("|").str[1].astype(float)
df =  df[~df["Extracted Bathrooms"].replace(np.nan, "0").str.contains("\|")]
df.loc[pd.to_numeric(df["Extracted Bathrooms"], errors="coerce").isnull(), "Extracted Bathrooms"] = np.nan
df["Extracted Bathrooms"] = df["Extracted Bathrooms"].astype(float)
df = pd.concat([df, dfsplit])

df["Extracted Area_JHB"] = df["area_jhb"].str.extract(r"in\s(.*)\s\-")
df["Extracted Area_JHB"] = df["Extracted Area_JHB"].replace(np.nan, "None")



df["Extracted listing agency"] = df[(df["listing_agency"]!= " Beds") | (df["listing_agency"]!= " Bathrooms")]["listing_agency"]
df["Extracted listing agency"] = df["Extracted listing agency"].replace(np.nan, "None").str.replace("Bathrooms", "None")


df["Extracted Type"] = df[(df["listing_agency"]!= " Beds") | (df["listing_agency"]!= " Bathrooms")]["area_jhb"].str.extract("Bedroom\s(.*?)\s")
df["Extracted Type"] = df["Extracted Type"].replace(np.nan, "None")

df["Extracted Available Month"] = df["occupation_date"].str.extract(r"Available:\s\d{1,2}\s(.*?)\s\d{4}")
df["Extracted Available Month"] = df["Extracted Available Month"].replace(np.nan, "None")



df = df[["id", "Extracted Price", "Extracted Bedrooms", "Extracted area", "Extracted Bathrooms", "Extracted Area_JHB", "Extracted listing agency", "Extracted Type", "Extracted Available Month"]]
print(df)
df.to_csv("data/cleaned_train.csv", index=False)