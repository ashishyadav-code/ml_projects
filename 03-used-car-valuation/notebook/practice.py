import pandas as pd

# Yeh raha real dataset ka sample DataFrame:
data = {
    "name": [
        "Maruti Swift Dzire VDI",
        "Hyundai i20 1.4 CRDi Asta",
        "BMW X1 sDrive20d",
        "Land Rover Discovery Sport TD4 HSE",
        "Tata Tiago 2019-2020 Revotron XZ",
    ],
    "mileage": [
        "23.4 kmpl",
        "22.54 kmpl",
        "19.0 km/kg",     # Notice: CNG car me km/kg likha hai!
        "12.83 kmpl",
        "23.84 kmpl",
    ],
    "engine": [
        "1248 CC",
        "1396 CC",
        "1995 CC",
        "1999 CC",
        "1199 CC",
    ],
    "max_power": [
        "74 bhp",
        "88.7 bhp",
        "187.7 bhp",
        "148.31 bhp",
        "83.81 bhp",
    ],
    "year": [2014, 2017, 2016, 2018, 2019],
    "km_driven": [145500, 60000, 45000, 35000, 22000]
}

df_sample = pd.DataFrame(data)

df_sample['brand'] = df_sample['name'].apply(lambda x: x.split()[0].lower().strip())
import time
year = time.localtime().tm_year
df_sample['age'] = year - df_sample['year']
df_sample['mileage_kmpl'] = df_sample['mileage'].str.extract(r'([0-9.]+)').astype(float)
df_sample['engine_cc'] = df_sample['engine'].str.extract(r'([0-9.]+)').astype(float)
df_sample['max_power_bhp'] = df_sample['max_power'].str.extract(r'([0-9.]+)').astype(float)
df_sample = df_sample.drop(columns=['name', 'mileage', 'engine', 'max_power', 'year'])   
print(df_sample)