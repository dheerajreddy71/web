import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# Load and prepare datasets
url_yield_data = "https://github.com/dheerajreddy71/Design_Project/raw/main/yield_df.csv"
url_crop_data = "https://github.com/dheerajreddy71/Design_Project/raw/main/Crop_recommendation.csv"
url_temp_data = "https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv"
url_pest_data = "https://github.com/dheerajreddy71/Design_Project/raw/main/ds2.csv"
url_price_data = "https://github.com/dheerajreddy71/Design_Project/raw/main/pred_data.csv"

yield_df = pd.read_csv(url_yield_data)
crop_recommendation_data = pd.read_csv(url_crop_data)
data = pd.read_csv(url_temp_data, encoding='ISO-8859-1')
pest_data = pd.read_csv(url_pest_data)
price_data = pd.read_csv(url_price_data, encoding='ISO-8859-1')

# Prepare Yield Prediction Model
yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

# Prepare Crop Recommendation Model
crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Prepare Temperature and Humidity Prediction Model
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Load Pest Data
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

with open(url_pest_data, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header row if present
    for row in csvreader:
        if len(row) >= 2:
            crop = row[0].strip().lower()
            pest = row[1].strip()
            crop_pest_data[crop] = pest
            if len(row) >= 3:
                planting_time = row[5].strip()
                planting_time_info[crop] = planting_time
                growth_stage = row[6].strip()
                growth_stage_info[crop] = growth_stage
                pesticides_row = row[4].strip()
                pesticides_info[crop] = pesticides_row

def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None  # Handle cases where crop_name is not found
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    pest_warnings = []

    if crop_name in crop_pest_data:
        pests = crop_pest_data[crop_name].split(', ')
        warning_message = f"\nBeware of pests like {', '.join(pests)} for {crop_name.capitalize()}.\n"

        if crop_name in planting_time_info:
            planting_time = planting_time_info[crop_name]
            warning_message += f"\nPlanting Time: {planting_time}\n"

        if crop_name in growth_stage_info:
            growth_stage = growth_stage_info[crop_name]
            warning_message += f"\nGrowth Stages of Plant: {growth_stage}\n"

        if crop_name in pesticides_info:
            pesticides = pesticides_info[crop_name]
            warning_message += f"\nUse Pesticides like: {pesticides}\n"
            
        pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)

# Prepare Price Prediction Model
price_data['arrival_date'] = pd.to_datetime(price_data['arrival_date'])
price_data['day'] = price_data['arrival_date'].dt.day
price_data['month'] = price_data['arrival_date'].dt.month
price_data['year'] = price_data['arrival_date'].dt.year
price_data.drop(['arrival_date'], axis=1, inplace=True)

price_X = price_data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
price_y = price_data[['min_price', 'max_price', 'modal_price']]

price_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

price_X_encoded = price_encoder.fit_transform(price_X)
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price_X_encoded, price_y, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(price_X_train, price_y_train)

# Streamlit interface
st.title("Agricultural Dashboard")

# Yield Prediction
st.header("Yield Prediction")
year = st.number_input("Year", min_value=2000, max_value=2100)
rainfall = st.number_input("Average Rainfall (mm per year)")
pesticides = st.number_input("Pesticides (tonnes)")
temp = st.number_input("Average Temperature (°C)")
area = st.text_input("Area")
item = st.text_input("Item")

if st.button("Predict Yield"):
    features_array = np.array([[year, rainfall, pesticides, temp, area, item]], dtype=object)
    transformed_features = yield_preprocessor.transform(features_array)
    predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)
    st.write(f"The predicted yield is {predicted_yield[0][0]:.2f} hectograms (hg) per hectare (ha).")

# Crop Recommendation
st.header("Crop Recommendation")
N = st.number_input("N")
P = st.number_input("P")
K = st.number_input("K")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Recommend Crop"):
    crop_features = [N, P, K, temperature, humidity, ph, rainfall]
    recommended_crop = crop_model.predict([crop_features])[0]
    st.write(f"Recommended Crop: {recommended_crop}")

# Temperature and Humidity Requirements
st.header("Temperature and Humidity Requirements")
crop_name = st.text_input("Crop Name")

if st.button("Get Requirements"):
    humidity, temperature = predict_requirements(crop_name)
    if humidity is not None and temperature is not None:
        st.write(f"Humidity Required: {humidity}%")
        st.write(f"Temperature Required: {temperature}°F")
    else:
        st.write("Crop not found.")

# Pest Warnings
st.header("Pest Warnings")
if st.button("Get Pest Warnings"):
    pest_warning = predict_pest_warnings(crop_name)
    st.write(pest_warning)

# Price Prediction
st.header("Price Prediction")
state = st.text_input("State")
district = st.text_input("District")
market = st.text_input("Market")
commodity = st.text_input("Commodity")
variety = st.text_input("Variety")
arrival_date = st.date_input("Arrival Date")

if st.button("Predict Prices"):
    if all([state, district, market, commodity, variety, arrival_date]):
        input_data = {
            'state': state,
            'district': district,
            'market': market,
            'commodity': commodity,
            'variety': variety,
            'arrival_date': pd.to_datetime(arrival_date)
        }
        input_df = pd.DataFrame([input_data])
        input_df['day'] = input_df['arrival_date'].dt.day
        input_df['month'] = input_df['arrival_date'].dt.month
        input_df['year'] = input_df['arrival_date'].dt.year
        input_df.drop(['arrival_date'], axis=1, inplace=True)
        input_encoded = price_encoder.transform(input_df)

        predicted_prices = price_model.predict(input_encoded)
        min_price, max_price, modal_price = predicted_prices[0]
        st.write(f"Min Price: {min_price}")
        st.write(f"Max Price: {max_price}")
        st.write(f"Modal Price: {modal_price}")
    else:
        st.write("Please fill all fields.")
