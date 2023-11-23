import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import streamlit as st
import pydeck as pdk

# Load your dataset (replace 'path/to/your/dataset.csv' with the actual path)
df = pd.read_csv("melb_data.csv")

# Clearing outliers in BuildingArea by taking out the top 5 percent and lower 5 percent
low, high = df["BuildingArea"].quantile([0.05, 0.95])
mask_area = df["BuildingArea"].between(low, high)
df = df[mask_area]

# Removing outlier from prices
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to exclude outliers
df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]

# Features and target
target = "Price"
features = ["BuildingArea", "Lattitude", "Longtitude"]
X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = make_pipeline(Ridge())
model.fit(X_train, y_train)

# Streamlit app
st.title("Apartment Price Prediction In Melbourne")

# Sidebar with user input
st.sidebar.header("User Input")
building_area = st.sidebar.slider("Building Area [m2]", float(X["BuildingArea"].min()), float(X["BuildingArea"].max()), float(X["BuildingArea"].mean()))
latitude = st.sidebar.slider("Latitude", float(X["Lattitude"].min()), float(X["Lattitude"].max()), float(X["Lattitude"].mean()))
longitude = st.sidebar.slider("Longitude", float(X["Longtitude"].min()), float(X["Longtitude"].max()), float(X["Longtitude"].mean()))
predict_button = st.sidebar.button("Predict")

if predict_button:
    # Make predictions
    prediction = model.predict([[building_area, latitude, longitude]])
    prediction = round(prediction[0], 2)

    # Display prediction
    st.subheader("Prediction")
    st.write(f"The predicted apartment price is ${prediction}")

    # Display the map with the selected location
    # Create a PyDeck scatterplot layer for the map
    scatterplot = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({"latitude": [latitude], "longitude": [longitude]}),
        get_position="[longitude, latitude]",
        get_radius=100,
        get_fill_color=[255, 0, 0],  # Red color
    )

    # Create a PyDeck map using the scatterplot layer
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=12, pitch=50),
        layers=[scatterplot],
    )

    # Display the PyDeck chart using st.pydeck_chart
    st.pydeck_chart(deck)
else:
    st.subheader("Map of Melbourne")
    
    # Create a PyDeck scatterplot layer for the default map
    default_map_scatterplot = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({"latitude": [latitude], "longitude": [longitude]}),
        get_position="[longitude, latitude]",
        get_radius=200,
        get_fill_color=[0, 0, 255],  # Blue color
        pickable=True,
        auto_highlight=True
    )

    # Create a PyDeck map using the scatterplot layer
    default_map_deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=12, pitch=50),
        layers=[default_map_scatterplot],
    )

    # Display the PyDeck chart using st.pydeck_chart
    st.pydeck_chart(default_map_deck)
    
    # Display the coordinates
    st.write(f"Default Location Coordinates: Latitude {latitude}, Longitude {longitude}")
# Include the user manual within the app
st.sidebar.markdown("# User Manual")
st.sidebar.markdown("""
**1. Introduction**
Welcome to the "Apartment Price Prediction In Melbourne" app! This user-friendly tool is designed to predict apartment prices based on user inputs like building area, latitude, and longitude.

**2. Getting Started**
2.1 **Sidebar Input:**
   - Use the sliders in the sidebar to set the Building Area, Latitude, and Longitude.
   - Building Area: Adjust the slider to specify the apartment's size in square meters.
   - Latitude: Set the latitude coordinate of the location.
   - Longitude: Set the longitude coordinate of the location.

**3. Making Predictions**
3.1 **Predict Button:**
   - Click the "Predict" button to get the predicted apartment price based on your input.

**4. Result Display**
4.1 **Prediction:**
   - After clicking "Predict," the app will display the predicted apartment price.

4.2 **Selected Location on Map:**
   - If the "Predict" button is clicked, a map will appear below showing the selected location with a red marker.

4.3 **Default Map of Melbourne:**
   - If the "Predict" button is not clicked, a default map of Melbourne will be displayed with a blue marker at the specified default coordinates.

4.4 **Default Location Coordinates:**
   - Below the default map, you can find the default location coordinates (latitude and longitude).

**5. Additional Information**
5.1 **About the App:**
   - The app uses a machine learning model to make predictions based on a Ridge regression algorithm.
   - Location coordinates are visualized on an interactive map using PyDeck and Mapbox.

5.2 **Note:**
   - Ensure that you provide valid inputs for building area, latitude, and longitude to get accurate predictions.

**6. Enjoy Using the App!**
Feel free to explore and have fun predicting apartment prices in Melbourne with our user-friendly app!
""")
