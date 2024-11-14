
import streamlit as st

# Title
st.title("Streamlit Demo")

# Markdown
st.markdown("""
This is a demo app showcasing a few of Streamlit's features.

Streamlit is a powerful Python library for creating web apps. It is easy to use and has a wide range of features, including:

* **Interactive widgets:** Streamlit makes it easy to create interactive widgets, such as sliders, dropdown menus, and radio buttons.
* **Charts and graphs:** Streamlit can generate a variety of charts and graphs, including line charts, bar charts, and pie charts.
* **Data display:** Streamlit can display data in a variety of ways, including tables, lists, and maps.
* **Deployment:** Streamlit apps can be deployed to Heroku with a single command.
""")

# Slider
slider_value = st.slider("Select a number:", 0, 100)
st.write(f"You selected: {slider_value}")

# Dropdown menu
dropdown_value = st.selectbox("Choose a color:", ["red", "green", "blue"])
st.write(f"You chose: {dropdown_value}")

# Radio buttons
radio_button_value = st.radio("Select a language:", ["English", "Spanish", "French"])
st.write(f"You selected: {radio_button_value}")

# Text area
text = st.text_area("Enter some text:")
if text:
    st.write(f"You entered: {text}")

# Button
if st.button("Click me!"):
    st.write("You clicked the button!")

# Chart
data = {"x": [1, 2, 3, 4, 5], "y": [6, 7, 2, 4, 5]}
st.line_chart(data)

# Map
map_data = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8783, "lon": -87.6233},
]
st.map(map_data)
