import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import folium_static
from st_aggrid import AgGrid
from streamlit_lottie import st_lottie
import requests
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit_tags import st_tags

# Title and Headers
st.title("Streamlit Components Showcase")

# Text and Markdown
st.text("Hello, Streamlit!")  # Display plain text
st.markdown("# Hello, Streamlit!")  # Markdown header
st.header("This is a Header")  # Level 1 header
st.subheader("This is a Subheader")  # Level 2 header

# Input Widgets
name = st.text_input("Enter your name")  # Text input box
bio = st.text_area("Tell us about yourself")  # Multi-line text area

# Button
if st.button("Click me"):  # Button that triggers an action
    st.write("Button clicked!")  # Action when button is clicked

# File Download
st.download_button("Download CSV", data="data,text", file_name="file.csv")  # Button to download a CSV file

# Checkbox
agree = st.checkbox("I agree")  # Checkbox for agreement

# Radio Button
option = st.radio("Choose an option", ["Option 1", "Option 2", "Option 3"])  # Radio button selection

# Selectbox
choice = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])  # Dropdown menu for selection

# Multiselect
choices = st.multiselect("Select options", ["Option 1", "Option 2", "Option 3"])  # Multiple selections

# Slider
value = st.slider("Pick a value", min_value=0, max_value=100, value=50)  # Slider for selecting a value

# Number Input
num = st.number_input("Enter a number", min_value=0, max_value=100, value=50)  # Numeric input box

# Date Input
date = st.date_input("Pick a date")  # Date picker

# Time Input
time = st.time_input("Pick a time")  # Time picker

# File Uploader
file = st.file_uploader("Upload a file")  # File upload widget

# Progress Bar
progress = st.progress(0)  # Progress bar with initial value

# Data Display
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
st.dataframe(df)  # Display interactive DataFrame
st.table(df)  # Display static DataFrame as a table

# Charts
st.line_chart([1, 2, 3, 4])  # Line chart
st.bar_chart({'A': [1, 2, 3], 'B': [4, 5, 6]})  # Bar chart
st.area_chart([1, 2, 3, 4])  # Area chart

# Matplotlib Plot
plt.plot([1, 2, 3, 4])  # Create a plot
st.pyplot()  # Display the plot

# Sidebar
st.sidebar.title("Sidebar Title")
st.sidebar.selectbox("Choose an option", ["Option 1", "Option 2"])  # Sidebar dropdown menu

# Container
with st.container():
    st.write("This is inside the container")  # Content inside container
    st.bar_chart({"data": [1, 2, 3, 4, 5]})  # Chart inside container

st.write("This is outside the container")  # Content outside container

# Columns
col1, col2, col3 = st.columns(3)
col1.write("Column 1")  # Content in column 1
col2.write("Column 2")  # Content in column 2
col3.write("Column 3")  # Content in column 3

col1.image("https://via.placeholder.com/150", caption="Image 1")  # Image in column 1
col2.image("https://via.placeholder.com/150", caption="Image 2")  # Image in column 2
col3.image("https://via.placeholder.com/150", caption="Image 3")  # Image in column 3

# Expander
with st.expander("Click to expand"):
    st.write("This content is hidden by default. Click to see more details.")  # Content inside expander
    st.line_chart({"data": [1, 3, 2, 4, 5]})  # Chart inside expander

# Tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1:
    st.write("This is Tab 1 content")  # Content for Tab 1
with tab2:
    st.write("This is Tab 2 content")  # Content for Tab 2
with tab3:
    st.write("This is Tab 3 content")  # Content for Tab 3

# Form
with st.form("my_form"):
    name = st.text_input("Name")  # Text input inside form
    age = st.slider("Age", 18, 100)  # Slider inside form
    submitted = st.form_submit_button("Submit")  # Form submission button

if submitted:
    st.write(f"Name: {name}, Age: {age}")  # Output after form submission

# Metric
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")  # Display metric with change indicator
st.metric(label="Humidity", value="50%", delta="-2%")  # Display metric with change indicator

# Selectbox
option = st.selectbox('Which number do you like best?', [1, 2, 3, 4, 5])  # Dropdown menu for number selection
st.write('You selected:', option)  # Display selected option

# File Uploader
uploaded_file = st.file_uploader("Choose a file")  # File upload widget

if uploaded_file is not None:
    data = uploaded_file.read()
    st.write(data)  # Display uploaded file data

# Date Input
from datetime import date
d = st.date_input("When's your birthday", date(2024, 1, 1))  # Date picker for birthday
st.write('Your birthday is:', d)  # Display selected date

# Plotly Chart
import plotly.express as px

df = pd.DataFrame({
    'Credit Score': [700, 750, 800, 850, 900],
    'Probability': [0.1, 0.3, 0.5, 0.7, 0.9]
})

fig = px.scatter(df, x='Credit Score', y='Probability', title='Credit Score vs. Default Probability')  # Plotly scatter plot
st.plotly_chart(fig)  # Display Plotly chart

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)  # Apply custom CSS styles
st.write("This Streamlit app has custom styling.")  # Inform about custom styling

# AgGrid
df = pd.DataFrame({
    'Customer ID': [1, 2, 3, 4],
    'Credit Score': [700, 650, 800, 720],
    'Default Probability': [0.1, 0.4, 0.05, 0.2]
})

AgGrid(df)  # Display DataFrame with AgGrid

# Interactive Maps with Folium
m = folium.Map(location=[45.4567, -123.4567], zoom_start=13)  # Create a Folium map
folium.Marker([45.4567, -122.4567], tooltip='Portland').add_to(m)  # Add marker to the map
folium_static(m)  # Display the map

# Streamlit Card
from streamlit_card import card

card(
    title="Credit Score",
    text="Your estimated credit score is 750.",
    image="https://via.placeholder.com/150",
    url="https://credit-score.com"
)  # Display a card with information

# Tags
tags = st_tags(
    label='Enter Tags:',
    text='Press enter to add more tags',
    value=['Finance', 'ML', 'Credit'],
    suggestions=['Streamlit', 'Python', 'Data Science'],
    maxtags=5,
    key='tags'
)  # Display and manage tags
st.write('Selected Tags:', tags)  # Display selected tags

# Pandas Profiling
df = pd.DataFrame({
    'Credit Score': [700, 750, 800, 850, 900],
    'Probability': [0.1, 0.3, 0.5, 0.7, 0.9]
})

profile = ProfileReport(df, title='Credit EDA Report')  # Generate profiling report
st.title("Credit Score Prediction EDA")  # Title for EDA section
st_profile_report(profile)  # Display profiling report

# Autocomplete
options = ['apple', 'banana', 'cherry', 'date', 'elderberry']
selected_option = st.selectbox('Select a fruit:', options)  # Dropdown for autocomplete
st.write('You selected:', selected_option)  # Display selected option

# Timeline (Commented out for now)
# import streamlit_vega_lite as st_vega_lite
# timeline_data = {
#     "data": {
#         "values": [
#             {"task": "Task 1", "start": "2024-01-01", "end": "2024-01-15"},
#             {"task": "Task 2", "start": "2024-01-16", "end": "2024-01-31"},
#         ]
#     },
#     "mark": "bar",
#     "encoding": {
#         "x": {"field": "start", "type": "temporal"},
#         "x2": {"field": "end"},
#         "y": {"field": "task", "type": "nominal"},
#     }
# }
# st_vega_lite.altair_chart(timeline_data, use_container_width=True)  # Display timeline chart

# Lottie Animation
url = "https://lottiefiles.com/free-animation/animation-project-WyMeBJCYDJ"  # Lottie file URL
try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    animation_data = response.json()  # Parse JSON data
except requests.exceptions.RequestException as e:
    st.error(f"Request error: {e}")  # Display error for request issues
    animation_data = None
except ValueError as e:
    st.error(f"JSON decoding error: {e}")  # Display error for JSON issues
    animation_data = None

if animation_data:
    st_lottie(animation_data, height=200, width=300)  # Display Lottie animation
else:
    st.write("Failed to load animation.")  # Inform about animation loading failure
