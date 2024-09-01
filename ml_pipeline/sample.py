# Continue with additional sections as needed
import streamlit as st

st.text("Hello, Streamlit!")
st.markdown("# Hello, Streamlit!")
st.header("This is a Header")
st.subheader("This is a Subheader")
st.title("This is a Title")

name = st.text_input("Enter your name")
bio = st.text_area("Tell us about yourself")

if st.button("Click me"):
    st.write("Button clicked!")
st.download_button("Download CSV", data="data,text", file_name="file.csv")
agree = st.checkbox("I agree")

option = st.radio("Choose an option", ["Option 1", "Option 2", "Option 3"])

choice = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])

choices = st.multiselect("Select options", ["Option 1", "Option 2", "Option 3"])
value = st.slider("Pick a value", min_value=0, max_value=100, value=50)
num = st.number_input("Enter a number", min_value=0, max_value=100, value=50)
date = st.date_input("Pick a date")

time = st.time_input("Pick a time")
file = st.file_uploader("Upload a file")
progress = st.progress(0)
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
st.dataframe(df)
st.table(df)
st.line_chart([1, 2, 3, 4])
st.bar_chart({'A': [1, 2, 3], 'B': [4, 5, 6]})
st.area_chart([1, 2, 3, 4])
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
st.pyplot()
st.sidebar.title("Sidebar Title")
st.sidebar.selectbox("Choose an option", ["Option 1", "Option 2"])
if st.checkbox("Show details"):
    st.write("Details are shown here.")
with st.container():
    st.write("This is inside the container")
    st.bar_chart({"data": [1, 2, 3, 4, 5]})

st.write("This is outside the container")
col1, col2, col3 = st.columns(3)

col1.write("Column 1")
col2.write("Column 2")
col3.write("Column 3")

col1.image("https://via.placeholder.com/150", caption="Image 1")
col2.image("https://via.placeholder.com/150", caption="Image 2")
col3.image("https://via.placeholder.com/150", caption="Image 3")
with st.expander("Click to expand"):
    st.write("This content is hidden by default. Click to see more details.")
    st.line_chart({"data": [1, 3, 2, 4, 5]})

#Tabs section allows to display different content content under different task 
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.write("This is Tab 1 content")

with tab2:
    st.write("This is Tab 2 content")

with tab3:
    st.write("This is Tab 3 content")

#forms sectio used to group multiple widgets together 
with st.form("my_form"):
    name = st.text_input("Name")
    age = st.slider("Age", 18, 100)
    submitted = st.form_submit_button("Submit")

if submitted:
    st.write(f"Name: {name}, Age: {age}")

#MAterics displays 

st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
st.metric(label="Humidity", value="50%", delta="-2%")

#Selectbox is a dropdown list from which users can choose an item
option = st.selectbox(
    'Which number do you like best?',
    [1, 2, 3, 4, 5])

st.write('You selected:', option)

#File uploader 
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = uploaded_file.read()
    st.write(data)

#Date brithday 
from datetime import date
d = st.date_input(
    "When's your birthday",
    date(2024, 1, 1))
#plotly for interactive visualisation
import plotly.express as px
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'Credit Score': [700, 750, 800, 850, 900],
    'Probability': [0.1, 0.3, 0.5, 0.7, 0.9]
})

# Plotly Chart
fig = px.scatter(df, x='Credit Score', y='Probability', title='Credit Score vs. Default Probability')

st.plotly_chart(fig)

#Customizable CSS 

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
)

st.write("This Streamlit app has custom styling.")

#Ag grid for interactive data tables 
from st_aggrid import AgGrid

# Sample Data
df = pd.DataFrame({
    'Customer ID': [1, 2, 3, 4],
    'Credit Score': [700, 650, 800, 720],
    'Default Probability': [0.1, 0.4, 0.05, 0.2]
})

# Displaying DataFrame with AgGrid
AgGrid(df)

st.write('Your birthday is:', d)

#Interactive maps using folium
import folium 
from streamlit_folium import folium_static 

m=folium.Map(location=[45.4567,-123.4567],zoom_start=13)

#Add a marker 
folium.Marker([45.4567,-122.4567],tooltip='Portland').add_to(m)

#Display the map
folium_static(m)

#Streamlit Cards for displaying summarize infromation metrics or other key performance KPIS
from streamlit_card import card

# Display a card
card(
    title="Credit Score",
    text="Your estimated credit score is 750.",
    image="https://via.placeholder.com/150",
    url="https://credit-score.com"
)

#TAgs can be useful interactive component for adding or categorising input dynamically
from streamlit_tags import st_tags

# Display tags
tags = st_tags(
    label='Enter Tags:',
    text='Press enter to add more tags',
    value=['Finance', 'ML', 'Credit'],
    suggestions=['Streamlit', 'Python', 'Data Science'],
    maxtags=5,
    key='tags'
)

st.write('Selected Tags:', tags)

#pandas-profiling can be used with Streamlit for an interactive exploration of data. This is extremely useful for EDA
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st

# Sample Data
df = pd.DataFrame({
    'Credit Score': [700, 750, 800, 850, 900],
    'Probability': [0.1, 0.3, 0.5, 0.7, 0.9]
})

# Generate the profiling report
profile = ProfileReport(df, title='Credit EDA Report')

# Display the report in Streamlit
st.title("Credit Score Prediction EDA")
st_profile_report(profile)

#Streamlit Autocomplete provied a type ahead search bar

# Options for autocomplete
options = ['apple', 'banana', 'cherry', 'date', 'elderberry']
selected_option = st.selectbox('Select a fruit:', options)
st.write('You selected:', selected_option)

#Streamlit timeline is usefull for showcasing timelines in a visual format 
#import streamlit_vega_lite as st_vega_lite

# Example data for the timeline
#timeline_data = {
##    "data": {
#       "values": [
#            {"task": "Task 1", "start": "2024-01-01", "end": "2024-01-15"},
#            {"task": "Task 2", "start": "2024-01-16", "end": "2024-01-31"},
#       ]
#    },
#    "mark": "bar",
#    "encoding": {
#        "x": {"field": "start", "type": "temporal"},
#        "x2": {"field": "end"},
#        "y": {"field": "task", "type": "nominal"},
#    }
#}

#st_vega_lite.altair_chart(timeline_data, use_container_width=True)

#Streamlit lottie for animations make the interface more dynamic and engaging.They provide visual feedback
from streamlit_lottie import st_lottie
import requests

# Lottie file URL
url = "https://lottiefiles.com/free-animation/animation-project-WyMeBJCYDJ"

# Fetch animation data
try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    animation_data = response.json()  # Parse the response as JSON
except requests.exceptions.RequestException as e:
    st.error(f"Request error: {e}")
    animation_data = None
except ValueError as e:
    st.error(f"JSON decoding error: {e}")
    animation_data = None

# Display animation
if animation_data:
    st_lottie(animation_data, height=200, width=300)
else:
    st.write("Failed to load animation.") 