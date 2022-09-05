import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import robust_scale


# Title
st.write("""
    # Diamond Price Prediction
""")

st.write('---')

# Model path
MODEL_PATH = 'model.pkl'

# Sidebar
st.sidebar.header('Specify Input')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Upload file or user input
if uploaded_file is not None:
    info = pd.read_csv(uploaded_file)
else:
    def user_input():
        carat = st.sidebar.slider('Carat', 0.2000, 5.0100, 0.7979)
        cut = st.sidebar.selectbox('Cut', ('Ideal','Premium','Good',' Very Good','Fair'))
        color = st.sidebar.selectbox('Color', ('E','I','J','H','F','G','D'))
        clarity = st.sidebar.selectbox('Clarity', ('SI2','SI1','VS1','VS2','VVS2','VVS1','I1','IF'))
        depth = st.sidebar.slider('Depth', 43.000, 79.000, 61.749)
        table = st.sidebar.slider('Table', 43.000, 95.000, 57.457)
        xx = st.sidebar.slider('X', 0.000, 10.740, 5.7311)
        yy = st.sidebar.slider('Y', 0.000, 58.900, 5.734)
        zz = st.sidebar.slider('Z', 0.000, 31.800, 3.538)

        # Create dictionary with all inputs
        data = {
            'carat':carat,
            'cut':cut,
            'color':color,
            'clarity':clarity,
            'depth':depth,
            'table':table,
            'x':xx,
            'y':yy,
            'z':zz }

        info = pd.DataFrame(data, index=[0])
        return info

info = user_input()

# Load clean data
diamond = pd.read_csv('clean_diamonds.csv')


# Intercorrelation Matrix Heatmap
st.subheader('Intercorrelation Matrix Heatmap')
fig = plt.figure(figsize=(12,10))
aux = diamond.drop(columns=['index'])
sns.heatmap(aux.corr(), annot=True, cmap='GnBu')
st.pyplot(fig)

st.write('---')

st.subheader('Model Prediction')

# Drop unnecessary columns
diamonds = diamond.drop(columns=['price','index'])
df = pd.concat([info,diamonds],axis=0)
st.write(df[:1])    # Show the first line

# Predict button
if st.button('Predict'):

    ## Preprocessing 
    df.drop(columns=['table','depth'], inplace=True)

    # Encode categorical variables
    encode = ['cut','color','clarity']

    # Apply LabelEncoder
    le = LabelEncoder()
    for col in encode:
        df['cut'] = le.fit_transform(df['cut'])
        df['color'] = le.fit_transform(df['color'])
        df['clarity'] = le.fit_transform(df['clarity'])

    # Apply robust_scale for all numeric values
    df = robust_scale(df)
    df = df[:1]

    # Load model
    model=''
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Prediction
    pred = model.predict(df)
    st.write('$',pred[0])
    st.write('---')
