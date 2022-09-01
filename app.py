import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import robust_scale
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools


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

diamond = pd.read_csv('clean_diamonds.csv')



# Data Visualization
if st.button('Data Visualization'):

    # CARAT
    st.header('CARAT')
    sns.set_theme(style='darkgrid')
    fig1 = plt.figure(figsize=(12,15))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='carat', color='dodgerblue', kde=True)
    plt.title('Carat distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='carat', y='price', color='dodgerblue')
    plt.title('Carat by price', fontdict={'fontsize':18})
    st.pyplot(fig1)

    st.write('---')

    # CUT
    st.header('CUT')
    fig2 = plt.figure(figsize=(12,15))
    plt.style.use('tableau-colorblind10')
    plt.subplot(2,2,1)
    sns.countplot(data=diamond, x='cut')
    plt.title('Cut countplot', fontdict={'fontsize':18})
    plt.subplot(2,2,2)
    size = [39.9, 25.5, 9, 22.3, 3.3]
    explode = (0.05, 0, 0, 0, 0)
    label = ['Ideal','Premium','Good',' Very Good','Fair']
    plt.pie(size ,labels=label, explode=explode, labeldistance=1.1, startangle=90, shadow=True, autopct='%1.1f%%')
    plt.title('Cut pieplot', fontdict={'fontsize':18})
    st.pyplot(fig2)
    fig2_ = plt.figure(figsize=(12,8))
    sns.barplot(data=diamond, x='cut', y='price')
    plt.title('Cut by price', fontdict={'fontsize':18})
    st.pyplot(fig2_)

    st.write('---')

    # COLOR 
    st.header('COLOR')
    fig3 = plt.figure(figsize=(12,15))
    plt.style.use('ggplot')
    plt.subplot(2,2,1)
    sns.countplot(data=diamond, x='color')
    plt.title('Color countplot', fontdict={'fontsize':18})  
    plt.subplot(2,2,2)
    size = [18, 10, 5, 15, 18, 21, 13]
    explode = (0, 0, 0, 0, 0, 0.05, 0)
    label = ['E','I','J','H','F','G','D']
    plt.pie(size ,labels=label, explode=explode, labeldistance=1.1, startangle=90, shadow=True, autopct='%1.1f%%')
    plt.title('Color pieplot', fontdict={'fontsize':18})
    st.pyplot(fig3)
    fig3_ = plt.figure(figsize=(12,8))
    sns.barplot(data=diamond, x='color', y='price')
    plt.title('Color by price', fontdict={'fontsize':18})
    st.pyplot(fig3_)

    st.write('---')

    # CLARITY
    st.header('CLARITY')
    fig4 = plt.figure(figsize=(12,15))
    plt.style.use('tableau-colorblind10')
    plt.subplot(2,2,1)
    sns.countplot(data=diamond, x='clarity')
    plt.title('Clarity countplot', fontdict={'fontsize':18})
    plt.subplot(2,2,2)
    size = [17, 24, 15, 22, 9, 7, 1, 5]
    explode = (0, 0.1, 0, 0, 0, 0, 0, 0)
    label = ['SI2','SI1','VS1','VS2','VVS2','VVS1','I1','IF']
    plt.pie(size ,labels=label, explode=explode, labeldistance=1.1, startangle=90, shadow=True, autopct='%1.1f%%')
    plt.title('Clarity pieplot', fontdict={'fontsize':18})
    st.pyplot(fig4)
    fig4_ = plt.figure(figsize=(12,8))
    sns.barplot(data=diamond, x='clarity', y='price')
    plt.title('Clarity by price', fontdict={'fontsize':18})
    st.pyplot(fig4_)

    st.write('---')

    # DEPTH
    st.header('DEPTH')
    fig5 = plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='depth', color='mediumblue', kde=True)
    plt.title('Depth distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='depth', y='price', color='mediumblue')
    plt.title('Depth by price', fontdict={'fontsize':18})
    st.pyplot(fig5)

    st.write('---')

    # Description about TABLE
    st.header('TABLE')
    fig6 = plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='table', color='darkgreen', kde=True, bins=50)
    plt.title('Table countplot', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='table', y='price', color='darkgreen')
    plt.title('Table by price', fontdict={'fontsize':18})
    st.pyplot(fig6)

    st.write('---')

    # Description about PRICE
    st.header('PRICE')
    fig7 = plt.figure(figsize=(12, 12))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='price', color='sandybrown', kde=True, bins=30)
    plt.title('Price distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.boxplot(data=diamond, x='price', color='sandybrown')
    plt.title('Price boxplot', fontdict={'fontsize':18})
    st.pyplot(fig7)

    st.write('---')

    # Description about X
    st.header('X')
    fig8 = plt.figure(figsize=(10,15))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='x', color='palevioletred', kde=True)
    plt.title('X distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='x', y='price', color='palevioletred')
    plt.title('X by price', fontdict={'fontsize':18})
    st.pyplot(fig8)

    st.write('---')

    # Description about Y
    st.header('Y')
    fig9 = plt.figure(figsize=(10,15))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='y', color='indigo', kde=True)
    plt.title('Y distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='y', y='price', color='indigo')
    plt.title('Y by price', fontdict={'fontsize':18})
    st.pyplot(fig9)

    st.write('---')

    # Description about Z
    st.header('Z')
    fig10 = plt.figure(figsize=(10,15))
    plt.subplot(2,1,1)
    sns.histplot(data=diamond, x='z', color='crimson', kde=True)
    plt.title('Z distribution', fontdict={'fontsize':18})
    plt.subplot(2,1,2)
    sns.scatterplot(data=diamond, x='y', y='price', color='crimson')
    plt.title('Z by price', fontdict={'fontsize':18})
    st.pyplot(fig10)

    st.write('---')

    st.header('Intercorrelation Matrix Heatmap')
    fig = plt.figure(figsize=(12,10))
    aux = diamond.drop(columns=['index'])
    sns.heatmap(aux.corr(), annot=True, cmap='GnBu')
    st.pyplot(fig)

st.write('---')

diamonds = diamond.drop(columns=['price','index'])
df = pd.concat([info,diamonds],axis=0)
st.write(df[:1])

if st.button('Predict'):

    df.drop(columns=['table','depth'], inplace=True)

    encode = ['cut','color','clarity']

    le = LabelEncoder()
    for col in encode:
        df['cut'] = le.fit_transform(df['cut'])
        df['color'] = le.fit_transform(df['color'])
        df['clarity'] = le.fit_transform(df['clarity'])


    df = robust_scale(df)
    df = df[:1]

    model=''
    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    pred = model.predict(df)
    st.write('$',pred[0])
    st.write('---')
