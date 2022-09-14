# Diamond Price Project
## EDA - Data Visualization - Machine Learning

![Jupyter-Notebook](https://jupyter.org/assets/logos/rectanglelogo-greytext-orangebody-greymoons.svg)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

### Problem
This dataset contains the prices and other attributes of almost 54,000 diamonds. I done an EDA, Data Visualization and Machine Learning model to predict diamond's price

## Features 👀
1. Price: price in US dollars ($326 to $18,823)
3. Carat: weight of the diamond (0.2 to 5.01)
4. Cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
5. Color: diamond colour, from J (worst) to D (best)
6. Clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
7. X: length in mm (0 to 10.74)
8. Y: width in mm (0 to 58.9)
9. Z: depth in mm (0 to 31.8)
10. Depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) --> (43 to 79)
11. Table: width of top of diamond relative to widest point (43 to 95)

## Tech 🛡

This project contains the following libraries:
- [Pandas](https://pandas.pydata.org/) - Data Manipulation
- [Matplotlib](https://matplotlib.org/stable/index.html) - Visualization
- [Seaborn](https://seaborn.pydata.org/index.html) - Visualization
- [Sklearn](https://scikit-learn.org/stable/) - Machine Learning (Model)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/master/index.html#) - AutoML (Model Selection) 
- [Streamlit](https://streamlit.io/) - Framework to create web apps 

## Installl 🛠

If you want to run locally on your computer, this project requires Linux, Jupyter Notebook in your preferred IDE and [Python>=3.9.x](https://www.python.org/).

After cloning or downloading it, please open the terminal (in same path where local repository is located) and run this command:  

```
pip install -r requirements.txt
```
This will download and install the requirements needed. In order to install auto-sklearn, please read the official [documentation](https://automl.github.io/auto-sklearn/master/installation.html).

## Let's run it 🦉
Please open the terminal (in same path where local repository is located) and run this command:  

```
streamlit run app.py
```
This command run the web app in your browser
## Demo 🪐

[Diamond-App](https://huggingface.co/spaces/francoalarcon00/diamond-app)

## Step by Step Analysis

1. Run data_cleaning.ipynb --> Clean process

2. Run glass_analysis.ipynb --> Analysis process and EDA

3. Run model_selection.ipynb --> Model selection and looking hyperparameters process

4. Run model.ipynb --> Modeling, training, testing and results process
## Source

https://www.kaggle.com/datasets/shivam2503/diamonds

## Author 🖋
*Franco Nicolás Alarcón* 🤟
