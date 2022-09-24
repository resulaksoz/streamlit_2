
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


html_temp = """
<div style="background-color:green;padding:2.5px">
<h1 style="color:white;text-align:center;">Car Price Prediction Project </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.write('Auto Scout data which using for this project, scraped from the on-line car trading company(https://www.autoscout24.com) in 2019, contains many features of 8 different car models. In this project, you will use the data set which is already preprocessed and prepared for algorithms .')

html_temp = """
<p style="font-weight:bold;">In this project, we will estimate the prices of cars using lasso regression algorithms.</p>



"""
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown('##### In this project, we are just interested in machine learning so we have already handled missing values, and outliers and made feature selections.')

st.markdown('### 1. Import Modules and Load Data ')
st.write('Firstly we will import the necessary libraries.')

st.code('import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import OneHotEncoder\nfrom scipy.stats import skew\nfrom sklearn.model_selection import cross_validate')
st.write('In this part, we will look at some data examples.')

df=pd.read_csv('car_price_prediction.csv')
df.rename(columns={'make_model':'Car Model','hp_kW':'HP','km':'KM','age':'Age','Gearing_Type':'Gearing Type'},inplace=True)
st.table(df.sample(10))

st.write('We separated the target feature from dataframe.')

st.code('X = df.drop(columns = ["price"])\ny = df.price')

st.write('We split the data for test and training.')

st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2')

st.write('We are writing our own functions to make our process faster.')

st.code('def trans_2(X_train, X_test):\n\tcat = X_train.select_dtypes("object").columns\n\tcat = list(cat)\n\tenc = OneHotEncoder(handle_unknown="ignore", sparse=False)\n\tX_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index,\n\t\tcolumns = enc.get_feature_names_out(cat))\n\tX_test_cat  = pd.DataFrame(enc.transform(X_test[cat]), index = X_test.index,\n\t\tcolumns = enc.get_feature_names_out(cat))\n\tX_train = X_train_cat.join(X_train.select_dtypes("number"))\n\tX_test = X_test_cat.join(X_test.select_dtypes("number"))\n\treturn X_train, X_test')

st.code('def train_val(model, X_train, y_train, X_test, y_test):\n\ty_pred = model.predict(X_test)\n\ty_train_pred = model.predict(X_train)\n\tscores = {"train": {"R2" : r2_score(y_train, y_train_pred),\n\t"mae" : mean_absolute_error(y_train, y_train_pred),\n\t"mse" : mean_squared_error(y_train, y_train_pred),\n\t"rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},\n\t"test": {"R2" : r2_score(y_test, y_pred),\n\t"mae" : mean_absolute_error(y_test, y_pred),\n\t"mse" : mean_squared_error(y_test, y_pred),\n\t"rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}\n\treturn pd.DataFrame(scores)')

st.write('We dealt with categorical data.')

st.code('X_train, X_test = trans_2(X_train, X_test)')

st.write('We scaled the data.')

st.code('scaler = MinMaxScaler()\nscaler.fit(X_train)\n\nX_train_scaled = scaler.transform(X_train)\nX_test_scaled = scaler.transform(X_test)')

st.write('We imported the lasso model, we trained and with the function we write we looked at the training results')

st.code('from sklearn.linear_model import Lasso\nlasso_model = Lasso()\nlasso_model.fit(X_train_scaled, y_train)\ntrain_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)')

st.markdown('###### Traning Results')
img_1=Image.open('proje_image-1.png')
st.image(img_1)

st.write('Our training r2 and test r2 scores are close to each other, from this, we can conclude that there is no underfit or overfit so our model is successful.')

st.markdown('## Cross Validation')

st.code("model = Lasso()\nscores = cross_validate(model, X_train_scaled, y_train,\n\t\tscoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],\n\t\tcv=10)")

st.code('scores = pd.DataFrame(scores, index = range(1, 11))\nscores.iloc[:,2:].mean()')
st.markdown('###### Cross Validation Results')
img_2=Image.open('proje_image-2.png')
st.image(img_2)

st.write('Cross-validation results are also close to training results.')

st.markdown('## Gridsearch')

st.code('from sklearn.model_selection import GridSearchCV')

st.code('alpha_space = np.linspace(0.01, 100, 100)')

st.code("lasso_model = Lasso()\n\nparam_grid = {'alpha':alpha_space}\n\nlasso_grid = GridSearchCV(estimator=lasso_model,\n\t\tparam_grid=param_grid,\n\t\tscoring='neg_root_mean_squared_error',\n\t\tcv=10,\n\t\tn_jobs=-1")

st.code("lasso_grid.fit(X_train_scaled,y_train)")

st.code("lasso_grid.best_params_\n{'alpha': 0.01}")

st.code("train_val(lasso_grid, X_train_scaled, y_train, X_test_scaled, y_test)")

img_3=Image.open('proje_image-3.png')
st.image(img_3)

st.write('We use Gridsearch to find optimal hyperparameters.')


st.markdown('## Pipeline')

st.write('We finished the necessary preparation and found optimum parameters now we can use the pipeline to create our model.')

st.code('cat = X.select_dtypes("object").columns\ncat = list(cat)')

st.code('from sklearn.compose import make_column_transformer\nfrom sklearn.preprocessing import OneHotEncoder\n\ncolumn_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat),\n\t\t\tremainder=MinMaxScaler())')


st.code('from sklearn.linear_model import Lasso\nfrom sklearn.pipeline import Pipeline\n\noperations = [("OneHotEncoder", column_trans), ("Lasso", Lasso(alpha=0.01))]\n\npipe_model = Pipeline(steps=operations)\n\npipe_model.fit(X, y)')

st.markdown('###### We created a pipeline model, from now we can make a prediction. ')

st.sidebar.markdown('# Car Features')
car_model=st.sidebar.selectbox('Car models',['Audi A3','Audi A1','Opel Insignia','Opel Astra','Opel Corsa','Renault Clio','Renault Espace','Renault Duster'])


car_age=st.sidebar.radio('Please select a age of car',(0,1,2,3))



car_hp=st.sidebar.slider("Please select the engine's hp ",min_value=40,max_value=240,value=90,step=10)


car_km=st.sidebar.number_input('KM of Car',min_value=0,max_value=300000)

car_gear=st.sidebar.radio('Please select gearing type:',('Manual','Automatic','Semi-automatic'))

my_dict={'Car Model':car_model,'Age':car_age,'HP':car_hp,'KM':car_km,'Gearing Type':car_gear}

df_1=pd.DataFrame.from_dict([my_dict])

st.table(df_1)

import pickle
filename='my_model'
model=pickle.load(open(filename,'rb'))

if st.button('Predict'):
    pred=model.predict(df_1)
    st.success(pred[0])