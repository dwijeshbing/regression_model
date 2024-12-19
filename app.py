import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def configure_page():
    st.set_page_config(page_title='Regression Model Generator App', layout='wide')


def display_header():
    st.write("""
    # Regression Model Generator

    In this implementation, select either **Random Forest** or **Linear Regression** to build your regression model.

    Try adjusting the hyperparameters!
    """)


def sidebar_configurations():
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
        """)
    
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    model_type = st.sidebar.selectbox('Select Model', ['Random Forest', 'Linear Regression'])
    
    params = {}
    
    if model_type == 'Random Forest':
        with st.sidebar.subheader('2.1. Learning Parameters'):
            params = {
                "n_estimators": st.sidebar.slider('Number of estimators (n_estimators)', 10, 1000, 100, 100),
                "max_features": st.sidebar.select_slider('Max features (max_features)', options=['sqrt', 'log2', None]),
                "min_samples_split": st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1),
                "min_samples_leaf": st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1),
            }
        
        with st.sidebar.subheader('2.2. General Parameters'):
            params.update({
                "random_state": st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1),
                "criterion": st.sidebar.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'poisson', 'friedman_mse']),
                "bootstrap": st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False]),
                "oob_score": st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True]),
                "n_jobs": st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1]),
            })
    
    elif model_type == 'Linear Regression':
        with st.sidebar.subheader('2.1. Linear Regression Parameters'):
            params = {
                "fit_intercept": st.sidebar.selectbox('Fit Intercept', [True, False]),
                "copy_X": st.sidebar.selectbox('Copy X', [True, False]),
            }
    
    return uploaded_file, split_size, params, model_type


def build_model(df, split_size, params, model_type):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    if model_type == 'Linear Regression':
        st.markdown('**Scaling features for Linear Regression...**')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if model_type == 'Random Forest':
        st.markdown('**Building Random Forest Model...**')
        with st.spinner('Building the model... This might take a moment.'):
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, Y_train)
        model = rf
    else:
        st.markdown('**Building Linear Regression Model...**')
        with st.spinner('Building the model... This might take a moment.'):
            lr = LinearRegression(**params)
            lr.fit(X_train, Y_train)
        model = lr

    st.success('Model built successfully!')

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = model.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = model.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(model.get_params() if model_type == 'Random Forest' else 'No additional parameters for Linear Regression.')

    model_file = 'model.pkl'
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

    st.download_button(label="Download Trained Model", data=open(model_file, 'rb'), file_name=model_file, mime='application/octet-stream')


def load_example_dataset():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    Y = pd.Series(housing.target, name='response')
    df = pd.concat([X, Y], axis=1)

    st.markdown('The California housing dataset is used as the example.')
    st.write(df.head(5))
    return df


def main():
    configure_page()
    display_header()
    
    uploaded_file, split_size, params, model_type = sidebar_configurations()
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        
        if st.button("Build Model"):
            build_model(df, split_size, params, model_type)
    else:
        st.info('Awaiting for CSV file to be uploaded. If you would like to use an example dataset, click below.')
        if st.button('Press to use Example Dataset'):
            df = load_example_dataset()
            build_model(df, split_size, params, model_type)

if __name__ == "__main__":
    main()