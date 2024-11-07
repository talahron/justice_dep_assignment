import numpy as np
import pandas as pd
import os
import pickle
import json
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def get_data():
    # reading tables
    path = r'C:\Users\Sivan\Desktop\עבודה\משימת ביית משרד המשפטים'
    data_path = os.path.join(path, 'test_data_engineer.csv')
    target_path = os.path.join(path, 'test_data_engineer_aggregated_case_result.csv')
    df = pd.read_csv(data_path, index_col=False, encoding='ISO-8859-8')
    target_df = pd.read_csv(target_path, index_col=False, encoding='ISO-8859-8')

    # adding target variable
    df = df.merge(target_df, how='inner', on='tnufa_endreasonName')
    df.rename(columns={'result_type': 'Y'}, inplace=True)

    print('Finished reading data from the local computer')
    return (df)


def remov_duplicates_and_missing_values(df):
    print(f'Original size {df.shape}')
    df.drop_duplicates(inplace=True)
    print(f'After removing duplicates {df.shape}')

    # In case there are duplicate values in the unique_id field, I leave the row with the latest moj_eventdate.
    # In case all dates are the same I take the row with the least missing values. If there is no difference, I take a random one.
    df['null_count'] = df.isnull().sum(axis=1)
    df.sort_values(by=['unique_id', 'moj_eventdate', 'null_count'], ascending=[True, False, True], inplace=True)
    df = df.drop_duplicates(subset='unique_id', keep='first')
    print(f'After reducing duplicates in the unique_id field {df.shape}')

    # if a particular field has missing values above  threshold - remove the entire field.
    threshold = 0.6 * len(df)
    missing_counts = df.isnull().sum()
    cols_to_keep = missing_counts[missing_counts <= threshold].index
    df = df[cols_to_keep]
    print(f'After dropping columns with lots of NULL values {df.shape}')

    # checks if there are fields that have the same value
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    print(f'After dropping columns with the same value {df.shape}')
    return (df)


def feature_engineering(df):
    # creating new fields from date fields
    df['moj_courtcaseopendate'] = pd.to_datetime(df['moj_courtcaseopendate'], errors='coerce', format='%d/%m/%Y')
    df['moj_eventdate'] = pd.to_datetime(df['moj_eventdate'], errors='coerce')
    df['days_from_event_to_open'] = (df['moj_courtcaseopendate'] - df[
        'moj_eventdate']).dt.days  # how long has it been since the lawsuit began?
    df['moj_courtcaseopendate_year'] = df['moj_courtcaseopendate'].dt.year
    df['moj_eventdate_year'] = df['moj_eventdate'].dt.year

    # TODO: Complete missing values from one variable by another variable.
    # For example - tnufa_essencegroupidName, tnufa_claimessenceidName, tnufa_seconderyclaimessenceidName
    # TODO: Check correlations and create new variables
    print(f'Before removing additional fields {df.shape}')
    df.drop(columns='null_count', inplace=True)  # May create noise
    df.drop(columns='tnufa_endreasonName', inplace=True)  # Y-field
    df.drop(columns=['moj_courtcaseopendate', 'moj_eventdate'],
            inplace=True)  # I created calculation fields from these fields.
    df.drop(columns='unique_id', inplace=True)  # Does not contribute to the model
    df.drop(columns='a_enddate', inplace=True)  # doesn't make sense because if I know the end date the lawsuit outcome.
    df.drop(columns='moj_stateresponsibilitylist', inplace=True)  # Correlative to the field - 'moj_damagelist'
    print(f'After removing additional fields {df.shape}')

    df.to_csv(r'C:\Users\Sivan\PycharmProjects\Ministry_of_Justice_home_mission\file1.csv')
    # handling missing values
    """
    Important !!!
    This is not an approach I would take in real life.
    Basically, you need to understand the data at a high level and do intelligent missing value completion.

    For example:
    There may be cases where the NULL value is important.
    Sometimes it is not possible to complete values from the most common value (this can create noise).
    Sometimes it is possible to complete information from one field from another and create a variable from both.
    The strategy of what value I want to complete depends on the distribution of the field and business understanding.
    ...
    """

    for column in df.columns:
        # if the column is numeric and has missing values, we fill them with the median (if it's a continuous variable)
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].median(), inplace=True)

        # If the column is categorical and has missing values, we can fill them with the mode (most frequent value)
        elif df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
    return (df)


def preprocess_data(df):
    # Splits into X and Y
    X = df.drop(columns=['Y'])
    y = df['Y'].map({"נדחה": 0, "פשרה": 1, "התקבל": 2}).values

    # Identify different types of columns (binary, low cardinality, categorical, continuous) and convert accordingly.
    binary_cols = [col for col in X.columns if X[col].nunique() == 2 and set(X[col].unique()) <= {0, 1}]
    # keep low cardinality columns as numeric to preserve ordinal relationships
    low_cardinality_cols = [col for col in X.select_dtypes(include=[np.number]).columns if
                            X[col].nunique() <= 5 and col not in binary_cols]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    continuous_cols = [col for col in X.select_dtypes(include=[np.number]).columns if
                       col not in binary_cols + low_cardinality_cols]

    # converts categorical values to dummy normalizes continuous values
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X[continuous_cols] = MinMaxScaler().fit_transform(X[continuous_cols])

    return X, y


def select_top_features(X, y, num_features):
    """
    In a real case I would effectively check which num_features bring the best results.
    Now I ran manually a few times and number 20 brought good results.
    """
    # Select the top features using ExtraTreesClassifier
    model = ExtraTreesClassifier(n_estimators=50)
    model.fit(X, y)
    selector = SelectFromModel(model, max_features=num_features, prefit=True)
    X_new = selector.transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected top {num_features} features: {selected_features}")
    return X_new, y


def train_and_evaluate_models(X, y, X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        print(f"{name} - Test F1 Score: {f1:.4f}, ROC AUC Score: {roc_auc:.4f}")
        """
        Logistic Regression - Test F1 Score: 0.7443, ROC AUC Score: 0.8476
        Decision Tree - Test F1 Score: 0.7108, ROC AUC Score: 0.7020
        Random Forest - Test F1 Score: 0.7506, ROC AUC Score: 0.8856
        XGBoost - Test F1 Score: 0.7310, ROC AUC Score: 0.8448

        selected model - RandomForestClassifier.
        I continue without greedy search for the best hyperparameters
        """

    # fit the selected model to the entire dataset
    model = RandomForestClassifier()
    model.fit(X, y)

    return model


def save_model(model, filename='model.pkl'):
    # Save the model to the current folder located in path
    path = os.path.join(os.getcwd(), filename)
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def save_random_rows_as_json(X, y, num_rows=5):
    """
    This function generates JSONS for future testing of the application
    """
    random_indices = np.random.choice(X.shape[0], num_rows, replace=False)
    for idx in random_indices:
        row_data = X[idx]
        y_value = y[idx]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{y_value}_{timestamp}.json"
        path = os.path.join(os.getcwd(), filename)
        row_dict = {f"feature_{i}": value for i, value in enumerate(row_data)}
        with open(path, 'w') as json_file:
            json.dump(row_dict, json_file)
        print(f"Saved row {idx} as {filename}")


def save_random_rows_as_csv(X, y, num_rows=5):
    """
    This function generates JSONS for future testing of the application
    """
    random_indices = np.random.choice(X.shape[0], num_rows, replace=False)
    for idx in random_indices:
        row_data = X[idx]
        y_value = y[idx]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{y_value}_{timestamp}.csv"
        path = os.path.join(os.getcwd(), filename)
        row_dict = {f"feature_{i}": value for i, value in enumerate(row_data)}
        row_df = pd.DataFrame([row_dict])
        row_df.to_csv(path, index=False)
        print(f"Saved row {idx} as {filename}")


if __name__ == "__main__":
    df = get_data()
    df = remov_duplicates_and_missing_values(df)
    df = feature_engineering(df)
    X, y = preprocess_data(df)

    X, y = select_top_features(X, y, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = train_and_evaluate_models(X, y, X_train, y_train, X_test, y_test)

    save_model(model)
    print("Model saved to 'model.pkl'")

    # Save 5 random rows as JSON files
    save_random_rows_as_json(X, y)
    # Save 5 random rows as CSV files
    save_random_rows_as_csv(X, y)
pass
