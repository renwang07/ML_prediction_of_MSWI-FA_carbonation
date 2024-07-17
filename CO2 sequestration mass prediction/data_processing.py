import pandas as pd
from sklearn import preprocessing
import shap
from sklearn.model_selection import train_test_split

def read_data(file):
    data = pd.read_excel(file, 0)
    data = data.iloc[:, :]
    return data

def scaler_data_processing(file):
    data = read_data(file)
    data_df = pd.DataFrame(data, columns=data.columns)
    feature_value = data_df.iloc[:, :-1]
    raw_feature_value = data_df.iloc[:, :-1]
    raw_feature_value = pd.DataFrame(raw_feature_value)
    target_value = data_df.iloc[:, -1]
    target_value = pd.DataFrame(target_value)
    cols = feature_value.columns
    raw_training_feature_value, raw_test_feature_value, raw_training_target_value, raw_test_target_value = train_test_split(raw_feature_value, target_value,
                                                                                            test_size=0.2, random_state=23)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # feature_value = min_max_scaler.fit_transform(feature_value)
    zscore_scaler = preprocessing.StandardScaler()
    feature_value = zscore_scaler.fit_transform(feature_value)
    feature_value = pd.DataFrame(feature_value, columns=cols)
    training_feature_value, test_feature_value, training_target_value, test_target_value = train_test_split(feature_value, target_value,
                                                                                            test_size=0.2, random_state=23)
    raw_feature_train_summary = shap.kmeans(raw_training_feature_value, 10)
    return (training_feature_value, training_target_value, test_feature_value, test_target_value, raw_feature_train_summary, data_df,
            raw_training_feature_value,raw_training_target_value,raw_test_feature_value,raw_test_target_value)
