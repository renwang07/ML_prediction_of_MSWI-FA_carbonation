import pandas as pd
import shap
from data_processing import scaler_data_processing
from model import DT_predict, MLP_predict, RF_predict, SVR_predict, GB_predict, KNN_predict, XGB_predict, LGBM_predict
from figure_output import SHAP_plot, predict_compare_plot, other_SHAP_plot, test_predict_plot
from data_statistics import violin_plot, heat_map
import os
import joblib
import warnings

warnings.filterwarnings("ignore")
SHAP_plot_save_path = './SHAP_plot2/'
feature_SHAP_plot_path = 'All feature SHAP scatter plots/'
heat_map_path = "heatmap/"
dependence_plot_save_path = 'SHAP dependence_plot/'
dataset_file = './Dataset1/dataset1.xlsx'

def result_plot(model_predict, raw_feature_train_summary, train_feature, train_target, train_data_predict, test_feature, test_target, test_data_predict,raw_train_feature, raw_train_target, raw_test_feature, raw_test_target):
    # test_predict_plot(test_data_predict, test_target)
    # predict_compare_plot(train_target, train_data_predict, test_target, test_data_predict)
    shap.initjs()
    SHAP_plot(model_predict, raw_feature_train_summary, raw_train_feature)
    # other_SHAP_plot(model_predict, raw_feature_train_summary, raw_train_feature,raw_train_target,raw_test_feature,raw_test_target)

def run_model():
    train_feature, train_target, test_feature, test_target, raw_feature_train_summary, data_df, raw_train_feature, raw_train_target, raw_test_feature, raw_test_target = scaler_data_processing(dataset_file)
    # train_feature, train_target, feature_train_summary = train_data_value(train_data_file)
    # test_feature, test_target = test_data_value(test_data_file)
    # violin_plot(data_df)
    # input_feature = pd.concat([train_feature, test_feature], ignore_index=True)
    # heat_map(input_feature)
    # heat_map(data_df)

    # model_predict, train_data_predict, test_data_predict, DT_model = DT_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict, MLP_model = MLP_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict, RF_model = RF_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict, KNN_model = KNN_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict,SVR_modle = SVR_predict(train_feature, train_target, test_feature, test_target)
    model_predict, train_data_predict, test_data_predict, GB_model = GB_predict(raw_train_feature, raw_train_target,raw_test_feature,raw_test_target)
    # model_predict, train_data_predict, test_data_predict, XGB_model = XGB_predict(train_feature,train_target,test_feature,test_target)
    # model_predict, train_data_predict, test_data_predict, LGBM_model = LGBM_predict(train_feature,train_target,test_feature,test_target)

    # joblib.dump(model, 'trained_model.pkl' )
    # joblib.dump(trained_model, 'trained_model1.pkl' )
    result_plot(model_predict, raw_feature_train_summary, train_feature, train_target, train_data_predict,test_feature, test_target, test_data_predict,raw_train_feature, raw_train_target, raw_test_feature, raw_test_target)

# Make a directory to save result picture
def make_directory():
    try:
        if not os.path.exists(SHAP_plot_save_path):
            os.mkdir(SHAP_plot_save_path)
        if not os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path):
            os.mkdir(SHAP_plot_save_path + feature_SHAP_plot_path)
        if not os.path.exists(SHAP_plot_save_path + heat_map_path):
            os.mkdir(SHAP_plot_save_path + heat_map_path)
        if not os.path.exists(SHAP_plot_save_path + dependence_plot_save_path):
            os.mkdir(SHAP_plot_save_path + dependence_plot_save_path)
        if os.path.exists(SHAP_plot_save_path) and os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path)\
                and os.path.exists(SHAP_plot_save_path + heat_map_path) and os.path.exists(SHAP_plot_save_path + dependence_plot_save_path):
            print("The SHAP plot directory already exists.")
        else:
            print("Successfully created folder, path is: " + SHAP_plot_save_path)
    except Exception as mkdir_error:
        print("Make Directory Error: " + str(mkdir_error))

def Initialization():
    print("Initialize...")
    make_directory()

if __name__ == "__main__":
    Initialization()
    run_model()