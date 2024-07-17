import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sort
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel

def MAPE(y, y_pre):
    y = np.array(y)
    y_pre = np.array(y_pre)
    return np.mean(np.abs((y_pre - y ) / y )) * 100

def SMAPE(y, y_pre):
    y = np.array(y)
    y_pre = np.array(y_pre)
    return 2.0 * np.mean(np.abs(y_pre - y) / (np.abs(y_pre) + np.abs(y))) * 100

def DT_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=False)
    param_grid = {
        'ccp_alpha': [0.001],
        'splitter': ['best'],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [3],
        'random_state': [8],
        'max_features': [25],
    }
    DT_model = DecisionTreeRegressor()
    grid_search = GridSearchCV(DT_model, param_grid, cv=kf, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    DT_model = grid_search.best_estimator_
    DT_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(DT_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(DT_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(DT_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = DT_model.predict(feature_train)
    test_predict = DT_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = DT_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = DT_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return DT_model.predict, train_predict, test_predict, DT_model

def MLP_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle= False)
    param_grid = {
                  "solver": ["adam"],
                  "max_iter": [300],
                  "activation": ["relu"],
                  "alpha": [0.75],
                  "hidden_layer_sizes": [(8, 3, 2)],
                  "learning_rate": ['adaptive'],
                  "random_state": [3],
                  "warm_start": [True],
                  "batch_size" : [10],
                  }
    MLP_model = MLPRegressor()
    # r2 = make_scorer(r2_score, greater_is_better = True)
    grid_search = GridSearchCV(MLP_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    MLP_model = grid_search.best_estimator_
    MLP_model.fit(feature_train, target_train)
    R2_train_score = MLP_model.score(feature_train, target_train)
    print("MLP R2 Train Scores: " + str(R2_train_score))
    # R2_train_score = r2_score(target_train, MLP_model.predict(feature_train))
    # print("MLP R2 Train Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = MLP_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = MLP_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return MLP_model.predict, train_predict, test_predict, MLP_model

def RF_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {
                  "n_estimators": [20],
                  "max_depth": [4],
                  "min_samples_split": [3],
                  "min_samples_leaf": [1],
                  "max_features": [25],
                  "random_state": [42],
                  }
    RF_model = RandomForestRegressor()
    grid_search = GridSearchCV(RF_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    RF_model = grid_search.best_estimator_
    RF_model.fit(feature_train, target_train)
    R2_train_score = RF_model.score(feature_train, target_train)
    print("RF R2 Train Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = RF_model.predict(feature_train)
    test_predict = RF_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = RF_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = RF_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return RF_model.predict, train_predict, test_predict, RF_model

def KNN_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {
                  'algorithm': ['kd_tree'],
                  'leaf_size': [5],
                  'weights': ['uniform'],
                  'n_neighbors': [2],
                  'p':[6],
                 }
    KNN_model = KNeighborsRegressor()
    grid_search = GridSearchCV(KNN_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    KNN_model = grid_search.best_estimator_
    KNN_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(KNN_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(KNN_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(KNN_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = KNN_model.predict(feature_train)
    test_predict = KNN_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = KNN_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = KNN_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return KNN_model.predict, train_predict, test_predict, KNN_model

def SVR_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    param_grid = {
        'kernel': ['rbf'],
        'C': [800],
        'gamma': [0.05],
        'epsilon': [7],
        'max_iter':[500],
    }
    SVR_model = SVR()
    grid_search = GridSearchCV(SVR_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    SVR_model = grid_search.best_estimator_
    SVR_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(SVR_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = SVR_model.predict(feature_train)
    test_predict = SVR_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = SVR_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = SVR_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return SVR_model.predict, train_predict, test_predict, SVR_model

def GB_predict(feature_train, target_train, feature_test, target_test):
    target_train = np.ravel(target_train)
    target_test = np.ravel(target_test)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "n_estimators": [300],
        "learning_rate": [0.018],
        "random_state": [32],
        "alpha": [0.7],
        "max_depth":[3],
        "max_features":[25],
        "min_samples_split":[3],
        "min_samples_leaf":[4],
        "subsample":[0.92],
        #"colsample_bylevel":[0.7,0.8,0.9],
        #"early_stopping_rounds":[5,10],
            }
    GB_model = GradientBoostingRegressor()
    grid_search = GridSearchCV(GB_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    GB_model = grid_search.best_estimator_
    GB_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(GB_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(GB_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(GB_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = GB_model.predict(feature_train)
    test_predict = GB_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = GB_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = GB_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))

    # GB_model = GradientBoostingRegressor()
    # np.random.seed(42)
    # rfecv = RFECV(GB_model, step=1, cv=kf, scoring='r2', importance_getter='feature_importances_' )
    # rfecv.fit(feature_train, target_train)
    # print( rfecv.support_)
    # print("Ranking %s" % rfecv.ranking_)
    # print( rfecv.n_features_)
    # print( rfecv.estimator_.feature_importances_)
    # gb_score = rfecv.score(feature_train, target_train)
    # print(gb_score)
    # gb_score1 = rfecv.score(feature_test, target_test)
    # print(gb_score1)
    # print("Grid Scores %s" % rfecv.cv_results_['mean_test_score'])
    # plt.figure()
    # plt.title('RFECV feature ranking change curve')
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score")
    # plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    # plt.grid(True)
    # plt.show()

    # GB_model = GradientBoostingRegressor()
    # trained_model = GB_model.fit(feature_train, target_train)
    # target_test_pred = GB_model.predict(feature_test)
    # predictions = [round(value) for value in target_test_pred]
    # r2 = r2_score(target_test, predictions)
    # print(GB_model.feature_importances_)
    # plt.bar(range(len(GB_model.feature_importances_)), GB_model.feature_importances_)
    # plt.show()
    # print("R2:"  + str(r2))
    # thresholds = sort(GB_model.feature_importances_)
    # for thresh in thresholds:
    #     # select features using threshold
    #     selection = SelectFromModel(GB_model, threshold=thresh, prefit=True)
    #     select_feature_train = selection.transform(feature_train)
    #
    #     # train model
    #     selection_model = GradientBoostingRegressor()
    #     selection_model.fit(select_feature_train, target_train)
    #
    #     # eval model
    #     select_feature_test = selection.transform(feature_test)
    #     target_test_pred = selection_model.predict(select_feature_test)
    #     predictions = [round(value) for value in target_test_pred]
    #     r2 = r2_score(target_test, predictions)
    #     print("Thresh=%.3f, n=%d, R2: %.2f%%" % (thresh, select_feature_train.shape[1], r2))

    # GB_model = GradientBoostingRegressor()
    # GB_model.fit(feature_train, target_train)
    # importances = GB_model.feature_importances_
    # feature_importances_df = pd.DataFrame({'feature': feature_train.columns, 'importance': importances})
    # feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
    # target_test_pred = GB_model.predict(feature_test)
    # r2 = r2_score(target_test, target_test_pred)
    # print(feature_importances_df)
    # print(r2)
    # param_grid = {"n_estimators": [18],
    #               "learning_rate": [0.3],
    #               "random_state": [42],
    #               "alpha": [0.1],
    #               "max_depth":[5],
    #               "min_samples_split":[2],
    #               "min_samples_leaf":[3],
    #               "subsample":[0.75],
    #               #"colsample_bylevel":[0.7,0.8,0.9],
    #               #"early_stopping_rounds":[5,10],
    #               }
    # GB_model = GradientBoostingRegressor()
    # r2 = make_scorer(r2_score, greater_is_better = True)
    # grid_search = GridSearchCV(GB_model, param_grid, cv=kf, scoring='r2')
    # grid_search.fit(feature_train, target_train)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    # print( grid_search.best_estimator_)
    # print(grid_search.cv_results_)
    # GB_model = grid_search.best_estimator_
    # learning_curve plot
    # all_target = np.concatenate((target_train, target_test), 0)
    # all_feature = np.concatenate((feature_train, feature_test), 0)
    # print(all_feature.shape)
    # print(all_target.shape)
    #
    # train_sizes, train_scores, test_scores = learning_curve(GB_model, all_feature, all_target, cv=10, random_state=42,
    #                                                         train_sizes=np.linspace(0.1, 1.0, 10))
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # print(test_scores_std)
    # plt.figure(figsize=(10, 6))
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.plot(train_sizes, train_scores_mean, label='training set score', marker='o')
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    # plt.plot(train_sizes, test_scores_mean, label='CV score', marker='o')
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
    # for i, j in zip(train_sizes, train_scores_mean):
    #     plt.text(i, j, '%.2f' % j, ha='center', va='bottom', fontsize=10)
    # for i, j in zip(train_sizes, test_scores_mean):
    #     plt.text(i, j, '%.2f' % j, ha='center', va='top', fontsize=10)
    # plt.xlabel('numble')
    # plt.ylabel('score')
    # plt.legend()
    # plt.title('learn curve')
    # plt.show()

    # GB_model = GradientBoostingRegressor()
    # GB_model = GradientBoostingRegressor(n_estimators=18, learning_rate=0.3, random_state=42, alpha=0.1,max_depth=5,min_samples_split=2,min_samples_leaf=3,subsample=0.75)

    # feature_importance = GB_model.feature_importances_
    # feature_name_list = ["temperature", "carbonation time", "CO2 pressure" , "CO2 concentration", "particle diameter",
    #                          "L/S", "CaO", "MgO", "Cl", "Na2O","K2O","SO3","SiO2", "Al2O3", "Fe2O3", "CO2 sequestration",
    #                      "total As", "total Cd","total Cr","total Cu","total Ni","total Pb","total Zn",
    #                      "leaching As", "leaching Cd","leaching Cr","leaching Cu","leaching Ni","leaching Pb","leaching Zn"]
    # sorted_indices = np.argsort(feature_importance)
    # sorted_feature_importance = np.array(feature_importance)[sorted_indices]
    # sorted_feature_name_list = np.array(feature_name_list)[sorted_indices]
    # y_pos = np.arange(len(feature_name_list))
    # plt.figure(dpi=600, figsize=(19, 20))
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # bars =plt.barh(y_pos, sorted_feature_importance, align='center', alpha=0.5)
    # plt.xticks(fontsize=30)
    # plt.yticks(y_pos, sorted_feature_name_list, fontsize=30)
    # plt.xlabel('Feature importance score', fontsize=30)
    # plt.rcParams['font.sans-serif'] = ['Arial']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    # for bar in bars:
    #     width = bar.get_width()
    #     if width > 0.01:
    #         plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', va='center', ha='left', fontsize=30,
    #                 fontweight='bold')
    #     else:
    #         plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', va='center', ha='left', fontsize=30,
    #                  fontweight='bold')
    # plt.tight_layout()
    # SHAP_plot_save_path = './SHAP_plot2/'
    # plt.savefig(SHAP_plot_save_path + "GB Feature Importance.png")
    # all_target = np.concatenate((target_train, target_test), 0)
    # all_predict = np.concatenate((train_predict, test_predict), 0)
    # all_R2 = r2_score(all_target, all_predict)

    # feature_importance = GB_model.feature_importances_
    # feature_name_list = ["temperature", "carbonation time", "CO2 pressure", "CO2 concentration", "particle diameter"
    #     , "L/S", "CaO", "MgO", "Cl", "Na2O", "K2O", "SO3", "SiO2", "Al2O3", "Fe2O3"]
    # sorted_indices = np.argsort(feature_importance)
    # sorted_feature_importance = np.array(feature_importance)[sorted_indices]
    # sorted_feature_name_list = np.array(feature_name_list)[sorted_indices]
    # y_pos = np.arange(len(feature_name_list))
    # plt.figure(dpi=600, figsize=(12, 12))
    # plt.barh(y_pos, sorted_feature_importance, align='center', alpha=0.5)
    # plt.yticks(y_pos, sorted_feature_name_list)
    # plt.xlabel('Feature Importance Score')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # SHAP_plot_save_path = './SHAP_plot2/'
    # plt.savefig(SHAP_plot_save_path + "GB Feature Importance.png")
    return GB_model.predict, train_predict, test_predict, GB_model

def XGB_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "n_estimators": [750],
        "learning_rate": [0.008],
        "random_state": [49],
        "alpha": [0.9],
        "max_depth": [1],
        "max_features": [28],
        "max_delta_step":[0],
        "min_child_weight": [1],
        "subsample": [0.95],
        "gamma":[0],
                 }
    XGB_model = XGBRegressor()
    grid_search = GridSearchCV(XGB_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    XGB_model = grid_search.best_estimator_
    XGB_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(XGB_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(XGB_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(XGB_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = XGB_model.predict(feature_train)
    test_predict = XGB_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = XGB_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = XGB_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return XGB_model.predict, train_predict, test_predict, XGB_model

def LGBM_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "num_leaves": [40],
        "n_estimators": [800],
        "learning_rate": [0.08],
        "max_depth": [8],
        "min_child_weight": [1],
        "min child sample": [3],
        "reg_alpha": [0.9],
        "subsample": [0.9],
        "verbosity": [-1],
    }
    LGBM_model = LGBMRegressor()
    grid_search = GridSearchCV(LGBM_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    LGBM_model = grid_search.best_estimator_
    LGBM_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(LGBM_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(LGBM_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(LGBM_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = LGBM_model.predict(feature_train)
    test_predict = LGBM_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = LGBM_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = LGBM_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return LGBM_model.predict, train_predict, test_predict, LGBM_model