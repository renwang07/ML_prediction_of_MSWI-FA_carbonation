import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator
from shap import summary_plot,heatmap_plot, dependence_plot
from shap.plots import heatmap, scatter , waterfall
import shap

SHAP_plot_save_path = './SHAP_plot2/'
feature_SHAP_plot_path = 'All feature SHAP scatter plots/'
dependence_plot_save_path = 'SHAP dependence_plot/'

def get_label_name(feature_name):
    ignore_list = ["temperature", "carbonation time", "particle diameter", "CaO", "MgO", "MnO", "L/S"]
    if feature_name in ignore_list:
        return feature_name
    elif feature_name == "CO2 pressure":
        return "$CO_2$ pressure"
    elif feature_name == "CO2 concentration":
        return "$CO_2$ concentration"
    elif feature_name == "SiO2":
        return "SiO2"
    elif feature_name == "Al2O3":
        return "Al2O3"
    elif feature_name == "Fe2O3":
        return "$Fe2O3$"
    else:
        return feature_name

def feature_name_replace(feature_name: str):
    if type(feature_name) is str:
        return feature_name.replace('/', '-')
    else:
        print("\033[32mTypeError: Type of parameter 'feature_name' is str, input value is " + str(type(feature_name))
              + "\033[0m")
def predict_compare_plot(train_target, train_predict, test_target, test_predict):
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 20
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    colors = sns.color_palette("colorblind")
    test_result = pd.concat([pd.DataFrame(test_target.values), pd.DataFrame(test_predict)], axis=1)
    train_result = pd.concat([pd.DataFrame(train_target.values), pd.DataFrame(train_predict)], axis=1)
    plt.plot([0, 2100], [0, 2100], linestyle='--', alpha=0.85, c='black',linewidth=2)
    plt.scatter(train_result.iloc[:, 0], train_result.iloc[:, 1], alpha=1, c='w',edgecolors=colors[0], s=150, label='Training Set',facecolors='DE8F05')
    plt.scatter(test_result.iloc[:, 0], test_result.iloc[:, 1], marker='^', alpha=1, c='w', edgecolors=colors[1], s=150, label='Test Set')
    plt.tick_params(labelsize=20)
    plt.xlabel('Actual OPTI', fontsize='20')
    plt.ylabel('Predicted OPTI', fontsize='20')
    plt.legend(loc=2, fontsize=20, markerscale=1.3, frameon=False)
    plt.grid(False)
    x_major_locator = MultipleLocator(420)
    plt.gca().xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(420)
    plt.gca().yaxis.set_major_locator(y_major_locator)
    plt.xlim((0, 2100))
    plt.ylim((0, 2100))
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "predict.png")
    plt.close()

def test_predict_plot(test_predict, test_target):
    plt.figure(dpi=600, figsize=(10, 6))
    plt.rcParams['font.size'] = 28
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    sample_number = []
    for number in range(len(test_predict)):
        sample_number.append(number+1)

    colors = sns.color_palette("colorblind")
    plt.plot(sample_number, test_target, label="Actual OPTI", linewidth=3, linestyle='-', marker='o', markersize='12')
    plt.plot(sample_number, test_predict, color=colors[1], label="Predicted OPTI", linewidth=3, linestyle='--', marker='^', markersize='12')
    plt.xlabel("sample number", fontsize=28)
    plt.ylabel("OPTI", fontsize=28)
    plt.legend(loc='upper right', fontsize=28, markerscale=2, frameon=False)
    plt.tick_params(labelsize=28)
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((0, 21))
    plt.ylim((0, 1200))
    x = MultipleLocator(3)
    y = MultipleLocator(240)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "test_predict.png")
    plt.close()

def relevance_plot(train_shap_values, feature_train, feature_name1, feature_name2):
    feature_values1 = feature_train[str(feature_name1)].values
    feature_values2 = feature_train[str(feature_name2)].values
    feature_name1_index = int(feature_train.columns.get_loc(str(feature_name1)))
    feature_name2_index = int(feature_train.columns.get_loc(str(feature_name2)))
    shap_values_sum = train_shap_values[:, feature_name1_index] + train_shap_values[:, feature_name2_index]
    bottom = shap_values_sum.min() - 1
    top = shap_values_sum.max() + 1
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_values1, feature_values2, shap_values_sum, c=shap_values_sum, cmap='jet')
    ax1.scatter3D(feature_values1, feature_values2, bottom - 1)
    ax1.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values_sum)):
        xs = [feature_values1[number], feature_values1[number]]
        ys = [feature_values2[number], feature_values2[number]]
        zs = [shap_values_sum[number], bottom - 1]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(str(feature_name1), fontsize=15)
    plt.ylabel(str(feature_name2), fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)
    plt.savefig(SHAP_plot_save_path + str(feature_name1) + "_" + str(feature_name2) + ".png")

def material_relevance_plot(feature_train_1, feature_train_2, train_shap_values_1, train_shap_values_2, feature_name1, feature_name2):
    shap_values = train_shap_values_1 + train_shap_values_2
    bottom = int(shap_values.min()) - 1
    top = int(shap_values.max()) + 1
    c = shap_values
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 13
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_train_1, feature_train_2, shap_values, c=c, cmap='jet')
    ax1.scatter3D(feature_train_1, feature_train_2, -25)
    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values)):
        xs = [feature_train_1[number], feature_train_1[number]]
        ys = [feature_train_2[number], feature_train_2[number]]
        zs = [shap_values[number], bottom]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(feature_name1, fontsize=15)
    plt.ylabel(feature_name2, fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)
    plt.savefig(SHAP_plot_save_path + feature_name1 + "_" + feature_name2 + ".png")

def calculate_material(feature_name_1, feature_name_2, train_shap_values, feature_train):
    feature1_num = int(feature_train.columns.get_loc(str(feature_name_1)))
    feature2_num = int(feature_train.columns.get_loc(str(feature_name_2)))
    feature_sum_value = feature_train[str(feature_name_1)].values + feature_train[str(feature_name_2)].values
    shap_sum_value = train_shap_values[:, feature1_num] + train_shap_values[:, feature2_num]
    return feature_sum_value, shap_sum_value

def get_material_shap_value(feature_name, train_shap_value, feature_train):
    feature_num = int(feature_train.columns.get_loc(str(feature_name)))
    feature_value = feature_train[str(feature_name)].values
    feature_shap_value = train_shap_value[:, feature_num]
    return feature_value, feature_shap_value

def SHAP_plot(predict_model, raw_feature_train_summary, raw_feature_train):
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer.shap_values(raw_feature_train)   #使用 `explainer` 对象来计算 `feature_train` 数据对应的 SHAP 值，出现进度条，用于表示计算的进度
    # feature_SHAP(9, feature_train['Fe2O3'], shap_values)

    # All features SHAP scatter plots
    feature_name_list = ["temperature", "carbonation time", "CO2 pressure" , "CO2 concentration", "particle diameter",
                             "L/S", "CaO", "MgO", "Cl", "Na2O","K2O","SO3","SiO2", "Al2O3", "Fe2O3", "CO2 sequestration",
                         "total As", "total Cd","total Cr","total Cu","total Ni","total Pb","total Zn",
                         "leaching As", "leaching Cd","leaching Cr","leaching Cu","leaching Ni","leaching Pb","leaching Zn"]
    for feature_name in feature_name_list:
        plt.figure(dpi=600, figsize=(10, 8))
        plt.rcParams['font.family'] = 'Arial'
        feature_value, SHAP_value = get_material_shap_value(feature_name, shap_values, raw_feature_train)
        plt.scatter(feature_value, SHAP_value, s=25)
        plt.tick_params(labelsize=23, pad=0.1)
        plt.xlabel(get_label_name(feature_name=feature_name) + " Value", fontsize=23)
        plt.ylabel("SHAP Value of " + get_label_name(feature_name=feature_name), fontsize=23)
        plt.tight_layout()
        plt.savefig(SHAP_plot_save_path + feature_SHAP_plot_path + feature_name_replace(feature_name) + " scatter.png")
        plt.close()

    # Dot summary plot
    plt.figure(dpi=600, figsize=(8, 8))
    plt.rcParams['font.size'] = 25
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tick_params(labelsize=25)
    summary_plot(shap_values, raw_feature_train, plot_type='dot', show=False, max_display=15)
    plt.rcParams.update({'font.size': 25, 'font.weight': 'bold'})
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance(15 features).png")
    plt.close()

    # plt.figure(dpi=600, figsize=(8, 8))
    # plt.rcParams['font.size'] = 20
    # plt.rcParams['font.sans-serif'] = ['Arial']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.tick_params(labelsize=15)
    # summary_plot(shap_values, raw_feature_train, plot_type='dot', show=False, max_display=30)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance(all features).png")
    # plt.close()

    # Bar summary plot
    plt.figure(dpi=600, figsize=(10, 8))
    plt.rcParams['font.size'] = 22
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    summary_plot(shap_values, raw_feature_train, plot_type='bar', show=False,max_display=15)


    fig = plt.gcf()
    ax = plt.gca()
    bars = ax.patches
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tick_params(labelsize=22)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left', fontsize=22, fontweight='bold')
    plt.rcParams.update({'font.size': 25, 'font.weight': 'bold'})
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance1(15 features).png")
    plt.close()

    # plt.figure(dpi=600, figsize=(10, 8))
    # plt.rcParams['font.size'] = 20
    # plt.rcParams['font.sans-serif'] = ['Arial']
    # plt.rcParams['axes.unicode_minus'] = False
    # summary_plot(shap_values, raw_feature_train, plot_type='bar', show=False,max_display=30)
    #
    # fig = plt.gcf()
    # ax = plt.gca()
    # bars = ax.patches
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.tick_params(labelsize=15)
    # for bar in bars:
    #     width = bar.get_width()
    #     ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
    #             va='center', ha='left', fontsize=15, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance1(all features).png")
    # plt.close()

    # # Heatmap Plot
    # plt.figure(dpi=600, figsize=(6, 6))
    # heatmap_plot(shap_values)
    # plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance2.png")
    # plt.tick_params(labelsize=15)
    # plt.close()

    # for feature_num in range(15):
    #     feature_SHAP(feature_number=feature_num, feature_name=feature_name_list[feature_num],
    #                  feature_train=feature_train[feature_name_list[feature_num]], SHAP_values=shap_values)

def feature_SHAP(feature_number, feature_name, feature_train, SHAP_values):
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt_x = feature_train
    feature_SHAP_value = SHAP_values[:, feature_number]
    plt.scatter(plt_x, feature_SHAP_value, s=10)
    plt.xlabel(feature_name + " Value")
    plt.ylabel("SHAP Value of " + feature_name)
    plt.savefig(SHAP_plot_save_path + feature_name_replace(feature_name=feature_name) + ".png")
    plt.close()

def other_SHAP_plot(predict_model, raw_feature_train_summary, raw_feature_train,raw_train_target,raw_test_feature,raw_test_target):
    fig = plt.figure(dpi=600, figsize=(6, 6))
    # expleiner = shap.Explainer(predict_model, feature_train)
    # shap_values = expleiner(feature_train)
    feature_name_list = ["temperature", "carbonation time", "CO2 pressure" , "CO2 concentration", "particle diameter",
                             "L/S", "CaO", "MgO", "Cl", "Na2O","K2O","SO3","SiO2", "Al2O3", "Fe2O3", "CO2 sequestration",
                         "total As", "total Cd","total Cr","total Cu","total Ni","total Pb","total Zn",
                         "leaching As", "leaching Cd","leaching Cr","leaching Cu","leaching Ni","leaching Pb","leaching Zn"]
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer(raw_feature_train)
    # # Heat map
    # plt.figure(dpi=1200, figsize=(12, 12))
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # heatmap(shap_values, show=False, max_display=20)
    # plt.tick_params(labelsize=13)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_heatmap.png")
    # plt.close()
    # # partial feature importance plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # shap.plots.bar(shap_values[0], show=False, max_display=25)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_partial_FI_plot.png")
    # plt.close()
    # # feature clustering plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.rcParams['font.family'] = 'Arial'
    # all_target = np.concatenate((raw_train_target, raw_test_target), 0)
    # all_feature = np.concatenate((raw_feature_train, raw_test_feature), 0)
    # clustering = shap.utils.hclust(all_feature, all_target)
    # shap.plots.bar(shap_values,clustering=clustering, clustering_cutoff=0.5,show=False,max_display=30)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_feature_clustering_plot.png")
    # plt.close()

    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer.shap_values(raw_feature_train)
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('leaching Zn', shap_values, raw_feature_train, interaction_index='CO2 sequestration', show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Leaching Zn',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nLeaching Zn', fontsize=25, fontweight='bold')

    interaction_feature_values = raw_feature_train['CO2 sequestration']
    data = {
        'feature_value': raw_feature_train['leaching Zn'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('leaching Zn')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/leaching Zn_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/leaching Zn_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('carbonation time', shap_values, raw_feature_train, interaction_index='leaching Ni',show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Carbonation Time',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nCarbonation Time', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['leaching Ni']
    data = {
        'feature_value': raw_feature_train['carbonation time'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('carbonation time')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/carbonation time_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/carbonation time_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('total Cu', shap_values, raw_feature_train, interaction_index='CO2 sequestration',show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Total Cu',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nTotal Cu', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['CO2 sequestration']
    data = {
        'feature_value': raw_feature_train['total Cu'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('total Cu')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/total Cu_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/total Cu_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('CO2 concentration', shap_values, raw_feature_train,interaction_index='leaching Pb', show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='CO2 Concentration',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nCO2 Concentration', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['leaching Pb']
    data = {
        'feature_value': raw_feature_train['CO2 concentration'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('CO2 concentration')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/CO2 concentration_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/CO2 concentration_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('total Ni', shap_values, raw_feature_train, show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Total Ni',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nTotal Ni', fontsize=25, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/total Ni_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('CO2 pressure', shap_values, raw_feature_train, interaction_index='L/S', show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='CO2 Pressure',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nCO2 Pressure', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['L/S']
    data = {
        'feature_value': raw_feature_train['CO2 pressure'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('CO2 pressure')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/CO2 pressure_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/CO2 pressure_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('CO2 sequestration', shap_values, raw_feature_train, interaction_index='total Cu', show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='CO2 Sequestration',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nCO2 Sequestration', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['total Cu']
    data = {
        'feature_value': raw_feature_train['CO2 sequestration'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('CO2 sequestration')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/CO2 sequestration_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/CO2 sequestration_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('MgO', shap_values, raw_feature_train, show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='MgO',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nMgO', fontsize=25, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/MgO_dependence.png')
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('leaching Pb', shap_values, raw_feature_train,interaction_index='CO2 concentration', show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Leaching Pb',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nLeaching Pb', fontsize=25, fontweight='bold')
    interaction_feature_values = raw_feature_train['CO2 sequestration']
    data = {
        'feature_value': raw_feature_train['leaching Pb'],
        'SHAP_value': shap_values[:, raw_feature_train.columns.get_loc('leaching Pb')],
        'interaction_feature_value': interaction_feature_values
    }
    df = pd.DataFrame(data)
    df.to_excel('./SHAP_plot2/SHAP dependence_plot/leaching Pb_dependence.xlsx', index=False)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/leaching Pb_dependence.png')
    plt.rcParams['font.family'] = 'Arial'
    plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    shap.dependence_plot('leaching Cu', shap_values, raw_feature_train, show=False,dot_size=45)
    plt.tick_params(labelsize=25)
    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xlabel(xlabel='Leaching Cu',fontsize=25, fontweight='bold')
    plt.ylabel(ylabel='SHAP value for\nLeaching Cu', fontsize=25, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/leaching Cu_dependence.png')
    plt.close()

    # # SHAP partial force plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # shap.plots.force(explainer.expected_value, shap_values[0], features=feature_name_list, show=False,matplotlib=True)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_partial_force_plot.png")
    # plt.close()

    # SHAP all force plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # shap.initjs()
    # # shap.force_plot(explainer.expected_value, shap_values.values,feature_train, show=False)
    # shap.force_plot(explainer.expected_value, shap_values[0:5], test_feature[0 : 5], plot_cmap="DrDb",
    #                 feature_names=feature_name_list)
    # plt.tick_params(labelsize=8)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_all_force_plot.png")
    # plt.close()

    # # partial_Decision plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # shap.decision_plot(explainer.expected_value, shap_values[0], feature_names = feature_name_list, show=False)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_partial_decision_plot.png")
    # plt.close()
    #
    # # all_Decision plot
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # shap.decision_plot(explainer.expected_value, shap_values, feature_names = feature_name_list, show=False)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(SHAP_plot_save_path + "SHAP_all_decision_plot.png")
    # plt.close()