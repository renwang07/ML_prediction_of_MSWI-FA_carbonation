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
    plt.plot([0, 350], [0, 350], linestyle='--', alpha=0.85, c='black', linewidth=2)
    plt.scatter(train_result.iloc[:, 0], train_result.iloc[:, 1], alpha=1, c='w',edgecolors=colors[0], s=150, label='Training Set',facecolors='DE8F05')
    plt.scatter(test_result.iloc[:, 0], test_result.iloc[:, 1], marker='^', alpha=1, c='w', edgecolors=colors[1], s=150, label='Test Set')
    plt.tick_params(labelsize=20)
    plt.xlabel('Actual carbon sequestration (g-$CO_2$/kg-FA)', fontsize='17')
    plt.ylabel('Predicted carbon sequestration (g-$CO_2$/kg-FA)', fontsize='17')
    plt.legend(loc=2, fontsize=20, markerscale=1.3, frameon=False)
    plt.grid(False)
    x_major_locator = MultipleLocator(50)
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.xlim((0, 350))
    plt.ylim((0, 350))
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "predict.png")
    plt.close()

def test_predict_plot(test_predict, test_target):
    plt.figure(dpi=600, figsize=(15, 9))
    plt.rcParams['font.size'] = 28
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    sample_number = []
    for number in range(len(test_predict)):
        sample_number.append(number+1)
    colors = sns.color_palette("colorblind")
    plt.plot(sample_number, test_target, label="Actual carbon sequestration", linewidth=3, linestyle='-', marker='o', markersize='12')
    plt.plot(sample_number, test_predict, color=colors[1], label="Predicted carbon sequestration", linewidth=3, linestyle='--', marker='^', markersize='12')
    plt.xlabel("Sample Number", fontsize=30)
    plt.ylabel("Carbon Sequestration(g-$CO_2$/kg-FA)", fontsize=30)
    plt.legend(loc='upper right', fontsize=28, markerscale=2, frameon=False)
    plt.tick_params(labelsize=30)
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((0, 52))
    plt.ylim((0, 400))
    x = MultipleLocator(10)
    y = MultipleLocator(80)
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
    shap_values = explainer.shap_values(raw_feature_train)
    # feature_SHAP(9, feature_train['Fe2O3'], shap_values)
    # All features SHAP scatter plots
    feature_name_list = ["temperature", "carbonation time", "CO2 pressure" , "CO2 concentration", "particle diameter",
                             "L/S", "CaO", "MgO", "Cl", "Na2O","K2O","SO3","SiO2", "Al2O3", "Fe2O3"]
    for feature_name in feature_name_list:
        plt.figure(dpi=600, figsize=(10, 10))
        plt.rcParams['font.family'] = 'Arial'
        feature_value, SHAP_value = get_material_shap_value(feature_name, shap_values, raw_feature_train)
        data = {
            'feature_value': feature_value,
            'SHAP_value': SHAP_value
                }
        df = pd.DataFrame(data)
        df.to_excel(SHAP_plot_save_path + feature_SHAP_plot_path + feature_name_replace(feature_name) + " scatter.xlsx")
        plt.scatter(feature_value, SHAP_value, s=45)
        plt.tick_params(labelsize=45, pad=0.1)
        plt.xlabel(get_label_name(feature_name=feature_name) + " Value", fontsize=45)
        plt.ylabel("SHAP Value of " + get_label_name(feature_name=feature_name), fontsize=45)
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
    summary_plot(shap_values, raw_feature_train, plot_type='dot', show=False)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance.png")
    plt.close()

    # Bar summary plot
    plt.figure(dpi=600, figsize=(10, 8))
    plt.rcParams['font.size'] = 22
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    summary_plot(shap_values, raw_feature_train, plot_type='bar', show=False)
    fig = plt.gcf()
    ax = plt.gca()
    bars = ax.patches
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tick_params(labelsize=22)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance1.png")
    plt.close()

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
    feature_name_list = ["temperature", "carbonation time", "CO2 pressure", "CO2 concentration", "particle diameter",
                         "L/S", "CaO", "MgO", "Cl", "Na2O", "K2O", "SO3", "SiO2", "Al2O3", "Fe2O3"]
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer(raw_feature_train)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    heatmap(shap_values, show=False, max_display=15)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_heatmap.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    waterfall(shap_values[0], show=False, max_display=15)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_waterfall_plot.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.plots.bar(shap_values[0], show=False, max_display=13)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_FI_plot.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    all_target = np.concatenate((raw_train_target, raw_test_target), 0)
    all_feature = np.concatenate((raw_feature_train, raw_test_feature), 0)
    clustering = shap.utils.hclust(all_feature, all_target)
    shap.plots.bar(shap_values,clustering=clustering, clustering_cutoff=0.5,show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_feature_clustering_plot.png")
    plt.close()
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer.shap_values(raw_feature_train)

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('MgO', shap_values, raw_feature_train, show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Mgo_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Fe2O3', shap_values, raw_feature_train, show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Fe2O3_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('carbonation time', shap_values, raw_feature_train, show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/carbonation time_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('CaO', shap_values, raw_feature_train, show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/CaO_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('L/S', shap_values, raw_feature_train, show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/L-S_dependence.png')
    plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.plots.force(explainer.expected_value, shap_values[0], features=feature_name_list, show=False,matplotlib=True)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_force_plot.png")
    plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.decision_plot(explainer.expected_value, shap_values[0], feature_names = feature_name_list,show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_decision_plot.png")
    plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.decision_plot(explainer.expected_value, shap_values, feature_names = feature_name_list,show=False)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_all_decision_plot.png")
    plt.close()