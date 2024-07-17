import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import spearmanr

violin_plot_savepath = './SHAP_plot2/dataset_statistics_violin_plot/'

def violin_plot(data_df):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["temperature"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='Temperature', ylabel='value(°C)')
    sns.boxplot(data=data_df["temperature"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Temperature_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["carbonation time"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='Carbonation Time', ylabel='value(h)')
    sns.boxplot(data=data_df["carbonation time"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'carbonation time_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 pressure"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$CO_2$ Pressure', ylabel='value(bar)')
    sns.boxplot(data=data_df["CO2 pressure"], width=0.04, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 pressure_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 concentration"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$CO_2$ Concentration', ylabel='value(%)')
    sns.boxplot(data=data_df["CO2 concentration"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 concentration_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["particle diameter"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='Particle Diameter', ylabel='value(μm)')
    sns.boxplot(data=data_df["particle diameter"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'particle diameter_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["L/S"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='L/S', ylabel='value(L/kg)')
    sns.boxplot(data=data_df["L/S"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'LS_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CaO"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='CaO', ylabel='value(%)')
    sns.boxplot(data=data_df["CaO"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CaO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["MgO"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='MgO', ylabel='value(%)')
    sns.boxplot(data=data_df["MgO"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'MgO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Cl"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='Cl', ylabel='value(%)')
    sns.boxplot(data=data_df["Cl"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Cl_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Na2O"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$Na_2O$', ylabel='value(%)')
    sns.boxplot(data=data_df["Na2O"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Na2O_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["K2O"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$K_2O$', ylabel='value(%)')
    sns.boxplot(data=data_df["K2O"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'K2O_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["SO3"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$SO_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["SO3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'SO3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["SiO2"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$SiO_2$', ylabel='value(%)')
    sns.boxplot(data=data_df["SiO2"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'SiO2_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Al2O3"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$Al_2O_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["Al2O3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Al2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Fe2O3"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$Fe_2O_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["Fe2O3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Fe2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 sequestration"], inner=None, cut=0, color='g', alpha= 0.45,linewidth=3).set(xlabel='$CO_2$ Sequestration', ylabel='value(g-$CO_2$/kg-FA)')
    sns.boxplot(data=data_df["CO2 sequestration"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 sequestration_statistics.png')
    plt.close()

def heat_map(data_df):
    SHAP_plot_save_path = './SHAP_plot2/'
    heat_map_path = SHAP_plot_save_path + "heatmap/"
    plt.figure(dpi=1000, figsize=(10, 10))
    plt.rcParams['font.size'] = 13
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # data_heat = np.corrcoef(data_df.values, rowvar=0)
    data_heat, p_value = spearmanr(data_df.values, axis=0)
    data_heat = pd.DataFrame(data=data_heat, columns=data_df.columns, index=data_df.columns)
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.family'] = 'Arial'
    colors = ["#581113", "#F6F6F6", "#202050"]
    n_colors = 100
    cmap = LinearSegmentedColormap.from_list("", ["#202050", "#1B5AAB", "#4EA2D2", "#BCD6EC", "#F6F6F6", "#FECEB3",
                                                  "#EF8565", "#B7201E", "#581113"])
    # input features
    # ax = sns.heatmap(np.round(data_heat, 2), square=True, annot=False, fmt='.2f', linewidths=.5, cmap=cmap,annot_kws={"size":14},
    #                  cbar_kws={'fraction': 0.046, 'pad': 0.03}, vmin=-1, vmax=1,
    #                  xticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure','$CO_2$ Concentration','Particle Diameter',
    #                               'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$'],
    #                  yticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure','$CO_2$ Concentration','Particle Diameter',
    #                               'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$'])
    # plt.xticks(fontsize=16,rotation=45,rotation_mode='anchor',ha='right')
    # plt.yticks(fontsize=16)
    # plt.savefig(heat_map_path + 'heatmap-input features.png', bbox_inches='tight')

    # all features
    ax = sns.heatmap(np.round(data_heat, 2), square=True, annot=False, fmt='.2f', linewidths=.5, cmap=cmap,annot_kws={"size":14},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03}, vmin=-1, vmax=1,
                     xticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure','$CO_2$ Concentration','Particle Diameter',
                                  'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$','$CO_2$ Sequestration'],
                     yticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure','$CO_2$ Concentration','Particle Diameter',
                                  'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$','$CO_2$ Sequestration'])
    plt.xticks(fontsize=16,rotation=45, rotation_mode='anchor',ha='right')
    plt.yticks(fontsize=16)
    plt.savefig(heat_map_path + 'heatmap-all features.png', bbox_inches='tight')