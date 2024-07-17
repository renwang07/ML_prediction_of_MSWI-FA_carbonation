import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import unicodeit
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
    sns.violinplot(data_df["temperature"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Temperature', ylabel='value(°C)')
    sns.boxplot(data=data_df["temperature"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Temperature_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["carbonation time"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Carbonation Time', ylabel='value(h)')
    sns.boxplot(data=data_df["carbonation time"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'carbonation time_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 pressure"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$CO_2$ Pressure', ylabel='value(bar)')
    sns.boxplot(data=data_df["CO2 pressure"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 pressure_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 concentration"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$CO_2$ Concentration', ylabel='value(%)')
    sns.boxplot(data=data_df["CO2 concentration"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 concentration_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["particle diameter"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Particle Diameter', ylabel='value(μm)')
    sns.boxplot(data=data_df["particle diameter"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'particle diameter_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["L/S"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='L/S', ylabel='value(L/kg)')
    sns.boxplot(data=data_df["L/S"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'LS_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CaO"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='CaO', ylabel='value(%)')
    sns.boxplot(data=data_df["CaO"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CaO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["MgO"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='MgO', ylabel='value(%)')
    sns.boxplot(data=data_df["MgO"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'MgO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Cl"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Cl', ylabel='value(%)')
    sns.boxplot(data=data_df["Cl"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Cl_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Na2O"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$Na_2O$', ylabel='value(%)')
    sns.boxplot(data=data_df["Na2O"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Na2O_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["K2O"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$K_2O$', ylabel='value(%)')
    sns.boxplot(data=data_df["K2O"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'K2O_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["SO3"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$SO_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["SO3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'SO3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["SiO2"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$SiO_2$', ylabel='value(%)')
    sns.boxplot(data=data_df["SiO2"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'SiO2_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Al2O3"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$Al_2O_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["Al2O3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Al2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Fe2O3"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$Fe_2O_3$', ylabel='value(%)')
    sns.boxplot(data=data_df["Fe2O3"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Fe2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["CO2 sequestration"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='$CO_2$ Sequestration', ylabel='value(g-$CO_2$/kg-FA)')
    sns.boxplot(data=data_df["CO2 sequestration"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'CO2 sequestration_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total As"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total As', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total As"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total As_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Cd"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Cd', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Cd"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Cd_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Cr"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Cr', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Cr"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Cr_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Cu"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Cu', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Cu"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Cu_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Ni"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Ni', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Ni"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Ni_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Pb"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Pb', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Pb"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Pb_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["total Zn"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Total Zn', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["total Zn"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'total Zn_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching As"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching As', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching As"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching As_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Cd"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Cd', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Cd"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Cd_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Cr"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Cr', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Cr"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Cr_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Cu"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Cu', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Cu"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Cu_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Ni"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Ni', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Ni"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Ni_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Pb"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Pb', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Pb"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Pb_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["leaching Zn"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='Leaching Zn', ylabel='value(mg/L)')
    sns.boxplot(data=data_df["leaching Zn"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'leaching Zn_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["OPTI"], inner=None, cut=0, color='purple', alpha=0.35, linewidth=3).set(xlabel='OPTI', ylabel='value')
    sns.boxplot(data=data_df["OPTI"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'OPTI_statistics.png')
    plt.close()

def heat_map(data_df):
    SHAP_plot_save_path = './SHAP_plot2/'
    heat_map_path = SHAP_plot_save_path + "heatmap/"
    plt.figure(dpi=1000, figsize=(15, 15))
    plt.rcParams['font.size'] = 6
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # data_heat = np.corrcoef(data_df.values, rowvar=0)
    data_heat, p_value = spearmanr(data_df.values, axis=0)
    data_heat = pd.DataFrame(data=data_heat, columns=data_df.columns, index=data_df.columns)
    plt.figure(figsize=(15 , 15))
    plt.rcParams['font.family'] = 'Arial'
    colors = ["#581113", "#F6F6F6", "#202050"]
    n_colors = 100
    cmap = LinearSegmentedColormap.from_list("", ["#202050", "#1B5AAB", "#4EA2D2", "#BCD6EC", "#F6F6F6", "#FECEB3",
                                                  "#EF8565", "#B7201E", "#581113"])
    # input features
    # ax = sns.heatmap(np.round(data_heat, 2), square=True, annot=True, fmt='.2f', linewidths=.5, cmap=cmap,annot_kws={"size":12},
    #                  cbar_kws={'fraction': 0.046, 'pad': 0.03}, vmin=-1, vmax=1,
    #                  xticklabels=['temperature', 'carbonation time', '$CO_2$ pressure','$CO_2$ concentration','particle diameter',
    #                               'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$',"$CO_2$ sequestration",
    #                      "total As", "total Cd","total Cr","total Cu","total Ni","total Pb","total Zn",
    #                      "leaching As", "leaching Cd","leaching Cr","leaching Cu","leaching Ni","leaching Pb","leaching Zn"],
    #                  yticklabels=['temperature', 'carbonation time', '$CO_2$ pressure','$CO_2$ concentration','particle diameter',
    #                               'L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$',"$CO_2$ sequestration",
    #                      "total As", "total Cd","total Cr","total Cu","total Ni","total Pb","total Zn",
    #                      "leaching As", "leaching Cd","leaching Cr","leaching Cu","leaching Ni","leaching Pb","leaching Zn"])
    # plt.xticks(fontsize=15,rotation=45,rotation_mode='anchor',ha='right')
    # plt.yticks(fontsize=15)
    # plt.savefig(heat_map_path + 'heatmap-input features.png', bbox_inches='tight')

    # all features
    ax = sns.heatmap(np.round(data_heat, 2), square=True, annot=False, fmt='.2f', linewidths=.5, cmap=cmap,annot_kws={"size":12},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03 }, vmin = -1, vmax=1,
                     xticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure', '$CO_2$ Concentration',
                                  'Particle Diameter','L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$',"$CO_2$ Sequestration",
                                 "Total As", "Total Cd","Total Cr","Total Cu","Total Ni","Total Pb","Total Zn",
                                 "Teaching As", "Teaching Cd","Teaching Cr","Teaching Cu","Teaching Ni","Teaching Pb","Teaching Zn",'OPTI'],
                     yticklabels=['Temperature', 'Carbonation Time', '$CO_2$ Pressure', '$CO_2$ Concentration',
                                  'Particle Diameter','L/S','CaO','MgO','Cl','$Na_2O$','$K_2O$','$SO_3$','$SiO_2$','$Al_2O_3$','$Fe_2O_3$',"$CO_2$ Sequestration",
                                 "Total As", "Total Cd","Total Cr","Total Cu","Total Ni","Total Pb","Total Zn",
                                 "Teaching As", "Teaching Cd","Teaching Cr","Teaching Cu","Teaching Ni","Teaching Pb","Teaching Zn",'OPTI'])
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    plt.xticks(fontsize=15,rotation=45, rotation_mode='anchor',ha='right')
    plt.yticks(fontsize=15)
    plt.savefig(heat_map_path + 'heatmap-all features.png', bbox_inches='tight')