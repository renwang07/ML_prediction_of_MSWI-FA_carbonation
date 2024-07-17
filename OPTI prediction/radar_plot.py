import numpy as np
import matplotlib.pyplot as plt

class Radar(object):
    def __init__(self, figure, title, labels, epoch, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]
        self.n = 6
        self.angles = np.arange(0, 360, 360.0 / self.n)
        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels = title, fontsize = 16, va="center", ha= "center", weight='bold')
        self.ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        self.ax.xaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        for ax, angle, label, i in zip(self.axes, self.angles, labels, epoch):
            ax.set_rgrids(i[1:], angle=angle, labels=label,fontsize=13,zorder=10)
            ax.spines['polar'].set_visible(False)
            ax.set_rlim(i[0], i[3])

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])

        values[0] = values[0]
        values[1] = ((values[1]-100) * 156 / 72) + 201
        values[2] = ((values[2]-0.7) * (-156) / 0.3) + 201
        values[3] = ((values[3]-2000) * 156 / 1962) + 201
        values[4] = ((values[4]-184) * 156 / 159) + 201
        values[5] = ((values[5]-0.7) * (-156) / 0.3) + 201

       #  values[0] = values[0]
       #  values[1] = ((values[1]-92) * 90 / 57) + 140
       #  values[2] = ((values[2]-0.72)*(-90) / 0.24) + 140
       #  values[3] = ((values[3]-2200) * 90 / 2145) + 140
       #  values[4] = ((values[4]-192) * 90 / 147) + 140
       #  values[5] = ((values[5]-0.72)*(-90) / 0.24) + 140
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))
    tit = ['', '', '', '', '', '']
    lab = [
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
            ]
    epo = [
        (201, 149, 97, 45),
        (100, 76, 52, 28),
        (0.7, 0.8, 0.9, 1),
        (2000, 1346, 692, 38),
        (184, 131, 78, 25),
        (0.7, 0.8, 0.9, 1)
            ]

    # lab = [
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     ]

    # epo = [
    #     (140, 110, 80, 50),
    #     (92, 73, 54, 35),
    #     (0.72, 0.80, 0.88, 0.96),
    #     (2200, 1485, 770, 55),
    #     (192, 126, 98, 45),
    #     (0.72, 0.80, 0.88, 0.96)
    #         ]
    plt.rcParams['font.family'] = 'Arial'
    radar = Radar(fig, tit, lab, epo)

    radar.plot([174.025, 82.316, 0.801, 1562.178, 118.709, 0.801], '-', lw=4, color='tab:red', alpha=1, label='DT', marker='o', markersize=10)
    radar.plot([48.242, 29.453, 0.984, 1603.48, 156.447, 0.984], '-', lw=4, color='tab:blue', alpha=1, label='MLP', marker='s', markersize=10)
    radar.plot([141.612, 48.889, 0.868, 41.204, 25.421, 0.872], '-', lw=4, color='tab:purple', alpha=1, label='KNN', marker='v', markersize=10)
    radar.plot([124.799, 32.538, 0.897, 1390.245, 128.488, 0.901], '-', lw=4, color='tab:brown', alpha=1, label='SVR', marker='^', markersize=10)
    radar.plot([118.815, 59.613, 0.907, 1550.67, 120.041, 0.907], '-', lw=4, color='tab:green', alpha=1, label='RF', marker='D', markersize=10)
    radar.plot([73.813, 42.875, 0.964, 214.014, 78.752, 0.964], '-', lw=4, color='tab:orange', alpha=1, label='GB', marker='x', markersize=10)
    radar.plot([172.251, 90.407, 0.805, 1562.846, 125.585, 0.805], '-', lw=4, color='tab:pink', alpha=1, label='XGB', marker='<', markersize=10)
    radar.plot([69.465, 44.507, 0.968, 1675.218, 141.098, 0.968], '-', lw=4, color='tab:cyan', alpha=1, label='LGBM', marker='>', markersize=10)

    # radar.plot([113.996, 74.012, 0.781, 1515.039, 124.023, 0.786], '-', lw=4, color='tab:red', alpha=1, label='DT', marker='o', markersize=10)
    # radar.plot([59.23, 39.657, 0.94, 1720.376, 171.221, 0.942], '-', lw=4, color='tab:blue', alpha=1, label='MLP', marker='s', markersize=10)
    # radar.plot([76.042, 42.001, 0.902, 60.077, 47.956 , 0.906], '-', lw=4, color='tab:purple', alpha=1, label='KNN', marker='v', markersize=10)
    # radar.plot([87.811, 57.337, 0.87, 1678.065, 134.598, 0.871], '-', lw=4, color='tab:brown', alpha=1, label='SVR', marker='^', markersize=10)
    # radar.plot([96.114, 65.165, 0.844, 1641.755, 124.891, 0.845], '-', lw=4, color='tab:green', alpha=1, label='RF', marker='D', markersize=10)
    # radar.plot([50.058, 35.371, 0.957, 309.388, 84.501, 0.96], '-', lw=4, color='tab:orange', alpha=1, label='GB', marker='x', markersize=10)
    # radar.plot([116.227, 76.227, 0.773, 1760.422, 131.906, 0.787], '-', lw=4, color='tab:pink', alpha=1, label='XGB', marker='<', markersize=10)
    # radar.plot([68.411, 52.48, 0.921, 1843.403, 142.245, 0.922], '-', lw=4, color='tab:cyan', alpha=1, label='LGBM', marker='>', markersize=10)

    legend = radar.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize = 15)
    plt.setp(legend.get_texts(), fontweight='bold')
    plt.show()