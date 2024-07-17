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
        self.ax.set_thetagrids(self.angles, labels = title, fontsize=16, va="center", ha="center", weight = 'bold')
        self.ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        self.ax.xaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        for ax, angle, label, i in zip(self.axes, self.angles, labels, epoch):
            ax.set_rgrids(i[1:], angle=angle, labels=label, fontsize=13, zorder=10)
            ax.spines['polar'].set_visible(False)
            ax.set_rlim(i[0], i[3])
    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
       #  values[0] = values[0]
       #  values[1] = (values[1]-20) * 30/18 + 34
       #  values[2] = ((values[2]-0.88) * (-30) / 0.12) + 34
       #  values[3] = (values[3]-284) * 30 / 279 + 34
       #  values[4] = (values[4]-95) * 30 / 90 +34
       #  values[5] = ((values[5]-0.88) * (-30) / 0.12) + 34

        values[0] = values[0]
        values[1] = ((values[1]-33) * 33 / 21) + 51
        values[2] = ((values[2]-0.7) * (-33) / 0.27) + 51
        values[3] = ((values[3]-411)*33 / 396) + 51
        values[4] = ((values[4]-109) * 33 / 96) + 51
        values[5] = ((values[5]-0.7) * (-33) / 0.27) + 51

        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))
    tit = ['', '', '', '', '', '']
    # lab = [
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #     list(('', '', '')),
    #         ]
    # epo = [
    #     (34, 24, 14, 4),
    #     (20, 14, 8, 2),
    #     (0.88, 0.92, 0.96, 1),
    #     (284, 191, 98, 5),
    #     (95, 65, 35, 5),
    #     (0.88, 0.92, 0.96, 1)
    #         ]
    lab = [
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        ]
    epo = [
        (51, 40, 29, 18),
        (33, 26, 19, 12),
        (0.7, 0.79, 0.88, 0.97),
        (411, 279, 147, 15),
        (109, 77, 45, 13),
        (0.7, 0.79, 0.88, 0.97)
            ]
    plt.rcParams['font.family'] = 'Arial'
    radar = Radar(fig, tit, lab, epo)

    # radar.plot([13.200, 6.200, 0.978, 244.101, 86.013, 0.978], '-', lw=4, color='tab:red', alpha=1, label='DT', marker='o', markersize=10)
    # radar.plot([23.689, 13.275, 0.930, 242.94, 84.887, 0.930], '-', lw=4, color='tab:blue', alpha=1, label='MLP', marker='s', markersize=10)
    # radar.plot([28.370, 17.266, 0.900, 22.891, 17.383, 0.900], '-', lw=4, color='tab:purple', alpha=1, label='KNN', marker='v', markersize=10)
    # radar.plot([28.576, 11.307, 0.898, 252.529, 85.120, 0.902], '-', lw=4, color='tab:brown', alpha=1, label='SVR', marker='^', markersize=10)
    # radar.plot([16.621, 10.796, 0.965, 242.419, 83.916, 0.965], '-', lw=4, color='tab:green', alpha=1, label='RF', marker='D', markersize=10)
    # radar.plot([6.385, 4.222, 0.994, 5.583, 5.276, 0.994], '-', lw=4, color='tab:orange', alpha=1, label='GB', marker='x', markersize=10)
    # radar.plot([4.738, 2.877, 0.997, 244.115, 85.891, 0.997], '-', lw=4, color='tab:pink', alpha=1, label='XGB', marker='<', markersize=10)
    # radar.plot([19.707, 12.26, 0.952, 243.196, 85.084, 0.952], '-', lw=4, color='tab:cyan', alpha=1, label='LGBM', marker='>', markersize=10)

    radar.plot([32.315,22.045,0.876,334.803,92.946,0.877], '-', lw=4, color='tab:red', alpha=1, label='DT', marker='o', markersize=10)
    radar.plot([34.273,23.387,0.860,324.276,92.353,0.865], '-', lw=4, color='tab:blue', alpha=1, label='MLP', marker='s', markersize=10)
    radar.plot([35.252,22.321,0.852,28.120,23.346,0.859], '-', lw=4, color='tab:purple', alpha=1, label='KNN', marker='v', markersize=10)
    radar.plot([45.303,29.405,0.756,345.541,88.022,0.758], '-', lw=4, color='tab:brown', alpha=1, label='SVR', marker='^', markersize=10)
    radar.plot([26.168,17.640,0.919,341.407,89.748,0.919], '-', lw=4, color='tab:green', alpha=1, label='RF', marker='D', markersize=10)
    radar.plot([19.562,13.328,0.954,15.038,14.317,0.955], '-', lw=4, color='tab:orange', alpha=1, label='GB', marker='x', markersize=10)
    radar.plot([26.178,16.059,0.919,351.951,91.527,0.921], '-', lw=4, color='tab:pink', alpha=1, label='XGB', marker='<', markersize=10)
    radar.plot([27.431,19.198,0.910,343.191,91.332,0.911], '-', lw=4, color='tab:cyan', alpha=1, label='LGBM', marker='>', markersize=10)

    legend = radar.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize = 15)
    plt.setp(legend.get_texts(), fontweight='bold')
    plt.show()