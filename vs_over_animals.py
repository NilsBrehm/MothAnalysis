from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

pathname = '/media/brehm/Data/MasterMoth/figs/'
data_set1 = ['2017-11-16-aa', '2017-11-17-aa']  # Carales
data_set2 = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-01-aa', '2017-12-04-aa', '2017-12-05-ab']  # Estigmene
data_set3 = ['2017-11-14-aa', '2018-01-26-aa']  # Creatonotos

gaps = np.arange(1, 16, 1)

for d in range(3):
    if d == 0:
        data_set = data_set1
        color = 'r-o'
    elif d == 1:
        color = 'b-o'
        data_set = data_set2
    else:
        color = 'g-o'
        data_set = data_set3

    vs_all = np.zeros((len(data_set), 15))
    for i in range(len(data_set)):
        filename = pathname + data_set[i] + '/intervals_mas_vs.npy'
        vs = np.load(filename).item()
        for k in range(0, 15):
            vs_all[i, k] = vs[k]['vs']

    vs_mean = np.mean(vs_all, 0)
    vs_sem = np.std(vs_all, 0) / np.sqrt(len(data_set))

    plt.errorbar(gaps, vs_mean, yerr=vs_sem, fmt=color)
    # plt.xlabel('Gap [ms]')
    # plt.ylabel('Vector Strength')
    # plt.title('Vector Strength: Estigmene (n=%s)' % str(len(data_set)))
    # plt.xticks(gaps)
    # plt.yticks(np.arange(0, np.max(vs_mean)+0.1, 0.05))
    # plt.show()

# Save Plot to HDD
plt.xlabel('Gap [ms]')
plt.ylabel('Vector Strength')
plt.xticks(gaps)
plt.yticks(np.arange(0, 0.6, 0.05))
plt.title('VS: Carales(red, n=2), Estigmene(blue, n=5), Creatonotos(green, n=2)')

figname = pathname + "VS_CEC.png"
fig = plt.gcf()
fig.set_size_inches(16, 12)
fig.savefig(figname, bbox_inches='tight', dpi=300)
plt.close(fig)

