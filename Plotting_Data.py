from IPython import embed
import myfunctions as mf
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']

PlotRectIntervals = True
PlotMothIntervals = False
PlotVS = False
PlotFICurves = False
PlotFIField = False

data_name = datasets[0]

for i in range(len(datasets)):  # Loop through all recordings in the list above
    data_name = datasets[i]
    pathname = "/home/brehm/PycharmProjects/mothanlysis/figs/" + data_name + "/"
    try:
        if PlotRectIntervals:
            mf.rect_intervals_plot(data_name)

        if PlotMothIntervals:
            mf.moth_intervals_plot(data_name, 10, 50)

        if PlotVS:
            # # Plot Vector Strength: # #
            # Load vs
            vs = np.load(pathname + 'intervals_mas_vs.npy').item()
            v_strength_01, v_strength_04 = [], []
            gap_01, gap_04 = [], []

            # Plot gap vs vs
            for k in vs:
                if vs[k]['tau'] == 0.0004:  # only look for tau = 0.1 ms
                    v_strength_04 = np.append(v_strength_04, vs[k]['vs'])
                    gap_04 = np.append(gap_04, vs[k]['gap'])
                elif vs[k]['tau'] == 0.0001:
                    v_strength_01 = np.append(v_strength_01, vs[k]['vs'])
                    gap_01 = np.append(gap_01, vs[k]['gap'])

            idx01 = np.argsort(gap_01)
            idx04 = np.argsort(gap_04)

            plt.subplot(2,1,1)
            plt.plot(gap_01[idx01], v_strength_01[idx01], 'k--o')
            plt.title('tau = 0.1 ms')
            plt.ylabel('Vector Strength')

            plt.subplot(2, 1, 2)
            plt.plot(gap_04[idx04], v_strength_04[idx04], 'k--o')
            plt.title('tau = 0.4 ms')
            plt.ylabel('Vector Strength')
            plt.xlabel('gap [s]')

            # Save Plot to HDD
            figname = pathname + "VS_50kHz.png"
            fig = plt.gcf()
            fig.set_size_inches(16, 12)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)

        if PlotFICurves:
            # # Plot FIField and FICurves: # #
            # Load FI Curve Data for given frequency and save the plot to HDD
            freqs_used = np.unique(np.load(pathname + 'frequencies.npy'))/1000
            for f in freqs_used:
                fi = np.load(pathname + 'FICurve_' + str(f) + '.npz')
                mf.plot_ficurve(fi['amplitude_sorted'], fi['mean_spike_count'], fi['std_spike_count'], f, fi['spike_threshold'], pathname, savefig=True)

        if PlotFIField:
            fifield = np.load(pathname + 'FIField_plotdata.npy')
            mf.plot_fifield(fifield, pathname, savefig=True)

    except FileNotFoundError:
        print('File not found')
        continue

print("--- Plotting took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
