import os
import json
import glob
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ..util import defines
from ..util import file_handling as fh


def get_tableau_colors():
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20


def plot_broken_line(ax, y, color, alpha=1.0, radius=0.4, connect=False, linewidth=1):
    x = range(len(y))
    for i, yi in enumerate(y):
        ax.plot((x[i]-radius, x[i]+radius), (yi, yi), c=color, alpha=alpha, linewidth=linewidth)
    if connect:
        for i, yi in enumerate(y[:-1]):
            ax.plot((x[i]+radius, x[i+1]-radius), (yi, y[i+1]), c=color, alpha=alpha, linewidth=linewidth)


def make_gate_plots():
    ddir = fh.makedirs(defines.exp_dir, 'rnn', 'bayes_opt_rnn_LSTM_reuse_mod_34_rerun', 'fold0', 'responses')
    text_filename = fh.makedirs(defines.data_dir, 'rnn', 'ngrams_n1_m1_rnn.json')
    output_dir = fh.makedirs(defines.web_dir, 'DRLD', 'gate_plots')

    colours = get_tableau_colors()

    with codecs.open(text_filename, 'r') as input_file:
        all_text = json.load(input_file)

    files = glob.glob(os.path.join(ddir, '*.csv'))
    for f in files:
        base = fh.get_basename(f)
        print base

        i_f_file = os.path.join(ddir, base + '_i_f.npy')
        i_r_file = os.path.join(ddir, base + '_i_r.npy')
        f_f_file = os.path.join(ddir, base + '_f_f.npy')
        f_r_file = os.path.join(ddir, base + '_f_r.npy')
        o_f_file = os.path.join(ddir, base + '_o_f.npy')
        o_r_file = os.path.join(ddir, base + '_o_r.npy')
        text = all_text[base]

        i_f = fh.unpickle_data(i_f_file)[:, 0, :]
        i_r = fh.unpickle_data(i_r_file)[:, 0, :]
        f_f = fh.unpickle_data(f_f_file)[:, 0, :]
        f_r = fh.unpickle_data(f_r_file)[:, 0, :]
        o_f = fh.unpickle_data(o_f_file)[:, 0, :]
        o_r = fh.unpickle_data(o_r_file)[:, 0, :]
        n_el, n_dim = i_f.shape

        f, axes = plt.subplots(6, sharex=True, sharey=True, figsize=(11, 7))
        f.subplots_adjust(hspace=0.2)

        plt.rcParams.update({'font.size': 22})

        gates = [i_f, i_r, f_f, f_r, o_f, o_r]
        legs = ['input (f)', 'input (r)', 'forget (f)', 'forget (r)', 'output (f)', 'output (r)', ]

        x = range(n_el)
        for index, gate in enumerate(gates):
            ax = axes[index]
            color = colours[index//2*2]
            gate_mean = np.mean(gate, axis=1)
            plot_broken_line(ax, gate_mean, color=color, radius=0.5, connect=True, linewidth=2)
            legend = ax.legend([legs[index]], loc='center left', bbox_to_anchor=(1.01, 0.5))
            for label in legend.get_texts():
                label.set_fontsize('small')
            for j in range(n_dim):
                plot_broken_line(ax, gate[:, j], color='grey', alpha=0.3, radius=0.4)
            ax.yaxis.set_ticks([0, 1])

        axes[-1].xaxis.set_ticks(x)
        axes[-1].set_xticklabels(text, rotation=90)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-1, n_el)
        output_filename = fh.make_filename(output_dir, base + '_gates', 'png')
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()


def make_vector_plots():
    ddir = fh.makedirs(defines.exp_dir, 'rnn', 'bayes_opt_rnn_LSTM_reuse_mod_34_rerun', 'fold0', 'responses')
    text_filename = fh.makedirs(defines.data_dir, 'rnn', 'ngrams_n1_m1_rnn.json')
    output_dir = fh.makedirs(defines.web_dir, 'DRLD', 'vector_plots')

    colours = get_tableau_colors()

    with codecs.open(text_filename, 'r') as input_file:
        all_text = json.load(input_file)

    all_c = None
    all_h = None
    seq_lengths = []

    print "Loading data"

    files = glob.glob(os.path.join(ddir, '*.csv'))
    for f in files:
        base = fh.get_basename(f)
        text = all_text[base]
        print base

        c_file = os.path.join(ddir, base + '_c.npy')
        h_file = os.path.join(ddir, base + '_h.npy')
        c = fh.unpickle_data(c_file)[:, 0, :]
        h = fh.unpickle_data(h_file)[:, 0, :]

        n_el, n_dim = c.shape

        n_el = min(n_el, 50)

        if n_el > 0:
            n_hidden = n_dim / 2
            c_f = c[:, :n_hidden]
            c_r = c[:, n_hidden:]
            h_f = h[:, :n_hidden]
            h_r = h[:, n_hidden:]

            X = np.vstack((h_f, h_r, c_f, c_r))

            model = TSNE(n_components=2, random_state=0)
            X2 = model.fit_transform(X)

            scale = np.max(X2[:, 0])

            fig, axes = plt.subplots(n_el, sharex=True, sharey=True, figsize=(6, n_el*2))
            fig.subplots_adjust(hspace=0)

            for i in range(n_el):
                if n_el > 1:
                    ax = axes[i]
                else:
                    ax = axes
                for j in range(4):
                    color = colours[j+6]
                    ax.plot((0, X2[i+n_el*j, 0]), (0, X2[i+n_el*j, 1]), color=color, linewidth=2)
                ax.axis('off')
                ax.text(scale*0.2, 0, text[i], size=13)
            if n_el > 1:
                ax = axes[0]
            else:
                ax = axes
            ax.legend(['hidden (f)', 'hidden (r)', 'memory (f)', 'memory (r)'], bbox_to_anchor=(1.7, 1.2))
            output_filename = fh.make_filename(output_dir, base + '_vectors', 'png')
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close()


def main():
    #make_gate_plots()

    make_vector_plots()

if __name__ == '__main__':
    main()
