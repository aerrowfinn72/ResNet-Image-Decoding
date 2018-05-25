'''Create figures for results of feature prediction'''

from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

import bdpy.fig as bfig
from bdpy.util import makedir_ifnot
import god_config as config

# Main #################################################################

def main():

    analysis_name = 'GenericObjectDecoding'
    resnet_reindex = config.resnet_reindex
    resnet_true_layers = config.resnet_true_layers

    alexnet_file = os.path.join('results-alexnet', analysis_name + '.pkl')
    resnet_file = os.path.join('results-resnet', analysis_name + '.pkl')
    output_file_featpred = os.path.join('results', config.analysis_name + '_featureprediction.pdf')

    # Load results -----------------------------------------------------
    with open(alexnet_file, 'rb') as f:
        print('Loading %s' % alexnet_file)
        alexnet_results = pickle.load(f)

    with open(resnet_file, 'rb') as f:
        print('Loading %s' % resnet_file)
        resnet_results = pickle.load(f)

    # Figure settings
    plt.rcParams['font.size'] = 7

    # Plot (feature prediction) ----------------------------------------
    fig, axes = plt.subplots(4,2,figsize=(8,9))
    num_plots = range(8)

    # Image
    plotresults(fig, axes, alexnet_results, resnet_results, num_plots)

    # Save the figure
    makedir_ifnot('results')
    plt.savefig(output_file_featpred, dpi=300)
    print('Saved %s' % output_file_featpred)

    plt.show()


# Functions ############################################################

def plotresults(fig, axes, alexnet_results, resnet_results, subplot_index):
    '''Draw results of feature prediction'''

    for feat_a, feat_b, si in zip(config.alexnet_features, config.resnet_features, subplot_index):

        df_a = alexnet_results.loc[alexnet_results["feature"] == feat_a]
        df_a = df_a.loc[:, ["feature", "roi", "mean_profile_correlation_image"]]
        df_a = df_a.reset_index()
        del df_a["index"]

        df_b = resnet_results.loc[resnet_results["feature"] == feat_b]
        df_b = df_b.loc[:, ["feature", "roi", "mean_profile_correlation_image"]]
        df_b = df_b.reset_index()
        del df_b["index"]

        df = pd.concat([df_a, df_b])
        df = df.reset_index()
        del df["index"]
        df = df.replace(config.resnet_features[si], config.resnet_true_layers[si])

        ax = axes[4 - (si % 4) - 1, si // 4]

        g = sns.factorplot(x="roi", y="mean_profile_correlation_image", hue="feature", data=df,
                           size=4, kind="bar", palette="muted", order=config.roi_labels,
                           legend_out=True, ax=ax)

        ax.set_xlabel("")
        ax.set_ylabel("Corr. coeff.")
        ax.set_ylim(0,0.5)
        ax.legend(prop={'size':8})

    # Adjust subplots
    plt.subplots_adjust(wspace=1.5, hspace=1.0)


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
