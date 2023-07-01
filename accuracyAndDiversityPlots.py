import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.size': 13})
PATH = 'INSERT_PATH_HERE'
SHOW = False

# REQUIRES: NEPTUNE.AI CSV EXPORT
files = [PATH+f for f in os.listdir(PATH) if f.startswith('PROJECT_NAME')]
recent = max(files, key=os.path.getctime) # most recent result export
df = pd.read_csv(recent)

def plotNCLComparison(ncl_types, dataset, metric='acc', basemodels=5, ncl_weights=[0, 0.25, 0.5, 0.75, 0.9, 1]):
    plt.clf()
    colors = ['C0', 'C1', 'C2']
    max_metric_avg = [[] for _ in range(len(ncl_types))]
    max_metric_std = [[] for _ in range(len(ncl_types))]
    for i, type in enumerate(ncl_types):
        df_type = df[df['parameters/ncl_type'] == type]
        df_type = df_type[df_type['parameters/dataset'] == dataset]
        df_type = df_type[df_type['parameters/base_models'] == basemodels]
        for weight in ncl_weights:
            df_base = df_type[df_type['parameters/ncl_weight'] == weight]
            if metric == 'acc':
                if 'stacked' in type:
                    max_metric_avg[i].append(df_base['meta_acc (max)'].mean())
                    max_metric_std[i].append(df_base['meta_acc (max)'].std())
                else:
                    max_metric_avg[i].append(df_base['ens_acc (max)'].mean())
                    max_metric_std[i].append(df_base['ens_acc (max)'].std())
            elif metric == 'ce':
                if 'stacked' in type:
                    max_metric_avg[i].append(df_base['test/stacked_ce (last)'].mean())
                    max_metric_std[i].append(df_base['test/stacked_ce (last)'].std())
                else:
                    max_metric_avg[i].append(df_base['test/ensemble_ce (last)'].mean())
                    max_metric_std[i].append(df_base['test/ensemble_ce (last)'].std())
            elif metric == 'disag':
                max_metric_avg[i].append(df_base['disag. (last)'].mean())
                max_metric_std[i].append(df_base['disag. (last)'].std())
            elif metric == 'kld':
                max_metric_avg[i].append(df_base['test/KL-divergence (last)'].mean())
                max_metric_std[i].append(df_base['test/KL-divergence (last)'].std())
            elif metric == 'ind-acc':
                max_metric_avg[i].append(df_base['test/accuracy (last)'].mean())
                max_metric_std[i].append(df_base['test/accuracy (last)'].std())

    for i, type in enumerate(ncl_types):
        l = 'SNCL (ours)' if 'stacked' in type else 'GNCL'
        plt.errorbar(ncl_weights, max_metric_avg[i], yerr=max_metric_std[i], label=l, color=colors[i], capsize=3)
    plt.xlabel('λ')
    if metric == 'acc':
        plt.ylabel('Test Accuracy')
    elif metric == 'ce':
        plt.ylabel('Cross-entropy')
    elif metric == 'disag':
        plt.ylabel('Avg. pair-wise disagreement rate')
    elif metric == 'kld':
        plt.ylabel('Avg. pair-wise KL-Divergence')
    elif metric == 'ind-acc':
        plt.ylabel('Average submodel test accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots\\comparison_{}_{}_basemodels={}.png'.format(dataset, metric, basemodels))
    plt.savefig('plots\\comparison_{}_{}_basemodels={}.svg'.format(dataset, metric, basemodels))
    if SHOW:
        plt.show()
    return max_metric_avg


def plotBasemodelComparison(ncl_types, dataset, ncl_weight, metric='acc', basemodels=[3,5,10,15,20]):
    plt.clf()
    colors = ['C0', 'C1', 'C2']
    max_metric_avg = [[] for _ in range(len(ncl_types)+1)] # +1 for GNCL + meta, without interactive training
    max_metric_std = [[] for _ in range(len(ncl_types)+1)]
    for i, type in enumerate(ncl_types):
        df_type = df[df['parameters/ncl_type'] == type]
        df_type = df_type[df_type['parameters/dataset'] == dataset]
        df_type = df_type[df_type['parameters/ncl_weight'] == ncl_weight]
        for base in basemodels:
            df_base = df_type[df_type['parameters/base_models'] == base]

            if metric == 'acc':
                if 'SNCL' in type:
                    max_metric_avg[i].append(df_base['meta_acc (max)'].mean())
                    max_metric_std[i].append(df_base['meta_acc (max)'].std())
                else:
                    max_metric_avg[i].append(df_base['ens_acc (max)'].mean())
                    max_metric_std[i].append(df_base['ens_acc (max)'].std())
                    max_metric_avg[i+1].append(df_base['meta_acc (max)'].mean())
                    max_metric_std[i+1].append(df_base['meta_acc (max)'].std())

            elif metric == 'ce':
                if 'SNCL' in type:
                    max_metric_avg[i].append(df_base['test/stacked_ce (last)'].mean())
                    max_metric_std[i].append(df_base['test/stacked_ce (last)'].std())
                else:
                    max_metric_avg[i].append(df_base['test/ensemble_ce (last)'].mean())
                    max_metric_std[i].append(df_base['test/ensemble_ce (last)'].std())
                    max_metric_avg[i+1].append(df_base['test/stacked_ce (last)'].mean())
                    max_metric_std[i+1].append(df_base['test/stacked_ce (last)'].std())
            elif metric == 'disag':
                max_metric_avg[i].append(df_base['disag. (last)'].mean())
                max_metric_std[i].append(df_base['disag. (last)'].std())
            elif metric == 'kld':
                max_metric_avg[i].append(df_base['test/KL-divergence (last)'].mean())
                max_metric_std[i].append(df_base['test/KL-divergence (last)'].std())
            elif metric == 'ind-acc':
                max_metric_avg[i].append(df_base['test/accuracy (last)'].mean())
                max_metric_std[i].append(df_base['test/accuracy (last)'].std())

    for i, type in enumerate(ncl_types):
        l = 'SNCL (ours)' if 'stacked' in type else 'GNCL'
        plt.errorbar(basemodels, max_metric_avg[i], yerr=max_metric_std[i], label=l, color=colors[i], capsize=3)
        if i == 1 and metric == 'acc':
            plt.errorbar(basemodels, max_metric_avg[i+1], yerr=max_metric_std[i+1], label='GNCL + meta-model', color=colors[i+1],capsize=3)
    plt.xlabel('Ensemble members M')
    if metric == 'acc':
        plt.ylabel('Test Accuracy')
    elif metric == 'ce':
        plt.ylabel('Cross-entropy')
    elif metric == 'disag':
        plt.ylabel('Avg. pair-wise disagreement rate')
    elif metric == 'kld':
        plt.ylabel('Avg. pair-wise KL-Divergence')
    elif metric == 'ind-acc':
        plt.ylabel('Average submodel test accuracy')
    # plt.title('Comparison of ensemble methods (λ={})'.format(ncl_weight))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots\\comparison_{}_{}_{}.png'.format(dataset, metric, ncl_weight))
    plt.savefig('plots\\comparison_{}_{}_{}.svg'.format(dataset, metric, ncl_weight))
    if SHOW:
        plt.show()
    return max_metric_avg


# print list of headers in table
# print(df.columns)
keys = ['Id', 'Creation Time', 'parameters/dataset', 'meta_acc (last)', 'ens_acc (last)', 'test/stacked_vs_averaged_delta (last)', 'disag. (last)', 
        'parameters/ncl_weight',  'parameters/ncl_type', 'test/dummy_models (last)', 'meta_acc (max)', 'ens_acc (max)', 'test/ensemble_ce (last)',
        'test/stacked_ce (last)', 'runID', 'test/stacked_mse (last)', 'trialID', 'parameters/base_models','test/accuracy (last)', 'test/KL-divergence (last)']

datasets = ['CIFAR-10', 'CIFAR-100']

# heatmap plots (ncl_weight x ensemble size) x Accuracy in color
for dataset in datasets:
    ncl_weight_x_base_models = []
    for metric in ['acc', 'disag', 'kld', 'ce', 'ind-acc']:
        for i, ncl_weight in enumerate([0, 0.25, 0.5, 0.75, 0.9, 1]):
            tmp = plotBasemodelComparison(ncl_types=['SNCL','GNCL'], dataset=dataset, metric=metric, ncl_weight=ncl_weight)
            

    for metric in ['acc', 'disag','kld', 'ce', 'ind-acc']:
        for basemodels in [3, 5, 10, 15, 20]:
            tmp = plotNCLComparison(ncl_types=['SNCL','GNCL'], dataset=dataset, metric=metric, basemodels=basemodels)
            if metric == 'acc':
                ncl_weight_x_base_models.append(tmp)

    ncl_weight_x_base_models = np.array(ncl_weight_x_base_models)

    SNCL_results = ncl_weight_x_base_models[:,0]
    GNCL_results = ncl_weight_x_base_models[:,1]


    # subtract and show diff
    diff = SNCL_results - GNCL_results
    fig, ax = plt.subplots(figsize=(8,6))
    
    norm = TwoSlopeNorm(vmin=-0.08, vcenter=0, vmax=0.08)  # Adjust vmin, vcenter, and vmax as needed
    im = ax.imshow(diff, cmap='RdBu_r', norm=norm)

    # plot heatmap (4x6) with colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels([3, 5, 10, 15, 20])
    ax.set_xticklabels([0, 0.0, 0.25, 0.5, 0.75, 0.9, 1])
    plt.xlabel('λ')
    plt.ylabel('Ensemble members M')
    plt.savefig('plots\\diff_SNCL_test_acc_lambda_M_{}.png'.format(dataset), bbox_inches='tight')
    plt.savefig('plots\\diff_SNCL_test_acc_lambda_M_{}.svg'.format(dataset), bbox_inches='tight')

