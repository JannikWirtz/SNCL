from conformalPredictionWithMAPIE import *
from generateAndPlotEntropies import *
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
plt.rcParams.update({'font.size': 13})


def calculateAUROC(id, ood, entropies, M, LAMBDA, results):
    # calculate AU-ROC for each method; semantic OOD
    print(id + ' vs.' + ood + ' M='+str(M) + ' lam=' + str(LAMBDA))
    plt.figure(figsize=(6, 5))
    plt.clf()
    for method in methods:
        labels = np.concatenate((np.zeros(len(entropies[method + id])), np.ones(len(entropies[method + ood]))))
        scores = np.concatenate((entropies[method + id], entropies[method + ood]))
        # calculate TNR@TPR95
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.argmin(np.abs(tpr - 0.95))
        print(method[:-1] + '\t AU-ROC: ' + str(round(100*roc_auc_score(labels, scores), 2)) + '% TNR@TPR95: ' + str(1-fpr[idx]))
        results[method[:-1] + ',AUROC,' + id + ',' + ood +',' + str(M) + ',' + str(LAMBDA)] = roc_auc_score(labels, scores)
        results[method[:-1] + ',TNR@TPR95,' + id + ',' + ood +',' + str(M) + ',' + str(LAMBDA)] = 1-fpr[idx]
        #plot curves
        plt.plot(fpr, tpr, label=method[:-1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
    plt.plot([0, 1], [0, 1], 'k--', label='x=y')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/OODD_' + id + '_vs_' + ood + '_M='+str(M) + '_lam=' + str(LAMBDA) + '.png')
    plt.savefig('./plots/OODD_' + id + '_vs_' + ood + '_M='+str(M) + '_lam=' + str(LAMBDA) + '.svg')


def entropyDensitiesPerModel(entropies, M, LAMBDA):
    plt.rcParams.update({'font.size': 15})
    colors = ['C0', 'C1', 'C2']
    linetypes = ['-', '--', ':']
    methods = ['SNCL_', 'GNCL_', 'Indep_']
    start, end = 0, 2.2
    for i, method in enumerate(methods):
        # clear plot
        plt.figure(figsize=(5, 5))
        plt.clf()
        for j, dataset in enumerate(['test', '5', 'SVHN']):
            density = stats.gaussian_kde(entropies[method + dataset])
            plt.plot(np.linspace(0, 2.3, 1000), density(np.linspace(0, 2.3, 1000)), color=colors[i], linestyle=linetypes[j], linewidth=1.5)

        # plot the density
        plt.legend(['CIFAR-10 test', 'CIFAR-10C (5)', 'SVHN'])
        plt.xlabel('Entropy (Nats)')
        plt.ylabel('Density')
        axes = plt.gca()
        axes.set_ylim([0,3.5])
        axes.set_xlim([start, end])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./plots/entropyDensity'+method+ 'M='+str(M) + '_lam=' + str(LAMBDA) + '.png')
        plt.savefig('./plots/entropyDensity'+method+ 'M='+str(M) + '_lam=' + str(LAMBDA) + '.svg')
        # plt.show()
    plt.rcParams.update({'font.size': 13})


def plotROCExample():    
    # plot an example roc curve with a random classifier and a perfect classifier
    plt.figure(figsize=(6, 5))
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--', label='random classifier (AUROC=50%)')
    plt.plot([0, 1], [1, 1], 'b', label='perfect classifier (AUROC=100%)')
    plt.plot([0, 0], [0, 1], 'b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')

    # annotate the graph showing better and worse arrows
    l = 0.2
    plt.annotate('', xytext=(0.5, 0.5), xy=(0.5-l, 0.5+l), arrowprops=dict(arrowstyle='->', color='green', lw = 2))
    plt.text(0.4, 0.6, 'better', color='green')
    plt.annotate('', xytext=(0.5, 0.5), xy=(0.5+l, 0.5-l), arrowprops=dict(arrowstyle='->', color='red', lw = 2))
    plt.text(0.6, 0.4, 'worse', color='red')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/OODD_example_roc.png')
    plt.savefig('./plots/OODD_example_roc.svg')
    plt.show()


if __name__ == '__main__':
    ### Parameters
    alphaRange = np.arange(0.01, 0.3, 0.01)
    results = {}

    for LAMBDA in [0.5, 0.9]:
        for M in [5,20]:
            # load entropies from npz file generated with generateAndPlotEntropies.py
            entropies = {}
            methods = ['SNCL_', 'GNCL_', 'Indep_']
            for method in methods[::-1]:
                for dataset in ['test', '1', '2', '3','4', '5', 'SVHN']:
                    fname = './data/entropies_' + method + dataset +'_M='+str(M) + '_lam=' + str(LAMBDA)  +'.npz'
                    data = np.load(fname)

                    entropies[method + dataset] = data['entropies']

            # plot entropies
            entropyDensitiesPerModel(entropies, M, LAMBDA)
            calculateAUROC('test', 'SVHN', entropies, M, LAMBDA, results)
            calculateAUROC('test', '5', entropies, M, LAMBDA, results)
            calculateAUROC('test', '4', entropies, M, LAMBDA, results)
            calculateAUROC('test', '3', entropies, M, LAMBDA, results)
            calculateAUROC('test', '2', entropies, M, LAMBDA, results)
            calculateAUROC('test', '1', entropies, M, LAMBDA, results)

    # save results hashmap to csv
    # turn comma seperated key to columns
    columns = ['method', 'metric', 'id', 'ood', 'M', 'LAMBDA','value']
    df = pd.DataFrame(columns=columns)
    for key in results.keys():
        values = key.split(',')
        df = pd.concat([df, pd.DataFrame([values + [results[key]]], columns=columns)])
    df.to_csv('./data/OODD_results.csv', header=False)
                
    # Plot AUROC for 3 methods and 5 intensities
    for M in [5, 20]:  
        for LAMBDA in [0.5, 0.9]:
            plt.clf()        
            for method in methods:
                plt.plot(range(1,6), [results[method[:-1] + ',AUROC,test,' + str(i) +',' + str(M) + ',' + str(LAMBDA)] for i in range(1,6)], label=method[:-1])
            plt.xlabel('Corruption Intensity')
            plt.ylabel('Average AUROC')

            plt.grid(True)
            plt.tight_layout()
            plt.legend(['SNCL (ours)', 'GNCL', 'Indep.'], loc='lower right')
            plt.savefig('./plots/AUROC_CIFAR-10C_M='+str(M) + '_lam=' + str(LAMBDA) + '.png')
            plt.savefig('./plots/AUROC_CIFAR-10C_M='+str(M) + '_lam=' + str(LAMBDA) + '.svg')
    
    plotROCExample()
