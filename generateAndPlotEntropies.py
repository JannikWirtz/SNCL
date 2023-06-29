import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.stats as stats
import matplotlib.pyplot as plt
from models import MetaLearner
from qualitativeAnalysesForSpecialization import extract_models
from temperatureScaling import ModelWithTemperature
from utils import load_SVHN, load_CIFAR10_testset
plt.rcParams.update({'font.size': 13})
EntropiesOfMethodAtShift = {}


def load_CIFAR10_corrupted(perturbation_type, intensity):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torch.stack([transform_test(img) for img in datasets[perturbation_type][intensity]]).cuda()
    testset = torch.utils.data.TensorDataset(testset, torch.tensor(labels).cuda())
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)
    return testloader


def confidenceVsAccuracy(outputs, labels):
    covered = set()
    correctCount = 0
    accuracies = []
    for confidenceThreshold in np.arange(0.9, 0.0, -0.1):
        for i in range(len(labels)):
            if i in covered or torch.max(outputs[i]) < confidenceThreshold:
                continue
            covered.add(i)
            if torch.argmax(outputs[i]) == labels[i]:
                correctCount += 1
        accuracies.append(correctCount/(max(1, len(covered))))
    return accuracies[::-1]


def entropy(outputs):
    """
        Shannon entropy quantifies the expected uncertainty inherent in the possible outcomes of a discrete random variable. 
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy() # GPU -> CPU
    return [stats.entropy(output) for output in outputs]
    

def plotEntropyDensityByDataset(snclEntropies, gnclEntropies, indEntropies, label=''):
    # plot the density curves of the entropy of the outputs of each submodel
    plt.figure()    
    plt.clf()
    # compute the density
    snclDensity = stats.gaussian_kde(snclEntropies)
    gnclDensity = stats.gaussian_kde(gnclEntropies)
    indDensity = stats.gaussian_kde(indEntropies)
    # plot the density
    start, end = 0, 2.3
    plt.plot(np.arange(start, end, 0.01), snclDensity(np.arange(start, end, 0.01)))
    plt.plot(np.arange(start, end, 0.01), gnclDensity(np.arange(start, end, 0.01)))
    plt.plot(np.arange(start, end, 0.01), indDensity(np.arange(start, end, 0.01)))
    plt.legend(['SNCL (ours)', 'GNCL', 'Indep.'])
    plt.xlabel('Entropy (Nats)')
    plt.ylabel('Density')
    axes = plt.gca()
    axes.set_ylim([0,4.0])
    axes.set_xlim([start, end])
    plt.grid(True)
    plt.tight_layout()

    # save fig
    plt.savefig('./plots/entropyDensity'+label+'.png')
    plt.savefig('./plots/entropyDensity'+label+'.svg')


def getOutputs(ensemble, dataloader, metaModel=None):
    outputs = []
    all_labels = []
    for batch in dataloader:
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        all_labels.append(labels)
        with torch.no_grad():
            outputs.append(torch.exp(ensemble.forward(inputs, log_softmax=True, eval=True)))
    outputs = torch.cat(outputs, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    if metaModel is None:
        averageEnsembleOutput = outputs.mean(dim=0)
        return outputs, averageEnsembleOutput
    
    with torch.no_grad():
        # [models x batch x classes]
        snclEnsembleOutput = metaModel.forward(outputs.cuda(), log_softmax=True) # [10000, 10]
        snclEnsembleOutput = torch.exp(snclEnsembleOutput) # log-prob. -> prob.
        return outputs, snclEnsembleOutput


def cifar_10_insample_test(snclEnsemble, gnclEnsemble, indEnsemble, metaModel):
    ## In sample test set Cifar-10
    print('CIFAR-10 test')
    testloader = load_CIFAR10_testset()
    _, snclEnsOutputs = getOutputs(snclEnsemble, testloader, metaModel)
    _, gnclEnsOutputs = getOutputs(gnclEnsemble, testloader)
    _, indEnsOutputs = getOutputs(indEnsemble, testloader)

    snclScaled = ModelWithTemperature(snclEnsemble, metaModel)
    snclScaled.set_temperature(testloader)
    gnclScaled = ModelWithTemperature(gnclEnsemble)
    gnclScaled.set_temperature(testloader)
    indScaled = ModelWithTemperature(indEnsemble)
    indScaled.set_temperature(testloader)

    # average softmax output per class across all samples by true label
    snclEnsOutputs_tmp = snclEnsOutputs.cpu().detach().numpy()
    gnclEnsOutputs_tmp = gnclEnsOutputs.cpu().detach().numpy()
    indEnsOutputs_tmp = indEnsOutputs.cpu().detach().numpy()
    sncl = np.array([snclEnsOutputs_tmp[labels == i].mean(axis=0) for i in range(10)])
    gncl = np.array([gnclEnsOutputs_tmp[labels == i].mean(axis=0) for i in range(10)])
    ind = np.array([indEnsOutputs_tmp[labels == i].mean(axis=0) for i in range(10)])

    for i in range(10):
        plt.clf()
        plt.plot(np.arange(0, 10, 1), sncl[i], label='SNCL (ours)')
        plt.plot(np.arange(0, 10, 1), gncl[i], label='GNCL')
        plt.plot(np.arange(0, 10, 1), ind[i], label='Indep.')
        plt.xlabel('Class index')
        plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
        plt.ylabel('Average softmax score per class')
        plt.grid(True)
        plt.legend()
        plt.savefig('./plots/softmaxScores_cifar10_test_class_' + str(i+1) + '.png')
        plt.savefig('./plots/softmaxScores_cifar10_test_class_' + str(i+1) + '.svg')
        # plt.show()

    # plot Accuracy by confidence threshold:
    snclAcc = confidenceVsAccuracy(snclEnsOutputs, labels)
    gnclAcc = confidenceVsAccuracy(gnclEnsOutputs, labels)
    indAcc = confidenceVsAccuracy(indEnsOutputs, labels)

    # plot entropy density
    snclEntropies = entropy(snclEnsOutputs)
    gnclEntropies = entropy(gnclEnsOutputs)
    indEntropies = entropy(indEnsOutputs)
    EntropiesOfMethodAtShift['SNCL_test'] = snclEntropies
    EntropiesOfMethodAtShift['GNCL_test'] = gnclEntropies
    EntropiesOfMethodAtShift['Indep_test'] = indEntropies

    plotEntropyDensityByDataset(snclEntropies, gnclEntropies, indEntropies, label='_CIFAR10_test')
    return (snclScaled, gnclScaled, indScaled), [snclAcc[0], gnclAcc[0], indAcc[0]]


def svhn_ood_test(snclEnsemble, gnclEnsemble, indEnsemble):
    # OOD with SHVN
    print('SVHN OOD Entropy')
    testloader = load_SVHN()
    snclEnsOutputs = snclEnsemble(testloader)
    gnclEnsOutputs = gnclEnsemble(testloader)
    indEnsOutputs = indEnsemble(testloader)

    snclEntropies = entropy(snclEnsOutputs)
    gnclEntropies = entropy(gnclEnsOutputs)
    indEntropies = entropy(indEnsOutputs)
    EntropiesOfMethodAtShift['SNCL_SVHN'] = snclEntropies
    EntropiesOfMethodAtShift['GNCL_SVHN'] = gnclEntropies
    EntropiesOfMethodAtShift['Indep_SVHN'] = indEntropies
    plotEntropyDensityByDataset(snclEntropies, gnclEntropies, indEntropies, label='_SVHN')


def cifar_10_corrupted_test(snclEnsemble, gnclEnsemble, indEnsemble, inSampleResults):
    # Shifted CIFAR-10
    accuracies = {}
    accuracies[0] = inSampleResults[1]

    for intensity in range(1, 6):
        print('Processing intensity ' + str(intensity) + '...')
        snclEntropies = []
        gnclEntropies = []
        indEntropies = []
        snclAccs = [] 
        gnclAccs = []
        indAccs = []
        for perturbation_type in perturbation_types:
            testloader = load_CIFAR10_corrupted(perturbation_type, intensity)
            snclEnsOutputs = snclEnsemble(testloader)
            gnclEnsOutputs = gnclEnsemble(testloader)
            indEnsOutputs = indEnsemble(testloader)

            snclEntropies = snclEntropies + entropy(snclEnsOutputs)
            gnclEntropies = gnclEntropies + entropy(gnclEnsOutputs)
            indEntropies = indEntropies + entropy(indEnsOutputs)

        EntropiesOfMethodAtShift['SNCL_'+str(intensity)] = snclEntropies
        EntropiesOfMethodAtShift['GNCL_'+str(intensity)] = gnclEntropies
        EntropiesOfMethodAtShift['Indep_'+str(intensity)] = indEntropies
            
        snclAccs = [acc[0] for acc in snclAccs]
        gnclAccs = [acc[0] for acc in gnclAccs]
        indAccs = [acc[0] for acc in indAccs]
        accuracies[intensity] = [snclAccs, gnclAccs, indAccs]

        plotEntropyDensityByDataset(snclEntropies, gnclEntropies, indEntropies, label='_shifted_intensity='+str(intensity))

    # plot
    plotEntropyDensitiesPerModel()


def plotEntropyDensitiesPerModel():
    colors = ['C0', 'C1', 'C2']
    linetypes = ['-', '--', ':']
    methods = ['SNCL_', 'GNCL_', 'Indep_']
    start, end = 0, 2.3
    for i, method in enumerate(methods):
        # clear plot
        plt.figure()    
        plt.clf()
        for j, dataset in enumerate(['test', '5', 'SVHN']):
            densityFunction = stats.gaussian_kde(EntropiesOfMethodAtShift[method + dataset])
            plt.plot(np.arange(start, end, 0.01), densityFunction(np.arange(start, end, 0.01)), color=colors[i], linestyle=linetypes[j])
    
        # plot the densities
        plt.legend(['CIFAR-10 test', 'CIFAR-10C (5)', 'SVHN'])
        plt.xlabel('Entropy (Nats)')
        plt.ylabel('Density')
        axes = plt.gca()
        axes.set_ylim([0,4.0])
        axes.set_xlim([start, end])
        plt.grid(True)
        plt.tight_layout()
        # save fig
        plt.savefig('./plots/entropyDensity'+method+'.png')
        plt.savefig('./plots/entropyDensity'+method+'.svg')


def boxplotMetricFromDict(metric, metricName, labels):
    def set_box_color(bp, color):
        plt.setp(bp['boxes'][0], color=color)
        plt.setp(bp['whiskers'][0], color=color)
        plt.setp(bp['caps'][0], color=color)
        plt.setp(bp['medians'][0], color=color)
        plt.setp(bp['boxes'][1:], color='black')
        plt.setp(bp['whiskers'][1:], color='black')
        plt.setp(bp['caps'][1:], color='black')
        plt.setp(bp['medians'][1:], color='black')
        for patch in bp['boxes']:
            patch.set(facecolor=color)

    plt.clf()
    colors = ['C0', 'C1', 'C2']    
    for i in range(len(labels)):
        data = []
        for intensity in range(6):
            data.append(metric[intensity][i]) if intensity > 0 else data.append([metric[0][i]])
        offset = 0.3*i - 0.3
        bp = plt.boxplot(data, positions=np.arange(6)*(len(labels)-1)+offset, widths=0.25, patch_artist=True)
        set_box_color(bp, colors[i])

    ticks = range(6)

    # draw temporary red and blue lines and use them to create legend
    for i in range(len(labels)):
        plt.plot([], c=colors[i], label=labels[i])
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ['test'] + list(ticks)[1:])
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel(metricName)
    plt.xlabel('Shift Intensity')
    plt.savefig('./plots/boxplot_' + metricName + '.png')
    plt.savefig('./plots/boxplot_' + metricName + '.svg')


if __name__ == '__main__':
    # Perturbation types of shifted cifar-10C dataset:
    # Hendrycks, D. and Dietterich, T. Benchmarking neural network robustness to common corruptions
    # and perturbations. In ICLR, 2019.
    perturbation_types = ['saturate', 'spatter', 'gaussian_blur', 'shot_noise', 'gaussian_noise',
                        'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 
                        'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'spatter', 'speckle_noise',
                        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    # pre-load the cifar-10 corrupted datasets
    datasets = {}
    for perturbation_type in perturbation_types:
        datasets[perturbation_type] = {}
        images = np.load('D:/Cifar-10-C/' + perturbation_type + '.npy')
        for intensity in range(5):
            datasets[perturbation_type][intensity+1] = images[10000*intensity:10000*(intensity+1)]
    labels = np.load('D:/Cifar-10-C/labels.npy')[:10000]

    # load models
    # TODO: fill in run_id's of previously trained models located in ./models/
    print('Please fill in the run_id of the pre-trained models to be used.')
    for LAMBDA in [0.5, 0.9]:
        for M in [5, 20]:
            if M == 20:
                averagedINDEnsembleSubmodels, indEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                if LAMBDA == 0.5:
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    sncl_run_id = 'INSERT_RUN_ID_HERE'
                elif LAMBDA == 0.9:
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    sncl_run_id = 'INSERT_RUN_ID_HERE'
            elif M == 5:
                averagedINDEnsembleSubmodels, indEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                if LAMBDA == 0.5:
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    sncl_run_id = 'INSERT_RUN_ID_HERE'
                elif LAMBDA == 0.9: 
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    sncl_run_id = 'INSERT_RUN_ID_HERE'
            

            # load sncl ensemble with metamodel
            snclEnsembleSubmodels, snclEnsemble = extract_models(sncl_run_id)
            weights = torch.load('./models/'+[file for file in os.listdir('./models/') if sncl_run_id in file and 'meta' in file][0])
            metaLayers = int(len(weights.keys())/2)
            metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(snclEnsembleSubmodels)).cuda().eval()
            metaModel.load_state_dict(weights)

            # Produce ouputs and plots
            scaledModels, inSampleResults = cifar_10_insample_test(snclEnsemble, gnclEnsemble, indEnsemble, metaModel)
            snclEnsemble, gnclEnsemble, indEnsemble = scaledModels
            svhn_ood_test(snclEnsemble, gnclEnsemble, indEnsemble)
            cifar_10_corrupted_test(snclEnsemble, gnclEnsemble, indEnsemble, inSampleResults)

            # save entropies to .npz files
            methods = ['SNCL_', 'GNCL_', 'Indep_']
            for method in methods:
                for dataset in ['test', '5', 'SVHN', '1', '2', '3','4']:
                    fname = './data/entropies_' + method + dataset +'_M='+str(M) + '_lam=' + str(LAMBDA)  +'.npz'
                    np.savez(fname, entropies=EntropiesOfMethodAtShift[method + dataset])
