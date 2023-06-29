import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from MAPIE.mapie.classification import MapieClassifier
from MAPIE.mapie.metrics import classification_coverage_score
from sklearn.metrics import accuracy_score
from qualitativeAnalysesForSpecialization import extract_models
from models import MetaLearner
from generateAndPlotEntropies import ModelWithTemperature
from utils import load_CIFAR10_testset
plt.rcParams.update({'font.size': 13})

# Adapted from: https://github.com/scikit-learn-contrib/MAPIE/blob/master/notebooks/classification/Cifar10.ipynb
class PytorchToMapie():
    """
    Class to make a pytorch model compatible
    with MAPIE. To do so, this class create fit, predict,
    predict_proba and _sklearn_is_fitted_ attributes to the model.
    """
    def __init__(self, outputs, calibIndices, testIndices):
        self.pred_proba = None
        self.model = None
        self.trained_ = True
        self.classes_ = range(10)
        self.caliIndices = calibIndices
        self.testIndices = testIndices
        self.outputs = outputs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == len(self.caliIndices):
            return self.outputs[self.caliIndices]
        elif len(X) == len(self.testIndices):
            return self.outputs[self.testIndices]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == len(self.caliIndices):
            return self.outputs[self.caliIndices].argmax(axis=1)
        elif len(X) == len(self.testIndices):
            return self.outputs[self.testIndices].argmax(axis=1)

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        self.model = model
        self.trained_ = True
        self.classes_ = 10


def count_null_set(y: np.ndarray) -> int:
    count = 0
    for pred in y[:, :]:
        if np.sum(pred) == 0:
            count += 1
    return count
   

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
        return outputs, averageEnsembleOutput.cpu().detach().numpy()
    
    with torch.no_grad():
        # [models x batch x classes]
        snclEnsembleOutput = metaModel.forward(outputs.cuda(), log_softmax=True) # [10000, 10]
        snclEnsembleOutput = torch.exp(snclEnsembleOutput) # logits -> probabilities
        return outputs, snclEnsembleOutput.cpu().detach().numpy()
    

def plot_coverages_and_widths(alpha, coverages, setSizes, labels, filename, seperate=True):
    if seperate:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        for i in range(len(setSizes)):
            axs.errorbar(1 - alpha, np.mean(setSizes[i], axis=0), yerr=np.std(setSizes[i], axis=0), capsize=3, alpha=0.75)
        axs.set_xlabel("1 - alpha")
        axs.set_ylabel("Average size of prediction sets")
        axs.legend(labels)
        axs.set_ylim(1, 4,3)
        plt.tight_layout()
        plt.savefig(filename + '_widths.png')
        plt.savefig(filename + '_widths.svg')
        # plt.show()

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        for i in range(len(coverages)):
            axs.errorbar(1 - alpha, np.mean(coverages[i], axis=0), yerr=np.std(coverages[i], axis=0), capsize=3, alpha=0.75)
        axs.set_xlabel("1 - alpha")
        axs.set_ylabel("Marginal Coverage")
        axs.plot(1 - alpha, 1 - alpha, label="x=y", color="black")
        axs.legend(['x=y'] + labels)
        axs.set_ylim(0.8, 1)
        plt.tight_layout()
        plt.savefig(filename + '_coverages.png')
        plt.savefig(filename + '_coverages.svg')
        # plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(len(coverages)):
            axs[0].errorbar(1 - alpha, np.mean(coverages[i], axis=0), yerr=np.std(coverages[i], axis=0), capsize=3, alpha=0.75)
            axs[1].errorbar(1 - alpha, np.mean(setSizes[i], axis=0), yerr=np.std(setSizes[i], axis=0), capsize=3, alpha=0.75)
        axs[0].set_xlabel("1 - alpha")
        axs[0].set_ylabel("Marginal Coverage")
        axs[0].plot(1 - alpha, 1 - alpha, label="x=y", color="black")
        axs[0].legend(['x=y'] + labels)
        axs[1].set_xlabel("1 - alpha")
        axs[1].set_ylabel("Average size of prediction sets")
        axs[1].legend(labels)
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.svg')
        # plt.show()

def plot_coverages_per_class(alpha, coverages, labels, filename):
    # Composite plot
    columns = 5
    fig, axs = plt.subplots(2, columns, figsize=(12, 10))
    for i, classIdx in enumerate(range(10)):
        for modelIdx in range(len(coverages)):
            axs[i // columns, i % columns].errorbar(1 - alpha, np.mean(coverages[modelIdx][:, classIdx], axis=0), yerr=np.std(coverages[modelIdx][:, classIdx], axis=0), capsize=3, alpha=0.75)
            axs[i // columns, i % columns].set_xlabel("1 - alpha")
            axs[i // columns, i % columns].set_ylabel("Conditional Coverage")
        axs[i // columns, i % columns].plot(1 - alpha, 1 - alpha, color="black")
        axs[i // columns, i % columns].legend(['x=y'] + labels)
        axs[i // columns, i % columns].set_title("Class " + str(classIdx+1))
        axs[i // columns, i % columns].set_ylim(0.8, 1)
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.svg')
    # plt.show()
    # Per class plots for thesis
    for classIdx in range(10):
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        for modelIdx in range(len(coverages)):
            axs.errorbar(1 - alpha, np.mean(coverages[modelIdx][:, classIdx], axis=0), yerr=np.std(coverages[modelIdx][:, classIdx], axis=0), capsize=3, alpha=0.75)
            axs.set_xlabel("1 - alpha")
            axs.set_ylabel("Conditional Coverage")
        axs.plot(1 - alpha, 1 - alpha, color="black")
        axs.legend(['x=y'] + labels)
        axs.set_ylim(0.8, 1)
        plt.tight_layout()
        plt.savefig(filename + '_class='+str(classIdx)+'.png')
        plt.savefig(filename + '_class='+str(classIdx)+'_widths.svg')
        # plt.show()


if __name__ == '__main__':
    ### Parameters
    calibrationSetSize = 2000
    trials = 20
    alphaRange = np.arange(0.01, 0.3, 0.01)
    for LAMBDA in [0.5, 0.9]:
        for M in [5, 20]:
            if M == 20:            
                averagedINDEnsembleSubmodels, indEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                if LAMBDA == 0.5: # M=20, ncl_weight=0.5
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    run_id = 'INSERT_RUN_ID_HERE'
                elif LAMBDA == 0.9: # M=20, ncl_weight=0.9
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    run_id = 'INSERT_RUN_ID_HERE'

            elif M == 5:
                averagedINDEnsembleSubmodels, indEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                if LAMBDA == 0.5:
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    run_id = 'INSERT_RUN_ID_HERE' 
                elif LAMBDA == 0.9:
                    averagedGNCLEnsembleSubmodels, gnclEnsemble = extract_models(run_id='INSERT_RUN_ID_HERE')
                    run_id = 'INSERT_RUN_ID_HERE'

            snclEnsembleSubmodels, snclEnsemble = extract_models(run_id)
            # load metamodel
            weights = torch.load('./models/'+[file for file in os.listdir('./models/') if run_id in file and 'meta' in file][0])
            metaLayers = int(len(weights.keys())/2)
            metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(snclEnsembleSubmodels)).cuda().eval()
            metaModel.load_state_dict(weights)

            # load testset
            testloader = load_CIFAR10_testset()
            labels = np.array(testloader.dataset.targets)

            # use the first 1000 samples out of the 10000 test samples for calibration (determining q thresholds)
            snclCoverages = [[] for _ in range(trials)] # marginal coverages
            gnclCoverages = [[] for _ in range(trials)] # marginal coverages
            indCoverages = [[] for _ in range(trials)] # marginal coverages
            snclCoveragesPerClass = [[] for _ in range(trials)]
            gnclCoveragesPerClass = [[] for _ in range(trials)]
            indCoveragesPerClass = [[] for _ in range(trials)]
            snclAverageSetSizes = [[] for _ in range(trials)]
            gnclAverageSetSizes = [[] for _ in range(trials)]
            indAverageSetSizes = [[] for _ in range(trials)]
            method_params = {"cumulated_score": ("cumulated_score", True)} # adaptive prediction sets
            
            for name, (method, include_last_label) in method_params.items():
                for trial in range(trials):
                    print("Trial: ", (1+trial), "/", trials)
                    calibrationIndices = np.random.choice(len(testloader.dataset), size=calibrationSetSize, replace=False)
                    testIndices = np.array([i for i in range(len(testloader.dataset)) if i not in calibrationIndices])

                    X_calib = testloader.dataset.data[calibrationIndices]
                    y_calib = labels[calibrationIndices]
                    X_test = testloader.dataset.data[testIndices]
                    y_test = labels[testIndices]

                    # new calibration dataloader with selected indices
                    calibrationSet = torch.utils.data.Subset(testloader.dataset, calibrationIndices)
                    calibrationLoader = torch.utils.data.DataLoader(calibrationSet, batch_size=1000)

                    snclScaled = ModelWithTemperature(snclEnsemble, metaModel)
                    snclScaled.set_temperature(calibrationLoader)
                    snclEnsembleOutput = np.array(snclScaled(testloader).cpu().detach())
                    gnclScaled = ModelWithTemperature(gnclEnsemble)
                    gnclScaled.set_temperature(calibrationLoader)
                    averagedGNCLEnsembleOutput = np.array(gnclScaled(testloader).cpu().detach())
                    indScaled = ModelWithTemperature(indEnsemble)
                    indScaled.set_temperature(calibrationLoader)
                    averagedINDEnsembleOutput = np.array(indScaled(testloader).cpu().detach())

                    snclModel = PytorchToMapie(snclEnsembleOutput, calibrationIndices, testIndices)
                    gnclModel = PytorchToMapie(averagedGNCLEnsembleOutput, calibrationIndices, testIndices)
                    indModel = PytorchToMapie(averagedINDEnsembleOutput, calibrationIndices, testIndices)
                    for modelIdx, model in enumerate([snclModel, gnclModel, indModel]):
                        y_preds, y_pss = {}, {}
                        mapie = MapieClassifier(estimator=model, method=method, cv="prefit", random_state=42)
                        mapie.fit(X_calib, y_calib)
                        y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphaRange, include_last_label=include_last_label)

                        nulls, coverages, accuracies, sizes = {}, {}, {}, {}
                        coverages_per_class = [0 for _ in range(10)]
                    
                        accuracies[name] = accuracy_score(y_test, y_preds[name])
                        print(modelIdx, " Accuracy: ", accuracies[name])
                        coverages[name] = [
                            classification_coverage_score(
                                y_test, y_pss[name][:, :, i]
                            ) for i, _ in enumerate(alphaRange)
                        ]
                        for classIdx in range(10):
                            coverages_per_class[classIdx] = [
                                classification_coverage_score(
                                    y_test[y_test == classIdx], y_pss[name][y_test == classIdx, :, i]
                                ) for i, _ in enumerate(alphaRange)
                            ]
                        sizes[name] = [
                            y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphaRange)
                        ]

                        if modelIdx == 0:
                            snclCoverages[trial] = coverages[name]
                            snclAverageSetSizes[trial] = sizes[name]
                            snclCoveragesPerClass[trial] = coverages_per_class
                        elif modelIdx == 1:
                            gnclCoverages[trial] = coverages[name]
                            gnclAverageSetSizes[trial] = sizes[name]
                            gnclCoveragesPerClass[trial] = coverages_per_class
                        elif modelIdx == 2:
                            indCoverages[trial] = coverages[name]
                            indAverageSetSizes[trial] = sizes[name]
                            indCoveragesPerClass[trial] = coverages_per_class

                snclCoverages = np.array(snclCoverages)
                gnclCoverages = np.array(gnclCoverages)
                indCoverages = np.array(indCoverages)
                snclAverageSetSizes = np.array(snclAverageSetSizes)
                gnclAverageSetSizes = np.array(gnclAverageSetSizes)
                indAverageSetSizes = np.array(indAverageSetSizes)
                snclCoveragesPerClass = np.array(snclCoveragesPerClass)
                gnclCoveragesPerClass = np.array(gnclCoveragesPerClass)
                indCoveragesPerClass = np.array(indCoveragesPerClass)
                filename = './plots/CP_' + method + '_M=' + str(M) + '_lam=' + str(LAMBDA)
                plot_coverages_and_widths(alphaRange, [snclCoverages, gnclCoverages, indCoverages], [snclAverageSetSizes, gnclAverageSetSizes, indAverageSetSizes], ['SNCL (ours)', 'GNCL', 'Indep.'], filename)
                filename = './plots/CP_' + method + '_M=' + str(M) + '_lam=' + str(LAMBDA)
                plot_coverages_and_widths(alphaRange, [snclCoverages, gnclCoverages, indCoverages], [snclAverageSetSizes, gnclAverageSetSizes, indAverageSetSizes], ['SNCL (ours)', 'GNCL', 'Indep.'], filename, True)
                filename = './plots/CP_class_coverages_' + method + '_M=' + str(M) + '_lam=' + str(LAMBDA)
                plot_coverages_per_class(alphaRange, [snclCoveragesPerClass, gnclCoveragesPerClass, indCoveragesPerClass], ['SNCL (ours)', 'GNCL', 'Indep.'], filename)
