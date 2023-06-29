import os
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from models import Ensemble, Ensemble_single, MetaLearner
from utils import load_cifar10
plt.rcParams.update({'font.size': 13})

def extract_models(run_id):
    """
        Method that extracts the individual models from the ensemble model
    """
    # get list of files in model folder starting with ensemble in name
    files = os.listdir('./models/')
    files = [file for file in files if run_id in file]
    files = [file for file in files if 'ensemble' in file.lower()]

    if len(files) == 0:
        raise Exception('no files found for:' + run_id)
    if len(files) > 1:
        raise Exception('multiple files found for:' + run_id)
    
    models = []
    weight_dict = torch.load('./models/'+files[0])
    num_models = weight_dict['layers.0.weight'].shape[0] // 16 # 16 filters in the first layer
    if weight_dict['fc2.bias'].shape[-1] != 10: # skip models trained on cifar-100
        raise Exception('Not a cifar-10 model:' + run_id)

    # transfer n models from ensemble to list of submodels
    for idx in range(num_models):
        models.append(Ensemble_single().cuda().eval())
        models[-1].set_weights(weight_dict, idx)

    ensembleModel = Ensemble((3,32,32), 10, 16, num_models).cuda().eval()
    ensembleModel.load_state_dict(weight_dict)
    return models, ensembleModel


def getPerClassAcc(outputs, labels):
    per_class_acc = []
    for i in range(10):
        per_class_acc.append((outputs[labels==i].argmax(dim=1) == i).float().mean().item())
    return per_class_acc


def getOutputs(models, dataloader, metaModel=None):
    outputs = []
    all_labels = []
    for model in models:
        outputs.append([])
    for batch in dataloader:
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        all_labels.append(labels)
        with torch.no_grad():
            for idx, model in enumerate(models):
                outputs[idx].append(torch.exp(model.forward(inputs, log_softmax=True, eval=True)[0]))
    outputs = [torch.cat(outputs[idx], dim=0) for idx in range(len(models))]
    all_labels = torch.cat(all_labels, dim=0)
    return outputs, all_labels


def removalExperiment(outputs, labels, metaModel=None):
    Trials = 20
    accPerSize = [0 for _ in range(len(outputs)+1)]
    for subset_size in reversed(range(len(outputs)+1)): # select 1 to n-1 models at a time
        if subset_size > len(outputs):
            accPerSize[subset_size] = 1
            continue
        for _ in range(Trials):
            # random subset of models
            subset = np.random.choice(len(outputs), subset_size, replace=False)
            # get ensemble output
            if metaModel:
                # mask outputs of subset; fill with 0.1 per class
                masked_outputs = [outputs[idx] if idx in subset else torch.full_like(outputs[idx], 0.1) for idx in range(len(outputs))]
                masked_outputs = torch.stack(masked_outputs)
                ensemble_output = metaModel.forward(masked_outputs, log_softmax=True)
                # get ensemble accuracy
                ensemble_acc = (ensemble_output.argmax(dim=1) == labels).float().mean().item()
            else:
                ensemble_output = torch.stack(outputs)[subset].mean(dim=0)
                ensemble_acc = (ensemble_output.argmax(dim=1) == labels).float().mean().item()

            accPerSize[subset_size] += ensemble_acc/Trials
        print('Ensemble size:', subset_size,' acc:', accPerSize[subset_size])
    return accPerSize[1:]

# load cifar-10
trainloader, testloader = load_cifar10()

# SNCL ensembles
SNCL_runIds = []
SNCL_runIds.append('INSERT_RUN_ID_HERE') # M=20, ncl_weight=0.5
SNCL_runIds.append('INSERT_RUN_ID_HERE') # M=10, ncl_weight=0.5
SNCL_runIds.append('INSERT_RUN_ID_HERE') # M= 5, ncl_weight=0.5


# Perform removal experiment and plot results
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 13})
for i, run_id in enumerate(SNCL_runIds):
    SNCLEnsembleSubmodels, SNCLEnsembleModel = extract_models(run_id)

    # load metamodel
    weights = torch.load('./models/' + [file for file in os.listdir('./models/') if run_id in file and 'meta' in file][0])
    metaLayers = int(len(weights.keys())/2)
    metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(SNCLEnsembleSubmodels)).cuda().eval()
    metaModel.load_state_dict(weights)
    outputs, all_labels = getOutputs(SNCLEnsembleSubmodels, testloader, metaModel)
    stackAccPerSize = removalExperiment(outputs, all_labels, metaModel)
    plt.plot(range(1,len(stackAccPerSize)+1), stackAccPerSize, label='SNCL (M='+str(len(stackAccPerSize))+', λ=0.5) (ours)', color='C'+str(i+2))

# AVERAGED ENSEMBLES
averagedEnsembleSubmodels, _ = extract_models(run_id='INSERT_RUN_ID_HERE') # M=20, ncl_weight=0.5 (GNCL)
outputs, _ = getOutputs(averagedEnsembleSubmodels, testloader) 
averageAccPerSize = removalExperiment(outputs, all_labels)
plt.plot(range(1,len(averageAccPerSize)+1), averageAccPerSize, label='GNCL (M=20, λ=0.5)', color='C0')

averagedEnsembleSubmodels, _ = extract_models(run_id='INSERT_RUN_ID_HERE') # M=20, ncl_weight=0 (independent)
outputs, _ = getOutputs(averagedEnsembleSubmodels, testloader) 
indepAverageAccPerSize = removalExperiment(outputs, all_labels)
plt.plot(range(1,len(indepAverageAccPerSize)+1), indepAverageAccPerSize, label='Independent', color='C1')

# plot
plt.legend(bbox_to_anchor=(0.5, -0.32), loc='lower center', ncol=3)
plt.ylabel('Test accuracy')
plt.xlabel(r'Member subset size $m$')
plt.grid(True)
plt.xlim(1, 20)
plt.ylim(0.74, 0.88)
plt.savefig('./plots/ensemble_members_dropped_vs_accuracy.svg', bbox_inches='tight')
plt.show()


