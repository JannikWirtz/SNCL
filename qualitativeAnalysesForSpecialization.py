import os
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from models import Ensemble, Ensemble_single, MetaLearner
from utils import load_cifar10


def extract_models(run_id):
    """
        Method that extracts the individual models from a saved ensemble model
    """
    # get list of files in model folder starting with ensemble in name
    files = os.listdir('./models/')
    files = [file for file in files if run_id in file]
    files = [file for file in files if 'ensemble' in file.lower()]

    if len(files) == 0:
        print("Please fill in the run_id's of the pre-trained models to be used an insure they are located in ./models/.")
        raise Exception('no files found for:' + run_id)
    
    models = []
    weight_dict = torch.load('./models/'+files[-1])
    num_models = weight_dict['layers.0.weight'].shape[0] // 16 # 16 is the number of filters in the first layer
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
    """
        Method that computes the outputs and accuracy of the submodels 
        and the ensemble model for the given dataloader.
    """
    outputs = [[] for _ in range(len(models))]
    all_labels = []
    for batch in dataloader:
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        all_labels.append(labels)
        with torch.no_grad():
            for idx, model in enumerate(models):
                outputs[idx].append(model.forward(inputs, eval=True)[0])
    outputs = [torch.cat(outputs[idx], dim=0) for idx in range(len(models))]
    all_labels = torch.cat(all_labels, dim=0)
    submodelPerClassAcc = [getPerClassAcc(outputs[i], all_labels) for i in range(len(models))]
    if metaModel is None:
        averagedEnsembleOutput = torch.stack(outputs).mean(dim=0)
        ensemblePerClassAcc = getPerClassAcc(averagedEnsembleOutput, all_labels)
        return outputs, averagedEnsembleOutput, submodelPerClassAcc, ensemblePerClassAcc, all_labels        
    
    with torch.no_grad():
        # [models x batch x classes]
        SNCLEnsembleOutput = metaModel.forward(torch.stack(outputs).cuda()) # [10000, 10]
        ensemblePerClassAcc = getPerClassAcc(SNCLEnsembleOutput, all_labels)
        return outputs, SNCLEnsembleOutput, submodelPerClassAcc, ensemblePerClassAcc, all_labels

                
def accExclusionDeltas(subModelOutputs, all_labels, metaModel, ensPerClassAcc, replacementValue=0.1, plotName=None):
    """
        Method that computes the per-class accuracy deltas for each submodel 
        after its exclusion from the ensemble, with the goal of assessing the
        specialization of each submodel.
    """
    exclusionAccs = []
    for modelIdx in range(len(subModelOutputs)):
        origOutputs = torch.stack(subModelOutputs)
        # exclude model by setting its outputs to replacementValue
        origOutputs[modelIdx] = torch.full_like(origOutputs[modelIdx], replacementValue)
        if metaModel is not None:
            withoutModel = metaModel.forward(origOutputs)
        else: # averaged predictions
            withoutModel = torch.mean(origOutputs, dim=0)
        
        # get per Class acc post 'exclusion'
        withoutModelPerClassAcc = getPerClassAcc(withoutModel, all_labels)
        exclusionAccs.append(withoutModelPerClassAcc)

    accLost = []
    for a in exclusionAccs:
        accLost.append(np.array(a) - np.array(ensPerClassAcc))

    # MOST NEGATIVE NUMBERS == SPECIALISATION CLASS (i.e. model is important for this class)
    accLost = np.array(accLost)
    plt.rcParams.update({'font.size': 13})
    plt.clf()
    plt.figure()
    from matplotlib.colors import TwoSlopeNorm
    m = 0.10 # Adjust m as needed
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)  
    plt.imshow(accLost, cmap='RdBu_r', norm=norm)
    plt.xlabel('Class index')
    plt.ylabel('Model index')
    plt.yticks(np.arange(0, len(subModelOutputs), 1), np.arange(1, len(subModelOutputs)+1, 1))
    plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
    plt.colorbar()

    if plotName is not None:
        plt.savefig('./plots/' + plotName +'_'+str(replacementValue) +'.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.savefig('./plots/' + plotName +'_'+str(replacementValue) +'.svg', transparent=True, bbox_inches='tight', pad_inches=0)


def visualizeWeightMatrices(metaModel):
    # show metaModel weights in heatmap plot
    submodelCount = metaModel.layers[0].weight.shape[1]//10
    specializations = None
    for modelIdx in range(submodelCount):
        model_prediction_weighting = metaModel.layers[0].weight.cpu().detach().numpy()[:10,10*modelIdx:(1+modelIdx)*10]
        if specializations is None:
            x = [0.1] * 10
            specializations = np.array(x).dot(model_prediction_weighting)
        else:
            specializations = np.vstack((specializations, np.array(x).dot(model_prediction_weighting)))

    # Create a centered coolwarm colormap
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=np.min(specializations), vcenter=0, vmax=-np.min(specializations))

    weight_matrix = metaModel.layers[0].weight.cpu().detach().numpy()

    weight_matrix = np.transpose(weight_matrix)
    diagonals = []
    colSum = []
    # plot per model weights to svg file
    plt.rcParams.update({'font.size': 20})
    for i in range(-1, submodelCount):
        plt.clf()
        tmp = weight_matrix
        if i >= 0:
            tmp = weight_matrix[10*i:10*(i+1), :]
            diagonals.append(np.diag(tmp))
            colSum.append(np.sum(tmp, axis=0))

        # Create a centered coolwarm colormap
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=np.min(weight_matrix), vcenter=0, vmax=-np.min(weight_matrix))

        plt.imshow(tmp, cmap='RdBu_r', interpolation='none', norm=norm)#, norm=norm)
        plt.xticks(np.arange(0,10),[i for i in np.arange(1,11)])
        plt.yticks(np.arange(0,10),[i for i in np.arange(1,11)])
        plt.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.ylabel('Source class index') # source submodel output
        plt.xlabel('Target class index') # target ensemble output
        plt.colorbar()
        plt.savefig('./plots/metaweights_of_model_'+str(i+1)+'.svg', transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.show()
            

if __name__ == "__main__":
            
    # load cifar-10
    trainloader, testloader = load_cifar10(10000)
    replacementValue = 0.1 # replacing model predictions with equal probability for all classes

    ### M=5, individual model exclusion
    # SNCL ensemble
    run_id = '' # 0.5 ncl
    SNCLEnsembleSubmodels, SNCLEnsembleModel = extract_models(run_id)
    weights = torch.load('./models/'+[file for file in os.listdir('./models/') if run_id in file and 'meta' in file][0])
    metaLayers = int(len(weights.keys())/2)
    metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(SNCLEnsembleSubmodels)).cuda().eval()
    metaModel.load_state_dict(weights)

    # plot weight matrices of SNCL metamodel
    visualizeWeightMatrices(metaModel)

    # SNCL inidividual model removal
    outputs, _,SNCLSubmodelPerClassAcc, SNCLEnsPerClassAcc, all_labels = getOutputs(SNCLEnsembleSubmodels, testloader, metaModel)
    accExclusionDeltas(outputs, all_labels, metaModel, SNCLEnsPerClassAcc, replacementValue, plotName='Specialization_0.5_SNCL_M=5')

    # GNCL ensemble
    GNCLSubmodels, _ = extract_models(run_id='') # 0.5 ncl
    outputs, _, GNCLSubmodelPerClassAcc, GNCLEnsPerClassAcc, all_labels = getOutputs(GNCLSubmodels, testloader)
    # GNCL inidividual model removal
    accExclusionDeltas(outputs, all_labels, None, GNCLEnsPerClassAcc, replacementValue, plotName='Specialization_0.5_GNCL_M=5')

    # Independent ensemble
    indepEnsembleSubmodels, _ = extract_models(run_id='') # 0.0 ncl
    outputs, _, indepSubmodelPerClassAcc, indepEnsPerClassAcc, all_labels = getOutputs(indepEnsembleSubmodels, testloader)
    # Independent ensemble: inidividual model removal
    accExclusionDeltas(outputs, all_labels, None, indepEnsPerClassAcc, replacementValue, plotName='Specialization_0.0_GNCL_M=5')


    ### M=20, individual model exclusion
    # SNCL
    run_id = '' # 0.5 ncl
    SNCLEnsembleSubmodels, SNCLEnsembleModel = extract_models(run_id)
    weights = torch.load('./models/'+[file for file in os.listdir('./models/') if run_id in file and 'meta' in file][0])
    metaLayers = int(len(weights.keys())/2)
    metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(SNCLEnsembleSubmodels)).cuda().eval()
    metaModel.load_state_dict(weights)
    outputs, _,SNCLSubmodelPerClassAcc, SNCLEnsPerClassAcc, all_labels = getOutputs(SNCLEnsembleSubmodels, testloader, metaModel)
    visualizeWeightMatrices(metaModel)
    accExclusionDeltas(outputs, all_labels, metaModel, SNCLEnsPerClassAcc, replacementValue, plotName='Specialization_0.5_ncl_SNCLEnsemble')

    # GNCL ensemble
    GNCLSubmodels, _ = extract_models(run_id='') # 0.5 ncl
    outputs, _, GNCLSubmodelPerClassAcc, GNCLPerClassAcc, _ = getOutputs(GNCLSubmodels, testloader)
    accExclusionDeltas(outputs, all_labels, None, GNCLPerClassAcc, replacementValue, plotName='Specialization_0.5_ncl_averagedEnsemble')

    # Independently trained ensemble
    indepSubmodels, _ = extract_models(run_id='') # 0.0 ncl
    outputs, _, indepSubmodelPerClassAcc, indepEnsPerClassAcc, _ = getOutputs(indepSubmodels, testloader)
    accExclusionDeltas(outputs, all_labels, None, indepEnsPerClassAcc, replacementValue, plotName='Specialization_0.0_ncl_averagedEnsemble')


    ### M=20, scatterplot of per class accuracies of submodels and ensemble
    # SNCL
    run_id = ''
    SNCLEnsembleSubmodels, SNCLEnsembleModel = extract_models(run_id) # 0.9 ncl
    weights = torch.load('./models/'+[file for file in os.listdir('./models/') if run_id in file and 'meta' in file][0])
    metaLayers = int(len(weights.keys())/2)
    metaModel = MetaLearner(num_layers=metaLayers, num_classes=10, num_modules=len(SNCLEnsembleSubmodels)).cuda().eval()
    metaModel.load_state_dict(weights)
    outputs, _,SNCLSubmodelPerClassAcc, SNCLEnsPerClassAcc, all_labels = getOutputs(SNCLEnsembleSubmodels, testloader, metaModel)

    # GNCL ensemble
    GNCLSubmodels, _ = extract_models(run_id='') # 0.9 ncl
    outputs, _, GNCLSubmodelPerClassAcc, GNCLPerClassAcc, _ = getOutputs(GNCLSubmodels, testloader)

    alpha = 0.5
    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    for i in range(len(SNCLEnsembleSubmodels)):
        if i == 0:
            ax.scatter(np.arange(1,11), SNCLSubmodelPerClassAcc[i], color='C0', label='SNCL members', marker='o', alpha=alpha)
            ax.scatter(np.arange(1,11), GNCLSubmodelPerClassAcc[i], color='C1', label='GNCL members', marker='x', alpha=alpha)
            continue
        ax.scatter(np.arange(1,11), SNCLSubmodelPerClassAcc[i], color='C0', marker='o', alpha=alpha)
        ax.scatter(np.arange(1,11), GNCLSubmodelPerClassAcc[i], color='C1', marker='x', alpha=alpha)
    
    ax.plot(np.arange(1,11), SNCLEnsPerClassAcc, label='SNCL ensemble', color='C0')
    ax.plot(np.arange(1,11), GNCLEnsPerClassAcc, label='GNCL ensemble', color='C1')
    ax.set_ylim(0.0, 1)
    ax.set_xlabel('Class index')
    ax.set_ylabel('Test accuracy')
    ax.xaxis.set_ticks(np.arange(1,11))
    plt.legend()
    plt.savefig('plots/submodel_per_class_acc_0.9.png')
    plt.savefig('plots/submodel_per_class_acc_0.9.svg')
    plt.show()
    print('done')

