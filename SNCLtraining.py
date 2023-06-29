import time
import os
import argparse
import numpy as np
import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import KLDivergence
from models import Ensemble, MetaLearner
from utils import load_cifar10, load_cifar100


# Define hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrstep', type=int, default=15, help='lrstep')
parser.add_argument('--lrgamma', type=int, default=0.1, help='lrgamma')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--base_models', type=int, default=20, help='number of models to train in parallel')
parser.add_argument('--ncl_type', type=str, default='SNCL', help='choose: GNCL, SNCL')
parser.add_argument('--ncl_weight', type=float, default=0.5, help='weight of the ncl penalty')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='learning rate for meta-model')
parser.add_argument('--meta_activation', type=str, default='relu', help='activation for meta-model')
parser.add_argument('--meta_layers', type=int, default=1, help='number of layers for meta-model')
parser.add_argument('--num_trials', type=int, default=1, help='number of trials')
parser.add_argument('--neptuneai', type=str, default='none', help='none | offline | sync')
parser.add_argument('--save_path', type=str, default='./models/', help='path to save the model')
parser.add_argument('--silent', type=bool, default=False)
args = parser.parse_args()


def train_model_NCL(run_id, trial_tracking, dataloaders, models, optimizers, scheduler, num_epochs):
    optimizer, metaOptimizer = optimizers
    model, metaModel = models
    trainloader, testloader = dataloaders
    nll_loss = nn.NLLLoss()
    since = time.time()
    
    epoch_losses = [[]] * args.base_models
    epoch_accs = [[]] * args.base_models
    for epoch in range(num_epochs):
        if not args.silent:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set models to training mode
                metaModel.train()
            else:
                model.eval()   # Set models to evaluation mode
                metaModel.eval()

            running_losses = [0] * args.base_models
            running_corrects = [0] * args.base_models

            # Iterate over data.
            if phase == 'train':
                start = time.time()
                steps = 0
                with torch.set_grad_enabled(phase == 'train'):
                    for batch in trainloader:
                        inputs = batch[0].cuda(non_blocking=False)
                        labels = batch[1].cuda(non_blocking=True)

                        # forward pass submodels
                        output = model(inputs) # [10,100,10] MODELS X BS X CLASSES
                        average_out = torch.mean(output, dim=0) # average ensemble output
                        
                        if args.ncl_type == 'SNCL':
                            ensemble_loss = nll_loss(torch.log(metaModel(output)), labels)
                        else:
                            ensemble_loss = nll_loss(torch.log(average_out), labels)

                        # get individual loss for each submodel
                        module_losses = [nll_loss(torch.log(output[i, ...]), labels) for i in range(args.base_models)]
                        average_submodel_loss = torch.mean(torch.stack(module_losses))

                        # backward + optimize
                        optimizer.zero_grad()
                        training_loss = args.ncl_weight * ensemble_loss + (1 - args.ncl_weight) * average_submodel_loss
                        training_loss.backward()
                        optimizer.step()
                        steps += 1

                        # forward pass meta-model; # always training metamodel for comparison: SNCL vs. (GNCL + meta-model)
                        # Input: [10,100,10] MODELS X BS X CLASSES => Output: [100,10] BS X CLASSES
                        meta_output = metaModel(output.detach()) 
                        # backward + optimize
                        metaOptimizer.zero_grad()
                        training_loss = nll_loss(torch.log(meta_output), labels)
                        training_loss.backward()
                        metaOptimizer.step()

                        # statistics
                        [loss.detach() for loss in module_losses]
                        for i in range(args.base_models): 
                            running_losses[i] += module_losses[i].item() * inputs.size(0)
                            predictions = torch.max(output[i], 1)
                            running_corrects[i] += torch.sum(predictions[1].data == labels.data).item()
                    
            elif phase == 'test':
                stats = {'ensemble_acc': [], 'meta_acc': [], 
                         'ensemble_mse': [], 'ensemble_ce': [], 
                         'meta_mse': [], 'meta_ce': [], 
                         'avg_dis': [], 'avg_kld': []}

                for inputs, labels in testloader:
                    inputs = inputs.cuda(non_blocking=False)
                    labels = labels.cuda(non_blocking=True)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        output = model(inputs, eval=True)
                        average_out = torch.mean(output, dim=0)
                        meta_out  = metaModel(output)
                        module_losses = [nll_loss(torch.log(output[i, ...]), labels) for i in range(args.base_models)]

                        ### STATS ###
                        # Module loss & acc
                        for i in range(args.base_models): # logged to neptune via epoch_loss
                            running_losses[i] += module_losses[i].item() * len(labels)
                            predictions = torch.max(output[i], 1)
                            running_corrects[i] += torch.sum(predictions[1].data == labels.data).item()

                        # Ensemble loss & acc (with and without meta model)
                        labels_enc = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).cuda()
                        _, ens_preds = torch.max(average_out, dim=1)
                        _, meta_preds = torch.max(meta_out, dim=1)

                        # Performance metrics
                        stats['ensemble_ce'].append(nll_loss(torch.log(average_out), labels).item())
                        stats['meta_ce'].append(nll_loss(torch.log(meta_out), labels).item())
                        stats['ensemble_mse'].append(torch.mean((average_out - labels_enc)**2).item())
                        stats['meta_mse'].append(torch.mean((meta_out - labels_enc)**2).item())
                        stats['ensemble_acc'].append(torch.sum(ens_preds == labels.data).item() / len(labels))
                        stats['meta_acc'].append(torch.sum(meta_preds == labels.data).item() / len(labels))

                        # Diversity metrics
                        stats['avg_dis'].append(get_avg_disagreement(args.base_models, output))
                        stats['avg_kld'].append(get_avg_kld(args.base_models, output))
                
                # Neptune logging of stats
                if trial_tracking is not None:
                    neptune_test_log(trial_tracking, stats)
                    trial_tracking["test/meta_vs_averaged_delta"].log(np.mean(stats['meta_acc']-np.mean(stats['ensemble_acc'])))
                
                if not args.silent:
                    print('Ensemble Accuracy: {:.4f}'.format(np.mean(stats['ensemble_acc'])))
                    print('Meta Accuracy: {:.4f}'.format(np.mean(stats['meta_acc'])))
                    print('Delta: {:.4f}'.format(np.mean(stats['meta_acc']-np.mean(stats['ensemble_acc']))))
                    print('*'*50)

            dummy_models_count = 0 # num of submodels that do not surpass random guessing
            for i in range(args.base_models):
                epoch_losses[i].append(running_losses[i] / len(trainloader.dataset) if phase == 'train' else running_losses[i] / len(testloader.dataset))
                epoch_accs[i].append(running_corrects[i] / len(trainloader.dataset) if phase == 'train' else running_corrects[i] / len(testloader.dataset))
                if not args.silent:
                    print('Model {}: {} Loss: {:.4f} Acc: {:.4f}'.format(i+1, phase, epoch_losses[i][-1], epoch_accs[i][-1]))
                if epoch_accs[i][-1] < 1/args.num_classes*1.2: # count dummy models
                    dummy_models_count += 1

            # average model performance
            average_loss = np.mean([epoch_losses[i][-1] for i in range(args.base_models)])
            average_acc = np.mean([epoch_accs[i][-1] for i in range(args.base_models)])
            
            # Log the aggregated losses separately to neptune.ai
            if trial_tracking is not None:
                trial_tracking["train/loss" if phase == 'train' else "test/loss"].log(average_loss)
                trial_tracking["train/accuracy" if phase == 'train' else "test/accuracy"].log(average_acc)
                if phase == 'test':
                    trial_tracking["train/time_per_step"].log((time.time() - start) / steps)
                    trial_tracking["train/time_per_epoch"].log(time.time() - start)
                    # record dummy models, encounterd in end-to-end training
                    trial_tracking["test/dummy_models"].log(dummy_models_count)

            if phase == 'test':
                scheduler.step() # lr scheduler step
                if not args.silent:
                    print('Dummy models: {}'.format(dummy_models_count))
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # save the model weights 
    suffix = 'CIF100' if args.num_classes == 100 else 'CIF10'
    name = 'ensemble_{}_{}_{}_{}.pt'.format(run_id, i, args.ncl_weight, suffix)
    torch.save(model.state_dict(), os.path.join(args.save_path, name))
    if trial_tracking is not None:
        trial_tracking["model_weights"].upload(os.path.join(args.save_path, name))
    name ='meta_{}_{}_{}.pt'.format(run_id, args.ncl_weight, suffix)
    torch.save(metaModel.state_dict(), os.path.join(args.save_path, name))
    if trial_tracking is not None:
        trial_tracking["meta_model_weights"].upload(os.path.join(args.save_path, name))
    return model


def neptune_test_log(run, stats):
    # loss of GNCL/(SNCL submodels with averaging)
    run["test/ensemble_mse"].log(np.mean(stats['ensemble_mse']))
    run["test/ensemble_ce"].log(np.mean(stats['ensemble_ce']))
    run["test/ensemble_acc"].log(np.mean(stats['ensemble_acc']))
    # loss of SNCL/(GNCL+meta)
    run["test/meta_mse"].log(np.mean(stats['meta_mse']))
    run["test/meta_ce"].log(np.mean(stats['meta_ce']))
    run["test/meta_acc"].log(np.mean(stats['meta_acc']))
    # diversity metrics of submodels
    run["test/disagreement"].log(torch.mean(torch.stack(stats['avg_dis'])))
    run["test/KL-divergence"].log(torch.mean(torch.stack(stats['avg_kld'])))


def get_avg_kld(num_models, outputs):
    outputs[outputs <= 1e-12] = 1e-12 # for numerical stability; otherwise, KLDivergence can inf
    kld = KLDivergence(False, 'mean').cuda()
    avg_kld = [0 for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                avg_kld[i] = kld(outputs[i], outputs[j])/(num_models-1)
    return torch.mean(torch.stack(avg_kld))


def disagreement_rate(p, q):
    p = torch.argmax(p, axis=1)
    q = torch.argmax(q, axis=1)
    return torch.sum(p != q) / len(p)


def get_avg_disagreement(num_models, outputs):
    avg_disag = [0 for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                avg_disag[i] += torch.mean(disagreement_rate(outputs[i], outputs[j])/(num_models-1))
    return torch.mean(torch.stack(avg_disag))


def main():
    # define the device
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, abort!")
        exit()

    if args.silent:
        print("Supressing performance prints")

    # get data loaders with normalization and basic data augmentation
    if args.num_classes == 10: # CIFAR10
        trainloader, testloader = load_cifar10(args.batch_size, shuffle=True, include_data_augmentation=True)
    elif args.num_classes == 100: # CIFAR100
        trainloader, testloader = load_cifar100(args.batch_size, shuffle=True, include_data_augmentation=True)

    # create the model
    model = Ensemble((3,32,32), args.num_classes, 16, args.base_models).cuda()

    # create Metalearner for stacking
    metaModel = MetaLearner(args.num_classes, args.base_models, args.meta_layers, args.meta_activation).cuda()
    metaOptimizer = optim.Adam(metaModel.parameters(), lr=args.meta_lr)

    # define the and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgamma)

    # init neptune tracking
    run_id = time.strftime("%Y%m%d-%H%M%S") + '_' + str(np.random.randint(1000, 9999))
    print('RUN-ID:', run_id)

    trial_tracking = None
    if args.neptuneai != 'none':
        # Neptune.ai experiment tracking
        trial_tracking = neptune.init_run(
            project="INSERT_PROJECT_NAME",
            api_token="INSERT_API_TOKEN",
            mode=args.neptuneai # offline, sync
        )
        
        params={
            'batch_size': args.batch_size,
            'epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'base_models': args.base_models,
            'ncl_type': args.ncl_type,
            'ncl_weight': float(args.ncl_weight),
            'meta_lr': args.meta_lr,
            'meta_activation': args.meta_activation,
            'meta_layers': args.meta_layers,
            "dataset": "CIFAR-10" if args.num_classes == 10 else "CIFAR-100",
        }
        trial_tracking["parameters"] = params
        trial_tracking["parameters/ncl_weight"] = float(args.ncl_weight)
        trial_tracking["ncl_weight"] = float(args.ncl_weight)
        trial_tracking["runID"] = run_id
        trial_tracking["models.py"].track_files("models.py")

    models = (model, metaModel)
    optimizers = (optimizer, metaOptimizer)
    dataLoaders = (trainloader, testloader)
    train_model_NCL(run_id, trial_tracking, dataLoaders, models, optimizers, scheduler, num_epochs=args.num_epochs)

    if trial_tracking is not None:
        trial_tracking.wait()
        trial_tracking.stop()


if __name__ == '__main__':
    for _ in range(args.num_trials):
        main()
        torch.cuda.empty_cache()