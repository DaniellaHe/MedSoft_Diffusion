import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from random import seed
import json
sys.path.append(os.path.abspath('./data/'))
import dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
torch.cuda.empty_cache()

ROOT_DIR = "./"

import csv
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from collections import defaultdict


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def get_activation(activations, name):

    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def train(training_dataset, testing_dataset, args, resume):
    # resnet18_weights = torch.load('./model_weights/resnet18-5c106cde.pth')
    # resnet18_weights['conv1.weight'] = resnet18_weights['conv1.weight'].sum(1, keepdim=True)
    # torch.save(resnet18_weights, './model_weights/resnet18_new.pth')

    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
    start_epoch = 0
    tqdm_epoch = range(start_epoch, args['EPOCHS']+1)

    model = models.resnet18()
    model.load_state_dict(torch.load('./model_weights/resnet18-5c106cde.pth'))
    pretrained_weights = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
    model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2).to(device)
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    losses = []
    iters = len(training_dataset) // args['Batch_Size']

    activations = {}
    features_list = []
    labels_list = []
    model.layer4[1].conv2.register_forward_hook(get_activation(activations, 'last_conv'))

    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        print('epoch:', epoch)
        for i, data in enumerate(training_dataset_loader):
            if i % 5 == 0:
                print(i, "/", iters)
            x = data["image"].to(device)  # torch.Size([1, 3, 256, 256])
            label = data["label"].to(device)
            input_x = x

            optimizer.zero_grad()
            outputs = model(input_x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            mean_loss.append(loss.data.cpu())

            if epoch % 5 == 0 and i == 0:
                feature = activations['last_conv'].cpu()
                features_list.append(feature)
                labels_list.append(label.cpu())

        losses.append(np.mean(mean_loss))
        print("loss:", losses[-1])

        if epoch % args['test_epoch'] == 0:
            print(str(epoch) + " epoch test:")
            model.eval()
            single_evaluation(model, testing_dataset, args, epoch, losses[-1])
            model.train()

            all_features = torch.cat(features_list).numpy()
            all_labels = torch.cat(labels_list).numpy()
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(all_features.reshape(all_features.shape[0], -1))
            plt.figure(figsize=(10, 5))
            plt.scatter(features_2d[all_labels == 0, 0], features_2d[all_labels == 0, 1], color='red', label='Class 0')
            plt.scatter(features_2d[all_labels == 1, 0], features_2d[all_labels == 1, 1], color='blue', label='Class 1')
            plt.legend()
            plt.savefig(f'./training_outputs/t-SNE_resnet18/epoch={epoch}_tsne_visualization.png')
            # plt.show()

        if epoch % args['save_epoch'] == 0:
            # scheduler.step()
            save(model=model, args=args, epoch=epoch)


def single_evaluation(model, testing_dataset, args, epoch, loss):
    predicted_prob = []
    label = []

    args['Batch_Size'] = 1
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    with torch.no_grad():
        for i, data in enumerate(testing_dataset_loader):
            x = data["image"].to(device)
            l = data["label"].to(device)
            input_x = x
            output = model(input_x)
            probabilities = torch.sigmoid(output)[:, 1]
            output_list = probabilities.tolist()
            label_list = l.tolist()

            predicted_prob.extend(output_list)
            label.extend(label_list)

        predicted_prob = torch.tensor(predicted_prob)
        label = torch.tensor(label)
        predicted_label = (predicted_prob > 0.5).float()
        auc = roc_auc_score(label.cpu().numpy(), predicted_label.cpu().numpy())
        accuracy = accuracy_score(label.cpu().numpy(), predicted_label.cpu().numpy())
        precision = precision_score(label.cpu().numpy(), predicted_label.cpu().numpy(), zero_division=1)
        recall = recall_score(label.cpu().numpy(), predicted_label.cpu().numpy())
        f1 = f1_score(label.cpu().numpy(), predicted_label.cpu().numpy())

        print("AUC:", auc)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1:", f1)

        with open(f'ARGS={args["arg_num"]}_results.csv', 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'loss', 'auc', 'accuracy', 'precision', 'recall', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch,
                             'loss': loss,
                             'auc': auc,
                             'accuracy': accuracy,
                             'precision': precision,
                             'recall': recall,
                             'f1': f1})

def save(model, args, epoch=0):
    model_save_path = f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}/checkpoint/classifier_epoch={epoch}.pt'
    torch.save(model.state_dict(), model_save_path)


def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """
    # make directories
    for i in ['./model/', "./training_outputs/t-SNE_resnet18"]:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # resume from final or resume from most recent checkpoint -> ran from specific slurm script?
    resume = 0
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args       file = 'args28.json'
    with open(f'{ROOT_DIR}configs/classifier/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    with open(f'ARGS={args["arg_num"]}_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'auc', 'accuracy', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print(file, args)
    # make arg specific directories
    for i in [f'./model/params-ARGS={args["arg_num"]}',
              f'./model/params-ARGS={args["arg_num"]}/checkpoint/',
              ]:
        try:
            os.makedirs(i)
        except OSError:
            pass

    if args["channels"] != "":
        in_channels = args["channels"]

    # load my vertebrae dataset
    training_dataset, testing_dataset = dataset.datasets(ROOT_DIR, args)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    if resume:
        if resume == 1:
            checkpoints = os.listdir(f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}/checkpoint')
            checkpoints.sort(reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"{ROOT_DIR}model/params-ARGS={args['arg_num']}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue

        else:
            file_dir = f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    # load, pass args
    train(training_dataset, testing_dataset, args, loaded_model)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    main()
