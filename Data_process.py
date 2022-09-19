import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import entropy

device = "cpu"
def margin(arr):
  arr.sort(reverse=True)
  diff = arr[0]-arr[1]
  return 1-diff

def leastConf(arr):
  pred = max(arr)
  norm = len(arr) // (len(arr)-1)

  return (1-pred)*norm

def classify(model,dataloaders ,domain, num_images=64):
    was_training = model.training
    model.eval()
    images_so_far = 0


    actual = []
    arrays = []
    preds = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[domain]):
            inputs = inputs.to(device)
            label = labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            actual.append(int(label[0]))
            preds.append(int(pred[0]))
            arrays.append(outputs.tolist()[0])

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # print(labels[j])
                # ax.set_title(f'predicted: {class_names[pred[j]]}')
                # imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return actual,preds,arrays
        model.train(mode=was_training)


def process():
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'alt_1km': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'alt_10km': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'alt_50km': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    folder = ['test','alt_1km','alt_10km','alt_50km']

    data_dir = 'Data_repo/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in folder}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                shuffle=True, num_workers=4)
                for x in folder}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in folder}
    class_names = image_datasets[folder[1]].classes

    model = torch.load("mobilenetV3_64_sigmoidTest.pth",map_location=torch.device(device))

    f1s = []
    avgleastConfs = []
    avgmargins = []
    avgentrops = []

    Dict = {}

    for fold in folder:
        print(fold)
        actual,preds,arrays = classify(model,dataloaders,fold)
        print(type(actual),type(preds),type(arrays))

        f = f1_score(actual, preds, average='macro')
        leastConfs = [leastConf(x) for x in arrays]
        margins = [margin(x) for x in arrays]
        entrops = [entropy(x, base=4) for x in arrays]
        Dict[fold] = {
            "LeastConf":leastConfs,
            "Margin":margins,
            "Entropy":entrops,
            }
        f1s.append(f)
        avgleastConfs.append(np.mean(leastConfs))
        avgmargins.append(np.mean(margins))
        avgentrops.append(np.mean(entrops))

    avgDict ={
    "avgf1Score":f1s,
     "avgLeastConf":avgleastConfs,
     "avgMargin":avgmargins,
     "avgEntropy":avgentrops,
    }


    return avgDict,Dict
