import argparse
import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train(args.dir, args.proc, args.arch, args.hunit, args.epochs, args.lr)

def train(data_dir, processing_unit, arch, h_unit, epochs, lr):
    if processing_unit == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif processing_unit == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    #dataset:
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    test_transforms =  transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    valid_transforms =  transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data =  datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    #Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # TODO: Build and train your network
    #designate model
    if arch == "50":
        model = models.resnet50(pretrained=True)
    elif arch == "152":
        model = models.resnet152(pretrained=True)
    else:
        print("architecture unknown, defaulting to resnet50.")
        model = models.resnet50(pretrained=True)
       
    #create classifier extension:
    classifier_input_size = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(classifier_input_size, 2*h_unit)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(2*h_unit, h_unit)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.3)),
                                            ('fc3', nn.Linear(h_unit, h_unit)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.4)),
                                            ('fc3', nn.Linear(h_unit, h_unit)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.3)),
                                            ('fc3', nn.Linear(h_unit, 102)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.2)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    model.fc = classifier
    #freezing all but the last layer:
    ct = 0
    for child in model.children():
        ct += 1
    
    i = 0
    for child in model.children():  
        i += 1
        if i < ct:
            for param in child.parameters():
                param.requires_grad = False

    #attach classifier to model  
    model.classifier = classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    #define criterion
    criterion = nn.NLLLoss()
    #define loss fxn
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    #designate model for GPU mode
    if device.type == "cuda":
        model.to(device)

    #design training loop:
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        print("Training epoch {}/{}... ".format(epoch+1, epochs))
        running_loss = 0
        for inputs, labels in trainloader:
            #go to training mode:
            model.train()
            steps += 1
            # Move data and model into GPU:
            if device.type == "cuda":
                inputs, labels = inputs.to(device), labels.to(device)
            inputs = Variable(inputs, requires_grad = True)

            #train pass:
            optimizer.zero_grad()
            logps = model(inputs)
            inputs.retain_grad()
            loss = criterion(logps, labels)
            loss.backward()
            inputs.retain_grad()
            optimizer.step()
            running_loss += loss.item()

        valid_loss = 0
        accuracy = 0
        model.eval()
        for inputs, labels in validloader:
            if device.type == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Train loss: {running_loss/len(trainloader):.3f}... ",
              f"Validation loss: {valid_loss/len(validloader):.3f}... ",
              f"Validation accuracy: {accuracy/len(validloader):.3f}...")

        running_loss = 0
        model.train()
    
    #saving model:
    model.class_to_idx = test_data.class_to_idx
    model.cpu()
    torch.save({'model': model,
                'h_unit': h_unit,
                'classifier_input':classifier_input_size,
                'learning_rate':lr,
                'state_dict': model.state_dict(), 
                'indexer': model.class_to_idx,
                'optimizer': optimizer.state_dict(),
                'criterion': criterion}, 
                'P2checkpoint.pth')
   
    
def main():
    parser = argparse.ArgumentParser(description='Define dataset, processing unit, resnet model, hidden units, epochs, and learning rate')
    parser.add_argument('dir',type = str, help = 'Training dataset pathway')    
    parser.add_argument('--proc', type = str, default = 'cpu', help = 'assign model training in CPU or GPU')
    parser.add_argument('--arch', type = str, default ='50', help = 'select resnet model ("50" or "152")')
    parser.add_argument('--hunit', type = int, default = 512, help = 'number of hidden units in model. default is 512.')
    parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs for training, default is 10.')
    parser.add_argument('--lr', type = float, default= 0.0011, help = 'model optimizer learning rate. default is 0.0011.')
    args = parser.parse_args()
    
    #data_dir, processing_unit, arch, h_unit, epochs, lr
    train(data_dir = args.dir, processing_unit = args.proc, arch = args.arch, h_unit = args.hunit, epochs = args.epochs, lr = args.lr)
    
if __name__ == "__main__":
    main()




