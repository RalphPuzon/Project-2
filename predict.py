import argparse
import matplotlib.pyplot as plt
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
import json

def processing_unit(processing_unit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def label_mapping(jfile):
    with open(jfile, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_model(savefile):
    save = torch.load(savefile, map_location=lambda storage, loc: storage)
    model = save['model']
    h_unit = save['h_unit']
    lr = save['learning_rate']
    model.class_to_idx = save['indexer']
    classifier_input_size = save['classifier_input']
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

    model.classifier = classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    #define criterion
    criterion = nn.NLLLoss()
    #define loss fxn
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.load_state_dict(save['state_dict'])
    model.eval()

    return model

def process_image(image_path):
    image = Image.open(image_path)
    # resize:
    if image.size[0] > image.size[1]:
        ratio = 256/image.size[1]
        image = image.resize((math.ceil(ratio*image.size[0]), 256))
    elif image.size[1] > image.size[0]:
        ratio = 256/image.size[0]
        image = image.resize((256,math.ceil(ratio*image.size[1])))
    else:
        image = image.resize(256,256)

    #crop:
    left = (image.size[0]-224)/2 
    lower = (image.size[1]-224)/2
    right = left + 224
    upper = lower + 224 
    image = image.crop((left, lower, right, upper))
    #color scaling:
    image = np.array(image)/255

    #normalization:
    mu = np.array([0.485, 0.456, 0.406])
    sigma = np.array([0.229, 0.224, 0.225])
    image = (image-mu)/sigma

    #transpose color channel to front:
    image = image.transpose((2, 0, 1))
    return image

def obj_predict(image, model, limit, device, cat_to_name):
    image = torch.from_numpy(image)
    if device.type == "cuda":
        image = image.type(torch.cuda.FloatTensor)
        image = image.unsqueeze(0)
        model.to(device)
        image.to(device)
        prediction = model(image)
        prediction = prediction.cpu()
    else:
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(0)
        prediction = model(image).cpu()
    ps = torch.exp(prediction)
    best_p, labels = ps.topk(limit, dim = 1)
    best_p = (best_p.detach().numpy().tolist()[0])
    labels = labels.detach().numpy().tolist()[0]

    calls = []
    for i in labels:
        for j in model.class_to_idx.items():
            if j[1] == i:
                calls.append(j[0])
            else:
                continue

    flowers = []
    for i in calls:
        flowers.append(cat_to_name[i])
    return best_p, calls, flowers, labels
    
def main():
    parser = argparse.ArgumentParser(description='Define processing unit, hidden units and class mapping file.')
    parser.add_argument('imagepath', type = str, help = 'directory of image to be predicted.')  
    parser.add_argument('--proc', type = str, default = 'cpu', help = 'assign model training in CPU or GPU.')
    parser.add_argument('--jfile', type = str, default ='cat_to_name.json', help = 'class mapping, default is "cat_to_name" JSON file.')
    parser.add_argument('--limit', type = int, default ='5', help = 'k for top k classes to be returned.')
    args = parser.parse_args()
    
    device = processing_unit(args.proc)
    cat_to_name = label_mapping(args.jfile)
    model = load_model('P2checkpoint.pth')
    model.eval()
    image = process_image(args.imagepath)
    figprobs, figlabels, f_calls, labels = obj_predict(image, model, args.limit, device, cat_to_name)
   
    
    print(f"Most probable class: {f_calls[0]}",
              f"Probability: {figprobs[0]}",
              f"Top {str(args.limit)} classes: {f_calls}"
              f"Top {str(args.limit)} probabilities: {figprobs}")

if __name__ == '__main__':
    main()
    
    

