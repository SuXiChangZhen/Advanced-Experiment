import torch
import numpy as np 
import torch.nn as nn 
import math
from models import *
import torch.nn.functional as F
from torchvision import models as torch_models
from torch.nn import DataParallel
from collections import OrderedDict
import torch.backends.cudnn as cudnn

class ModelPT:
    """
    Wrapper class around PyTorch models.

    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, dataset_name, batch_size=400):
        self.model_name = model_name
        self.batch_size = batch_size
        if dataset_name == 'imagenet':
            self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
            self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        elif dataset_name == 'cifar-10':
            self.mean = np.reshape([0.4914, 0.4822, 0.4465], [1, 3, 1, 1])
            self.std = np.reshape([0.2023, 0.1994, 0.2010], [1, 3, 1, 1])
        else:
            self.mean = np.reshape([0.491, 0.482, 0.446], [1, 3, 1, 1])
            self.std = np.reshape([0.202, 0.199, 0.201], [1, 3, 1, 1])
             
        if model_name in ['pt_vgg', 'pt_resnet', 'pt_inception', 'pt_densenet', 'pt_alexnet']:
            model = model_class_dict[model_name](pretrained=True)
            model = DataParallel(model.cuda())
        else:
            if model_name == 'vgg':
                model = model_class_dict[model_name]('VGG16')
            else:
                model = model_class_dict[model_name]()
            
            if dataset_name == 'cifar-10':            
                checkpoint = torch.load('../cifar-10/trainedModel/' + model_name + 'model.pth')
                checkpoint = checkpoint['net']
                new_checkpoint = OrderedDict()
                for k,v in checkpoint.items():
                    name = k[7:]
                    new_checkpoint[name] = v 
                model.load_state_dict(new_checkpoint)
            else:
                checkpoint = torch.load('../kaggle-cifar10/trainedModel/' + model_name + '.pth.tar')
                model.load_state_dict(checkpoint['state_dict'])
            model = model.to('cuda')
            model = DataParallel(model)
                
            # model.float()
        cudnn.benchmark = True
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)
        model.eval()
        self.model = model
        

    def predict(self, x, batch=None):
        if batch is not None:
            self.batch_size = batch
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

    def predict_result(self, x, batch=None):
        if batch is not None:
            self.batch_size = batch
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        predict_vals_list = []
        predict_labels_list = []
        with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                output = self.model(x_batch_torch)
                if self.model_name == 'pt_alexnet':
                    output = F.softmax(output, dim=1)
                predict_vals, predict_labels = output.max(1)
                predict_vals = predict_vals.cpu().numpy()
                predict_labels = predict_labels.cpu().numpy()
                predict_vals_list.append(predict_vals)
                predict_labels_list.append(predict_labels)
        predict_vals = np.hstack(predict_vals_list)
        predict_labels = np.hstack(predict_labels_list)
        return predict_vals, predict_labels

          

model_class_dict = {'pt_vgg': torch_models.vgg16_bn,
                    'pt_resnet': torch_models.resnet50,
                    'pt_inception': torch_models.inception_v3,
                    'pt_densenet': torch_models.densenet121, 
                    'pt_alexnet': torch_models.alexnet, 
                    'nin': NiN, 
                    'vgg': VGG, 
                    'allconv': AllConv}
# model = ModelPT('pt_vgg',12)
# data = torch.randn(12,3,224,224)
# data = data.cuda()

# output = model.model(data)
# print (output)


# model_path_dict = {}
