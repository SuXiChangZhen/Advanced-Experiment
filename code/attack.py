# -*- coding: utf-8 -*-
# The demo of attack on imagenet

import argparse
import numpy as np 
import helper
from DE import differential_evolution
import model
import data
from Logger import *


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def init_perturbation(pixel=1, popsize=400, size=32):
    # pixel: the number of pixel
    bounds = [(0, size-1), (0, size-1), (0, 1), (0, 1), (0, 1)]
    bounds = bounds * pixel
    
    coordinate = np.random.randint(bounds[0][0], bounds[0][1], (pixel*popsize, 2))
    color = np.clip(np.random.normal(loc=128, scale=127, size=(pixel*popsize, 3)), 0, 255)
    color = color / 255
    parameters = np.hstack((coordinate, color))
    parameters = parameters.reshape((popsize, pixel*5))
    return bounds, parameters

class PixelAttacker():
    def __init__(self, model, data, class_name, size=32):
        self.model = model
        self.data = data
        self.class_name = class_name
        self.size = size
    
    def model_predict(self, parameters, img, label, model):
        print('label', label)
        batch_size = parameters.shape[0]
        image = helper.perturbation(parameters, img)
        output = model.predict(image, batch_size)
        # output = softmax(output)
        result = output[:, label]
        return result

    # def model_predict_cifar10(self, parameters, img, label, model):
    #     image = helper.perturbation(parameters, img)
    #     net = infer.makeModel('VGG16')
    #     output = infer.infer(image, net)
    #     output = softmax(output)
    #     result = output[:, label]
    #     return result
    
    def attack_one(self, img_id, model, model_name, target=None, pixel_count=1, maxiter=100, popsize=400):
        if target is None:
            logger = Logger('./Log')
        else:
            logger = Logger('./Log')
        origin_img = self.data[img_id]
        target_flag = target is not None
        target_class = target if target_flag else self.class_name[img_id]
        bounds, init = init_perturbation(pixel_count, popsize, self.size)
        def fitness_func_untarget(parameters):            
            return self.model_predict(parameters, origin_img, target_class, model)
        def fitness_func_target(parameters):
            return -self.model_predict(parameters, origin_img, target_class, model)

        fitness_func = fitness_func_target if target_flag else fitness_func_untarget

        result, fitness_iteration = differential_evolution(func=fitness_func, bounds=bounds, init=init, popsize=popsize, maxiter=maxiter, verbose=True)
        logger.output_log(str(img_id) + '.log', fitness_iteration)
        pert = np.array([result])
        attack_img = helper.perturbation(pert, origin_img)
        # output = model.predict(attack_img, 1)
        output_vals, output_labels = self.model.predict_result(attack_img)
        return result, output_vals, output_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 多值参数返回的是一个参数的list
    parser.add_argument('--model', nargs='+', type=str, choices=model.model_class_dict.keys(), default='pt_alexnet',
                        help='models to evaluate.')
    parser.add_argument('--pixel', nargs='+', type=int, default=1, help='the number of pixel to perturbate')
    parser.add_argument('--maxiter', nargs=1, type=int, default=100, help='generation of DE')
    parser.add_argument('--popsize', nargs=1, type=int, default=400, help='size of population of DE')
    parser.add_argument('--dataset', nargs=1, type=str, default='imagenet', help='dataset to evaluate')
    parser.add_argument('--samples', nargs=1, type=int, default=100, help='sample the dataset randomly')
    parser.add_argument('--targeted', nargs=1, type=bool, default=False, help='target or non-target attack')

    
    