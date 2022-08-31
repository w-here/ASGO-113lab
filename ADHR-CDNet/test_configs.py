#-*-coding:utf-8-*-
import numpy as np

class Config:
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 5e-4

    num_epochs =40
    batch_size =1
    use_gpu = True

    num_workers = 0
    show_every =50
    save_every = 5000
    test_every = 100
    image_every = 100
    tensorboard_path='./tensorboard'
    save_path = './models/'

