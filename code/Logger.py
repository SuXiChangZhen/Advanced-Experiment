# -*- coding: utf-8 -*-
import os
import numpy as np

class Logger(object):
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
    def output_log(self, name, content):
        with open(os.path.join(self.path, name), 'w') as fh:
            fh.write(str(content))        

