import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

logger=logging.getLogger(__name__)

def Trainer(object):

    def __init__(self,model,optimizer):
        self.model=model
        self.optimizer=optimizer

    def get_model_output(self,X):
        return self.model.forward(X)
