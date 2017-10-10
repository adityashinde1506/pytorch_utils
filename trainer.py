import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

import logging
logger=logging.getLogger(__name__)

class Trainer(object):

    def __init__(self,model,optimizer):
        self.model=model
        self.optimizer=optimizer

    def get_model_output(self,X):
        return self.model.forward(X)

    def train(self,batch_generator,loss_function,print_every=10,epochs=20):
        for i in range(epochs):
            batch_gen=batch_generator()
            model.zero_grad()
            total_loss=0
            b=0
            while 1:
                try:
                    _X,_y=next(batch_gen)
                except:
                    break
                train=A.Variable(torch.from_numpy(_X))
                targets=A.Variable(torch.from_numpy(_y).squeeze())
                output=self.get_model_output(train)
                loss=loss_function(output,targets)
                loss.backward()
                optimizer.step()
                total_loss+=loss.data[0]
                if b % print_every ==0:
                    logger.debug("Epoch :{} Batch:{} Loss:{}".format(i,b,total_loss))
                    total_loss=0
                b+=1

class RecurrentTrainer(Trainer):
    
    def __init__(self,model,optimizer):
        super(RecurrentTrainer,self).__init__(model,optimizer)

    def get_model_output(self,_input,_hidden):
        return self.model.forward(_input,_hidden)

    def train_on_batch(self,_input,_hidden,_labels,loss_fn):
        output,_hidden=self.get_model_output(_input,_hidden)
        loss=loss_fn(output,_labels.contiguous().view(-1))
        loss.backward()
        return loss.data[0],_hidden

    def train(self,data,batch_generator,loss_function,print_every=10,epochs=20):
        for i in range(epochs):
            batch_gen=batch_generator(data)
            self.model.zero_grad()
            hidden=self.model.init_hidden(data.shape[0])
            total_loss=0
            b=0
            while 1:
                try:
                    _X,_y=next(batch_gen)
                except:
                    break
                train=A.Variable(torch.from_numpy(_X))
                targets=A.Variable(torch.from_numpy(_y).squeeze())
                batch_loss,hidden=self.train_on_batch(train,A.Variable(hidden.data),targets,loss_function)
                self.optimizer.step()
                total_loss+=batch_loss
                if b % print_every ==0:
                    logger.debug("Epoch :{} Batch:{} Loss:{}".format(i,b,total_loss))
                    total_loss=0
                b+=1




