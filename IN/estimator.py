from __future__ import print_function

from datetime import datetime
from timeit import default_timer as timer

import numpy as np

import torch

def logger(s):
    """Simple logger function which prints date/time"""
    print(datetime.now(), s)

class Estimator():
    """Estimator class"""
    #initialization
    def __init__(self, model, loss_func, opt='Adam',
                 train_losses=None, valid_losses=None,
                 cuda=False):

        self.model = model
        if cuda:
            self.model.cuda()
            
        self.loss_func = loss_func
        if opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)

        self.train_losses = train_losses if train_losses is not None else []
        self.valid_losses = valid_losses if valid_losses is not None else []

        logger('Model: \n%s' % model)
        logger('Parameters: %i' %
               sum(param.numel() for param in model.parameters()))

    def training_step(self, inputs, targets):
        """Applies single optimization step on batch"""
        self.model.zero_grad()# free gradients
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def fit_gen(self, train_generator, n_batches=1, n_epochs=1,
                valid_generator=None, n_valid_batches=1, verbose=0):
        """Runs batch training for a number of specified epochs."""
        #train on batches
        # print('training_function')
        epoch_start = len(self.train_losses)
        # print('n_batches ',n_batches)
        epoch_end = epoch_start + n_epochs
        for i in range(epoch_start, epoch_end):
            logger('Epoch %i' % i)
            start_time = timer()
            sum_loss = 0
            # Train the model
            self.model.train()
            for j in range(n_batches):
                # print(next(train_generator))
                batch_input, batch_target = next(train_generator)
                # print('batch_input ',batch_input,' batch_target ',batch_target)
                batch_loss = (self.training_step(batch_input, batch_target).item())
                
                sum_loss += batch_loss
                if verbose > 0:
                    logger('  Batch %i loss %f' % (j, batch_loss))
            end_time = timer()
            avg_loss = sum_loss / n_batches
            self.train_losses.append(avg_loss)
            logger('  training loss %.3g time %gs' %
                   (avg_loss, (end_time - start_time)))

            # Evaluate the model on the validation set
            if (valid_generator is not None) and (n_valid_batches > 0):
                self.model.eval()
                valid_loss = 0
                for j in range(n_valid_batches):
                    valid_input, valid_target = next(valid_generator)
                    valid_loss += (self.loss_func(self.model(valid_input), valid_target)
                                   .cpu().item())
                valid_loss = valid_loss / n_valid_batches
                self.valid_losses.append(valid_loss)
                if(valid_loss<= min(self.valid_losses)):
                    torch.save(self.model, f'best_model_IN.pkl')
                    print(f'Epoch_stop: {i}')
          
                logger('  validate loss %.3g' % valid_loss)
       
#
    def predict(self, generator, model_name,n_batches, concat=True):
        model = torch.load(model_name)
        self.model.eval()
        outputs = []
        for j in range(n_batches):
            test_input, test_target = next(generator)
            outputs.append(self.model(test_input))
        if concat:
            outputs = torch.cat(outputs)
        return outputs