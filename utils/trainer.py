import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


#########################################################################################################
def batch_generator (X, Y, batch_size=10):
    """
    Принимает на вход матрицы трейна и теста
    На выходе выдаёт перемешанные и разделённые по батчам идексы
    """
    if X.shape[0] == Y.shape[0]:
        idxs = np.arange(X.shape[0]) # выбираем все строки нашего набора данных
        np.random.shuffle(idxs) # рандомно встряхиваем индексы
        idxs_pad = idxs[(X.shape[0] % batch_size):] # убираем первые лишние индексы (выравнивание)
        masks = idxs_pad.reshape(X.shape[0] // batch_size, batch_size) # получаем готовые маски
        
        for i in range(masks.shape[0]):
            yield X[masks[i]], Y[masks[i]]
            
    else:
        print(f"ERROR!: X size is {X.shape[0]} != {Y.shape[0]} which is Y size")
#########################################################################################################



#########################################################################################################
class Callback():
    def __init__ (self, 
                  X,
                  Y, 
                  writer, 
                  loss_function,
                  delimeter=100, 
                  batch_size=10):
        
        self.X = X
        self.Y = Y
        self.step = 0
        self.writer = writer 
        self.loss_function = loss_function
        self.delimeter = delimeter
        self.batch_size = batch_size            

        return


    def forward(self, model, loss):
        self.step += 1
        self.writer.add_scalar('LOSS/train', loss, self.step)
        
        if self.step % self.delimeter == 0:
            test_generator = tqdm(
                                    batch_generator(self.X, self.Y, self.batch_size), 
                                    leave=False, 
                                    total=len(self.X)//self.batch_size,
                                    dynamic_ncols=True                                        
                                 )
            
            model.eval() 
            
            test_loss = 0
            for it, (x_test, y_test) in enumerate(test_generator):
                output = model(x_test) 
                test_loss += self.loss_function(output, y_test).cpu().item()*len(x_test)
            
            test_loss /= len(self.X)
            
            self.writer.add_scalar('LOSS/test', test_loss, self.step)

            
          
    def __call__(self, model, loss):
        return self.forward(model, loss)
#########################################################################################################



# #########################################################################################################
# class Trainer ():
#     def __init__(
#                     self, 
#                     model,
#                     callback=None,
#                     batch_generator=None
#                 ):
        
#         self.model = model
#         self.callback = callback
#         self.batch_generator = batch_generator

#         return
    
    

#     def train_model (self, X, Y, optimizer, loss_function, epoch=100, lr=1e-3, batch_size=10):
#         # загружаем параметры в оптимизатор и настраиваем его (как в Attention is all tou need)
#         optim = optimizer(self.model.parameters(), lr=lr)
        
#         iterations = tqdm(range(epoch), desc='epoch', leave=False, dynamic_ncols=True) 
#         iterations.set_postfix({'train epoch loss': np.nan})

#         for it in iterations:
#             # Создаём генератор по тренировочной выборке
#             train_generator = tqdm  (
#                                         batch_generator(X, Y, batch_size), 
#                                         leave=False, 
#                                         total=len(X)//batch_size,
#                                         dynamic_ncols=True
#                                     )
            
#             # Запускаем функцию обучения на эпохе
#             epoch_loss = self.train_epoch(generator=train_generator, optimizer=optim, loss_function=loss_function)
            
#             iterations.set_postfix({'train epoch loss': epoch_loss})

#         return
    


#     def train_epoch (self, generator, optimizer, loss_function):
#         epoch_loss = 0
#         total = 0

#         for it, (batch_of_x, batch_of_y) in enumerate(generator):
#             batch_loss = self.train_batch(batch_of_x, batch_of_y, optimizer, loss_function)
            
#             if self.callback is not None:
#                 with torch.no_grad():
#                     self.callback(self.model, batch_loss)
                
#             epoch_loss += batch_loss*len(batch_of_x)
#             total += len(batch_of_x)

#         return
    


#     def train_batch (self, input, target, optimizer, loss_function):
#         self.model.train()
#         self.model.zero_grad()
        
#         output = self.model(input)
        
#         loss = loss_function(output, target)
#         loss.backward()

#         optimizer.step()

#         return loss.cpu().item()
# #########################################################################################################



#########################################################################################################
def trainer(X,
            Y,
            model, 
            loss_function,
            count_of_epoch, 
            batch_size, 
            optimizer,
            lr = 0.001,
            callback = None):

    optim = optimizer(model.parameters(), lr=lr)
    
    iterations = tqdm(range(count_of_epoch), desc='epoch', leave=False, dynamic_ncols=True) 
    iterations.set_postfix({'train epoch loss': np.nan})

    for it in iterations:
        train_generator = tqdm  (
                                    batch_generator (X, Y, batch_size), 
                                    leave=False, 
                                    total=len(X)//batch_size,
                                    dynamic_ncols=True
                                )
        
        epoch_loss = train_epoch(train_generator=train_generator, 
                                 model=model, 
                                 loss_function=loss_function, 
                                 optimizer=optim, 
                                 callback=callback)
        
        iterations.set_postfix({'train epoch loss': epoch_loss})
#########################################################################################################



#########################################################################################################
def train_epoch(train_generator, 
                model, 
                loss_function, 
                optimizer, 
                callback = None):
    
    epoch_loss = 0
    total = 0
    for it, (batch_of_x, batch_of_y) in enumerate(train_generator):
        batch_loss = train_on_batch(model, batch_of_x, batch_of_y, optimizer, loss_function)
        
        if callback is not None:
            with torch.no_grad():
                callback(model, batch_loss)
            
        epoch_loss += batch_loss*len(batch_of_x)
        total += len(batch_of_x)
    
    return epoch_loss/total
#########################################################################################################



#########################################################################################################
def train_on_batch(model, x_batch, y_batch, optimizer, loss_function):
    model.train()
    model.zero_grad()
    
    output = model(x_batch)
    
    loss = loss_function(output, y_batch)
    loss.backward()

    optimizer.step()

    return loss.cpu().item()
#########################################################################################################