import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


#########################################################################################################
def batch_generator (X, Y, batch_size=10):
    """
    Генератор батчей для обучения.

    Args:
        - X (numpy.ndarray): Массив признаков.
        - Y (numpy.ndarray): Массив целевых значений.
        - batch_size (int, optional): Размер батча. По умолчанию 10.

    Yields:
        - tuple: Батч признаков и соответствующих целевых значений.
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
    """
    Класс для отслеживания метрик обучения и вывода их в TensorBoard.

    Args:
        - X (numpy.ndarray): Массив признаков для валидации.
        - Y (numpy.ndarray): Массив меток для валидации.
        - writer (SummaryWriter): Объект для записи метрик в TensorBoard.
        - loss_function (torch.nn.Module): Функция потерь для оценки качества модели.
        - delimeter (int): Частота записи метрик (каждые `delimeter` итераций).
        - batch_size (int): Размер мини-пакета для валидации.

    Methods:
        - forward(model, loss): Вызывается после каждой итерации обучения для записи метрик.
    """
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



#########################################################################################################
def train_on_batch(model, x_batch, y_batch, optimizer, loss_function):
    """
    Функция для обучения модели на одном мини-пакете данных.

    Args:
        - model (torch.nn.Module): Модель для обучения.
        - x_batch (torch.Tensor): Мини-пакет признаков.
        - y_batch (torch.Tensor): Мини-пакет меток.
        - optimizer (torch.optim.Optimizer): Оптимизатор для обновления параметров модели.
        - loss_function (torch.nn.Module): Функция потерь для обучения модели.

    Returns:
        - float: Значение функции потерь на данном мини-пакете данных.
    """
    model.train()
    model.zero_grad()
    
    output = model(x_batch)
    
    loss = loss_function(output, y_batch)
    loss.backward()

    optimizer.step()

    return loss.cpu().item()
#########################################################################################################



#########################################################################################################
def train_epoch(train_generator, 
                model, 
                loss_function, 
                optimizer, 
                callback = None):
    """
    Функция для обучения модели на одной эпохе.

    Args:
        - train_generator (generator): Генератор мини-пакетов данных.
        - model (torch.nn.Module): Модель для обучения.
        - loss_function (torch.nn.Module): Функция потерь для обучения модели.
        - optimizer (torch.optim.Optimizer): Оптимизатор для обновления параметров модели.
        - callback (Callback, optional): Обратный вызов для отслеживания метрик обучения.

    Returns:
        - float: Среднее значение функции потерь на этой эпохе.
    """
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
def trainer(X,
            Y,
            model, 
            loss_function,
            count_of_epoch, 
            batch_size, 
            optimizer,
            lr = 0.001,
            callback = None):
    """
    Функция для обучения модели.

    Args:
        - X (torch.Tensor): Тензор признаков для обучения.
        - Y (torch.Tensor): Тензор меток для обучения.
        - model (torch.nn.Module): Модель для обучения.
        - loss_function (torch.nn.Module): Функция потерь для обучения модели.
        - count_of_epoch (int): Количество эпох обучения.
        - batch_size (int): Размер мини-пакета.
        - optimizer (torch.optim.Optimizer): Оптимизатор для обновления параметров модели.
        - lr (float): Скорость обучения.
        - callback (Callback, optional): Обратный вызов для отслеживания метрик обучения.

    Returns:
        - None
    """

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