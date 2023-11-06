import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Callable
from tqdm.notebook import tqdm
from preprocessing.preprocessor import ImagePreprocessor



def jaccard_loss(predicted, target, smooth=1e-5):
    """
    Вычисляет Jaccard loss (IoU) между предсказанными и целевыми изображениями.

    :param predicted: Предсказанные изображения в формате NumPy.
    :param target: Целевые изображения в формате NumPy.
    :param smooth: Сглаживающий параметр для избежания деления на ноль.

    :return: Значение Jaccard loss.
    """
    predicted = torch.tensor(predicted)
    target = torch.tensor(target).detach()
    
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    jaccard_index = (intersection + smooth) / (union + smooth)
    loss = 1.0 - jaccard_index
    
    return loss



def dice_loss(predicted, target, smooth=1e-5):
    """
    Вычисляет Dice loss между предсказанными и целевыми изображениями.

    :param predicted: Предсказанные изображения в формате NumPy.
    :param target: Целевые изображения в формате NumPy.
    :param smooth: Сглаживающий параметр для избежания деления на ноль.

    :return: Значение Dice loss.
    """
    predicted = torch.tensor(predicted)
    target = torch.tensor(target).detach()
    
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice_coefficient
    
    return loss



def analyzer (src : List[torch.Tensor], tgt: np.ndarray, estimator: Callable) -> None:
    """
    Анализирует результаты оценки моделей с разными функциями потерь и долями данных для обучения.

    :param src: Список тензоров с изображениями.
    :param tgt: Целевые изображения в формате NumPy.
    :param estimator: Функция оценки, например, jaccard_loss или dice_loss.

    :return: Ничего не возвращает, но отображает график ошибок.
    """
    # Инициализируем препроцессор
    P = ImagePreprocessor()

    # Инициализируем сетку 
    x = [0.1, 0.25, 0.5]
    Y = np.zeros((2, 3))

    # Заводим тензор под хранение прогнозов
    predicts = torch.zeros((len(src), 1, 164, 164))

    for k, loss in enumerate([torch.nn.BCELoss(), torch.nn.MSELoss()]):
        for l, p in enumerate(x):
            # Выбираем нужную модель
            model = torch.load(f"saved_models/test_{1-p}/{loss.__str__()}/small_UNet.Adam.0.0001")

            # Получаем все предсказания
            for i in tqdm(range(len(src))):
                predicts[i:i+1] = model(src[i:i+1]).detach()
            
            # Формируем список np.ndarray чанков и собираем их в единое изображение
            p_chunks = [P.to_img(predicts[i]) for i in range(len(predicts))]
            rec_pred = P.unite(p_chunks, chunk_size=164, mask=True)
            
            # Записываем в матрицу результатов итоги оценки
            Y[k, l] = estimator(rec_pred, rec_pred)

    # Отрисовываем полученный график
    plt.figure(figsize=(20, 10))
    plt.title(f'График ошибки в выбранной метрике')
    plt.xlabel('Доля данных, использованных для обучения')
    for k, loss in enumerate([torch.nn.BCELoss(), torch.nn.MSELoss()]):
        plt.plot(x, Y[k], label=f"{loss.__str__()}")

    plt.legend()
    plt.show() 

    return