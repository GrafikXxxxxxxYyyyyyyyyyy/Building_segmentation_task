# Buildings segmentation task (Skoltech)
Данный репозиторий представляет собой задачу обучения нейронной сети для сегментирования зданий на спутниковых изображениях


### Dataset description:
- Папка с набором данных "data/" включает в себя два исследовательских объекта (Ventura и Santa-Rosa).
- RGB-каналы находятся в отдельных файлах в формате .tif
- Разметка (метки истинной информации) находится в файле all.tif


### Task:
Это задача бинарной семантической сегментации. Требуется обучить сверточную нейронную сеть (CNN) прогнозировать маску сегментации с использованием предоставленных спутниковых изображений. Окончательные результаты прогнозирования должны быть представлены для тестовой области (вы должны выбрать тестовую область из набора данных) в соответствии с соответствующими оценочными метриками.


### Model:
В качестве модели было решено реализовать модель UNet: https://arxiv.org/pdf/1505.04597.pdf 

UNet используется в данной задаче по следующим причинам:
1. Изначально модель UNet разрабатывалась как раз для задачи сгементации изображений
2. Лёгкость реализации модели
3. Небольшое количество требуемых данных для обучения модели (а это нам важно, поскольку выборка в данном задании бедная)
4. Высокая скорость обучения модели
5. *Мой личный исследовательский интерес (U-Net это backbone архитектура для предсказания шума в DDPM)

Возможно в будущем в данный проект будет внедрена и VGG: https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/


### Training:
Процесс обучения проиллюстрирован в файле TrainingLoop.ipynb
С процессом обучения для разных моделей можно ознакомиться запустив команду "tensorboard --logdir logs/" внизу ноутбука

Если описывать его кратко, то построен он следующим образом:
- Определяются размеры картинок для входа модели и для удобства пользования инициализируется ImagePreprocessor() - класс для обработки изображений в рамках данного проекта
- Из полученных изображений строится выборка данных, путём нарезания изображений на чанки
- Далее полученная выборка очищается от бесполезных данных (удаляются все чанки, где крайне мало или нет зданий) и рандомным образом перемешивается
- Далее происходит разбиение выборки на train/test датасеты, инициализация модели и гиперпараметров, а затем обучение


Подробнее об обучении:
- Для обучения использовалась одна и та же архитектура уменьшенной модели U-Net
- Обучение шло на сетке из двух гиперпараметров:
    >1. p=[0.9, 0.75, 0.5] - доля от общей выборки, взятой в качестве test семплов 
    >2. loss=[BCELoss, MSELoss] - функции потерь, с которыми обучались архитектуры
    >3. В результате получаем 6 обученных архитектур, которые можем сравнивать между собой в рамках исследования подходов к решению данной задачи
- После обучения модели, результат работы обученной сети сравнивается с истинной разметкой


### Results:
С результатами обучения модели можно ознакомиться в файле Results.ipynb

Если описывать полученные результаты кратко, то можно сказать следующее:
- В качестве оценки качества модели было решено использовать следующий подход:
    > 1. Выбранное изображение разбивается на чанки (в том числе и пустые чанки без зданий)
    > 2. Модель строит предсказание для всех полученных чанков, составляющих изображение 
    > 3. Спрогнозированная сегментация по чанкам собирается обратно в единое изображение

- Происходит качественная оценка работы выбранной модели, путём визуального сопоставления сегментационных масок, истинной и построенной обученной моделью
- Происходит количественная оценка и сопоставление работоспособности всех обученных моделей на выбранном изображении, используя такие критерии качества, как IoU или Dice loss