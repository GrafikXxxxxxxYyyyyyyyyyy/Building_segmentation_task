import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt



class ImagePreprocessor ():
    def __init__(self):
        return
    

    
    def read_image (self, dir):
        """
        Input: dir_name
        Output: np.ndarray=(size, size, 3), np.ndarray=(size, size, 1) - BGR uint8 images
        """
        # Читаем 3 канала изображения
        img_red = cv2.imread(f'data/{dir}/RED.tif')
        img_grn = cv2.imread(f'data/{dir}/GRN.tif')
        img_blue = cv2.imread(f'data/{dir}/BLUE.tif')

        # Читаем сегментационную маску
        mask = cv2.imread(f'data/{dir}/all.tif')

        # Формируем цветное BGR изображение
        img = np.zeros((img_red.shape[0], img_red.shape[1], 3))
        img[:, :, 0] = img_blue[:, :, 0]
        img[:, :, 1] = img_grn[:, :, 0]
        img[:, :, 2] = img_red[:, :, 0]

        return img.astype(dtype='uint8'), mask[:, :, 0:1].astype(dtype='uint8')
    


    def destruct_to_chunks (self, 
                            img, 
                            chunk_size=164, 
                            output_size=256, 
                            mask=False, 
                            as_tensor=True):
        """
        Разбивает большое изображение на сегменты нужного размера
        """
        # Определяем количество чанков
        N = max(img.shape[0] // chunk_size, img.shape[1] // chunk_size) + 1

        # Растягиваем изображение
        img = cv2.resize(img, (N*chunk_size, N*chunk_size))

        # Собираем нарезанные чанки
        chunks = []
        for n_x in range(N):
            for n_y in range(N):
                if mask is True:
                    # Сохраняем чанк без изменений
                    sub_img = img[n_x*chunk_size:(n_x+1)*chunk_size, n_y*chunk_size:(n_y+1)*chunk_size].copy()
                    chunks.append(sub_img[:, :, None])
                else:
                    # Падим чанк нулями до нужного размера [можно ввести self.mirroring(src, output_size)]
                    indent = (output_size - chunk_size) // 2
                    sub_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
                    sub_img[indent:-indent, indent:-indent, :] = img[n_x*chunk_size:(n_x+1)*chunk_size, n_y*chunk_size:(n_y+1)*chunk_size, :].copy()
                    chunks.append(sub_img)

        # Если требуется вернуть в качестве тензора
        if as_tensor is True:
            # Если передана маска, то надо вернуть бинарную сегментацию
            if mask is True:
                t_chunks = torch.zeros((N*N, 1, chunk_size, chunk_size), dtype=torch.long)
            else:
                t_chunks = torch.zeros((N*N, 3, output_size, output_size))

            # В цикле по ивсем нарезанным чанкам собираем тензор
            for k, chunk in enumerate(chunks):
                t_chunks[k:k+1] = self.to_tensor(chunk, mask)

            return t_chunks
        else:
            return chunks
        


    def unite (self, 
               chunks, 
               chunk_size=164, 
               mask=False):
        """
        Input: List [np.ndarray]
        """
        N = int(np.sqrt(len(chunks)))

        if mask is True:
            united_img = np.zeros((chunk_size*N, chunk_size*N, 1), dtype=np.uint8)
        else:
            united_img = np.zeros((chunk_size*N, chunk_size*N, 3), dtype=np.uint8)

        for i in range(N):
            for j in range(N):
                united_img[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size, :] = chunks[i*N + j]
                
        return united_img
    


    def to_tensor (self, img, mask=True):
        """
        Input: np.ndarray = (size, size, n_channels)
        Output: torch.tensor = [1, n_channels, size, size]
        """
        # Нормировка значений в зависимости от маски
        if mask is True:
            t_img = torch.LongTensor((img / 255.0) > 0)    
        else:
            t_img = torch.FloatTensor(img / 255.0)

        # [256, 256, 3] --> [3, 256, 256]
        t_img = torch.transpose(t_img, 0, 2)
        
        return t_img.unsqueeze(0) # [3, 256, 256] --> [1, 3, 256, 256]
    


    def to_img (self, tensor, target_size=164):
        """
        Input: torch.tensor = [n_channels, size, size]
        Output: np.ndarray = (size, size, n_channels)
        """
        img = torch.transpose(tensor*255, 0, 2).detach().numpy().astype(dtype=np.uint8)
        img = np.clip(img, 0, 255, dtype=np.uint8)

        if img.shape[0] > target_size:
            indent = (img.shape[0] - target_size) // 2
            return img[indent:-indent, indent:-indent, :]
        else:
            return img
    


    def clear_dataset (self, images, masks, p=0.1):
        """
        Очищает выборку данных от шумовых чанков
        """
        norm = masks.shape[2] * masks.shape[2]

        clear_imgs = torch.zeros_like(images)
        clear_masks = torch.zeros_like(masks)

        idx = 0
        for i in range(len(masks)):
            if masks[i].sum() >= p * norm:
                clear_imgs[idx] = images[i]
                clear_masks[idx] = masks[i]
                idx += 1
        
        clear_imgs = clear_imgs[:idx]
        clear_masks = clear_masks[:idx]

        return clear_imgs, clear_masks
    


    def show (self, img, tgt=None):
        """
        Input: img - np.ndarray, tgt - np.ndarray/None
        """
        if tgt is None:
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            if img.shape[0] != tgt.shape[0]:
                indent = (img.shape[0] - tgt.shape[0]) // 2
                red_mask = np.zeros_like(img)
                red_mask[indent:-indent, indent:-indent, 2:3] = tgt
            else:
                red_mask = np.zeros_like(img)
                red_mask[:, :, 2:3] = tgt

            plt.figure(figsize=(30, 10))

            plt.subplot(1, 3, 1) 
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original image')

            plt.subplot(1, 3, 2) 
            plt.imshow(cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB))
            plt.title('Segmentation mask')

            plt.subplot(1, 3, 3)
            segm = cv2.addWeighted(img, 1, red_mask, 0.9, 0)
            plt.imshow(cv2.cvtColor(segm, cv2.COLOR_BGR2RGB))
            plt.title('Mixture')
            plt.show()

        return
        