from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(
            self,
            path_dataset: Path,
            path_images: Path,
            labels: dict[int, str],
            transforms: Callable,
            is_test: bool = False
        ):

        self.transforms = transforms
        self.is_test = is_test

        if self.is_test:
            self.labels = labels

        # Предобрабатываем изображения
        self.preprocess_images(path_dataset, path_images)

    def load_an_image(self, row, path_images):
        # Соединяем переданную папку с названием файла в датафрейме
        file_path = f"{path_images}/{row['image_name']}"

        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')

            # Масштабируем изображение для лучшей сходимости
            img_array = np.array(img_rgb) / 255.0

            # Переводим изображение в тензор
            img_tensor = self.transforms(img_array).float()

            return img_tensor if self.is_test else (img_tensor, int(row['label']))
    
    def preprocess_images(self, path_dataset, path_images):
        # Открываем датафрейм с изображениями (создаём тренировочную выборку)
        path = path_dataset
        df = pd.read_csv(path)

        # Создаём список изображений;
        # Нам нужен список кортежей. Кортеж - пара тензор-метка
        img_list = df.apply(
            lambda row: self.load_an_image(row, path_images),
            axis=1
        ).tolist()

        self.__dataset = img_list

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)
