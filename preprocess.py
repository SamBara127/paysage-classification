from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from splitter import Sample

class SceneDataset(Dataset):
    def __init__(
            self,
            image_samples: list[Sample],
            images_path: Path,
            labels: dict[int, str],
            transforms: Callable,
        ):

        self.transforms = transforms
        self.images_path = images_path
        self.labels = labels

        # Предобрабатываем изображения
        self.preprocess_images(image_samples)

    def load_an_image(self, sample):
        # Соединяем переданную папку с названием файла в датафрейме
        file_path = f"{self.images_path}/{sample['image_name']}"

        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')

            # Масштабируем изображение для лучшей сходимости
            img_array = np.array(img_rgb) / 255.0

            # Переводим изображение в тензор
            img_tensor = self.transforms(img_array).float()

            return (img_tensor, int(sample['label']))
    
    def preprocess_images(self, image_samples):
        # Создаём список изображений;
        # Нам нужен список кортежей. Кортеж -- пара тензор-метка
        img_list = list(map(
            lambda sample: self.load_an_image(sample),
        image_samples))

        self.__dataset = img_list

    def __getitem__(self, index):
        return self.__dataset[index]
    
    def __len__(self):
        return len(self.__dataset)
