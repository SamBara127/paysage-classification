from pathlib import Path
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import SceneDataset
from splitter import Sample


def model_test(
    model,
    samples_test: list[Sample],
    path_images: Path,
    transforms,
    batch_size: int = 256
) -> List[Tensor]:
               
    # Создаём экземпляры датасета и модели
    dataset = SceneDataset(
        samples_test,
        path_images,
        {}, # В тестовом режиме названия меток не нужны
        transforms,
    )

    model.test()

    # Формируем из датасета батчи
    batched_dataset = DataLoader(dataset, batch_size)

    predictions = []

    batched_dataset_with_progress_bar = tqdm(
        batched_dataset, unit=f'батч по {batch_size}'
    )
        
    # Проходимся по батчам
    for img in batched_dataset_with_progress_bar:
        img = img.to(model.device)

        predictions.append(model(img))

    return predictions
