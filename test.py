from pathlib import Path
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import SceneDataset


def model_test(
    model,
    path_test_csv: Path,
    path_images: Path,
    transforms,
    batch_size: int = 256
) -> List[Tensor]:
               
    # Создаём экземпляры датасета и модели
    dataset = SceneDataset(
        path_test_csv,
        path_images,
        {}, # В тестовом режиме названия меток не нужны
        transforms,
        is_test=True
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
