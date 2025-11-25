from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import SceneDataset
from classifier import SceneClassifier

def model_train(
    path_train_csv: Path,
    path_images: Path,
    labels: dict[int, str],
    transforms,
    epochs: int = 20,
    batch_size: int = 256
) -> SceneClassifier:
    
    # Создаём экземпляры датасета и модели
    dataset = SceneDataset(
        path_train_csv,
        path_images,
        labels,
        transforms
    )

    model = SceneClassifier(epochs, len(labels))
    model.train()

    # Формируем из датасета батчи
    batched_dataset = DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(epochs):
        print(f'Эпоха {epoch + 1} / {epochs}...')
        epoch_acc = []

        batched_dataset_with_progress_bar = tqdm(
            batched_dataset, unit=f'батч по {batch_size}'
        )
        
        # Проходимся по батчам --
        # вся логика обучения и классификации реализована внутри модели
        for images, labels in batched_dataset_with_progress_bar:
            output = model({'images': images, 'labels': labels})
            epoch_acc.append(output)

            batched_dataset_with_progress_bar.set_postfix(
                Точность=f'{int(output * 100) / 100}'
            )
    
    return model
