import argparse
import numpy as np
from dataset import SceneDataset
from scene_classifier import SceneClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description='Аргументы для пайплайна')
    
    transform_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([150,150]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    ])

    print('transforms created')

    # Создаём экземпляры датасета и модели
    dataset = SceneDataset("/kaggle/input/d/nitishabharathi/scene-classification/train-scene classification/train.csv",
                        '/kaggle/input/label/labels.txt',
                        '/kaggle/input/d/nitishabharathi/scene-classification/train-scene classification/train',
                        transform_list)
    model = SceneClassifier(10)

    print("Preprocessed, slava tebe gospodi")

    # Формируем из датасета батчи
    batched_dataset = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    for epoch in range(10):
        print(f'Эпоха {epoch} началась')
        epoch_acc = []
        
        # Проходимся по батчам - вся логика обучения и классификации реализована внутри модели
        for images, labels in batched_dataset:
            output = model({'images': images, 'labels': labels})
            epoch_acc.append(output)

        print(np.mean(epoch_acc))

if __name__ == '__main__':
    print('main started')    
    main()