import argparse
import torch
from torchvision import transforms
import kagglehub
from pathlib import Path

from splitter import read_splits, create_splits
from preprocess import SceneDataset
from test import model_test
from train import model_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Аргументы для пайплайна')

    parser.add_argument('--mode', type=str, default='train', help='Режим работы')
    parser.add_argument('--e', type=int, default=10, help='Количество эпох обучения')
    parser.add_argument('--bs', type=int, default=256, help='Размер батча данных для обучения и теста')
    parser.add_argument('--ss', type=int, default=150, help='К квадрату какого размера будут приведены изображения')
    parser.add_argument('--path_labels', type=str, default='data/labels.txt', help='Путь до текстового файла с названиями классов')
    parser.add_argument('--path_splits', type=str, default='data/', help='Путь до CSVшек с выборками train-eval-test, включая префикс')
    parser.add_argument('--path_checkpoint', type=str, default='scene_classifier.pt', help='Файл для сохранения и загрузки чекпоинтов')
    parser.add_argument('--dataset_name', type=str, default='nitishabharathi/scene-classification', help='Название датасета на KaggleHub')
    parser.add_argument('--dataset_splits', type=str, default='0.7-0.2-0.1', help='Как разделить датасет (три числа от 0 до 1 через дефис)')

    args = parser.parse_args()

    # Для начала, скачем датасет! Если оный уже был скачан,
    # KaggleHub сам определит это и ничего перекачивать не будет.
    PATH_DATASET = Path(
        kagglehub.dataset_download(args.dataset_name)
    )
    # print(PATH_DATASET)

    PATH_IMAGES = PATH_DATASET / 'train-scene classification' / 'train'
    PATH_CSV = PATH_DATASET / 'train-scene classification' / 'train.csv'

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(args.path_labels, 'r', encoding='utf-8')
    label_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))
    labels = {label_no: label_name for label_no, label_name in enumerate(label_list)}

    # Как мы хочем выполнить препроцессинг изображений
    transform_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([args.ss, args.ss]), # Ориг. разрешение: [150, 150]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f'Подготовка к {args.mode} завершена, обработка данных...')


    if args.mode == 'train':
        # Читаем CSV-шку со сплитами на train-eval-test,
        # либо создаём её с нуля, если таковой не существует
        try:
            [train_names, eval_names] = read_splits(['train', 'eval'], args.path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(PATH_CSV, args.dataset_splits, args.path_splits)
            [train_names, eval_names] = read_splits(['train', 'eval'], args.path_splits)

        # Обучаем с нуля модель и сохраняем чекпоинты
        model = model_train(
            train_names,
            eval_names,
            PATH_IMAGES,
            labels,
            transform_list,
            args.e,
            args.bs
        )

        torch.save(model, args.path_checkpoint)
        print('Обучение завершено!')


    if args.mode == 'test':
        # Читаем CSV-шку со сплитами на train-eval-test,
        # либо создаём её с нуля, если таковой не существует
        try:
            [test_names] = read_splits(['test'], args.path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(PATH_CSV, args.dataset_splits, args.path_splits)
            [test_names] = read_splits(['test'], args.path_splits)


        # Загружаем модель и прогоняем на тестовых данных
        model = torch.load(args.path_checkpoint, weights_only=False)

        predictions = model_test(
            model,
            test_names,
            PATH_IMAGES,
            transform_list,
            args.bs
        )

        results = torch.cat(predictions, dim=0)
        # print(results)
        unique, counts = results.unique(return_counts=True)

        print('Результаты:')
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            print(f'{labels.get(cls)} -- {cnt} шт.')
