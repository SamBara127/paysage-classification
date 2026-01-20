from pathlib import Path

from dvc.repo import Repo


def check_dataset(dataset_size: int, path_dataset: Path) -> bool:
    if Path(path_dataset).is_dir():
        return dataset_size == sum(
            f.stat().st_size for f in Path(path_dataset).glob('**/*') if f.is_file()
        )
    return False


def download_dataset():
    try:
        repo = Repo()
        repo.pull()
    except BaseException:
        print('Не удалось получить датасет с DVC! Возможно, у вас нет доступа.')
        exit(-1)
