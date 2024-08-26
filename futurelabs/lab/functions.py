from pathlib import Path

class Serializable:
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Serializable):
                result[key] = value.to_dict()
            elif not key.startswith('_'):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)  # Cria uma instância sem chamar __init__
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(obj, key, Serializable.from_dict(value))
            else:
                setattr(obj, key, value)
        return obj

class DirectoryManager:
    """Responsável pela criação e verificação de diretórios"""
    @staticmethod
    def create_folder_if_not_exists(path: Path):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

def get_section_folder(config: dict[str, str]) -> Path:
    work_folder = config["work_folder"]
    project_folder = config["project_name"].lower().replace(" ", "_")
    laboratory_folder = config["laboratory_name"].lower().replace(" ", "_")
    section_folder = config["section_name"].lower().replace(" ", "_")

    return Path(work_folder) / project_folder / laboratory_folder / section_folder

def get_project_folder(config: dict[str, str]) -> Path:
    work_folder = config["work_folder"]
    project_folder = config["project_name"].lower().replace(" ", "_")

    return Path(work_folder) / project_folder