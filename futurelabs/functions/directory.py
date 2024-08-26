# Módulo de manipulação de diretórios
import os


class DirectoryTree:
    def __init__(self, root: str):
        self.root = root

    def directory_exists(self, parent_folder: str) -> bool:
        return os.path.isdir(parent_folder)


    def list_folders(self, parent_folder: str) -> list:
        if self.directory_exists(parent_folder):
            return [self._build_tree(parent_folder)]

        return []

    def _build_tree(self, folder: str) -> dict:
        child_folders = []
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                child_folders.append(self._build_tree(item_path))
        return {
            "folder": os.path.basename(folder),
            "child": child_folders
        }

    def _join_folder(self, parent_folder: str, folder: str = None) -> str:

        if parent_folder:
            parent_folder = parent_folder.lower().replace(" ", "_")

        if folder:
            folder = folder.lower().replace(" ", "_")
            parent_folder = os.path.join(parent_folder, folder)

        folders = self.list_folders(parent_folder)
        if len(folders) == 0:
            return [], None

        sections = set()
        for folder in folders[0]["child"]:
            sections.add(self._format_folder_name(folder['folder']))

        return list(sections), parent_folder


    def get_projects(self):
        return self._join_folder(self.root)

    def get_labs(self, parent_folder: str, folder: str = None) -> [list, str]:
        return self._join_folder(parent_folder, folder)

    def get_sections(self, parent_folder: str, folder: str) -> [list, str]:
        return self._join_folder(parent_folder, folder)

    def get_chart(self, parent_folder: str, folder: str) -> [list, str]:
        return self._join_folder(parent_folder, folder)


    def _format_folder_name(self, folder_name: str) -> str:
        return folder_name.replace("_", " ").capitalize()



