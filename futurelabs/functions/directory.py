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

    def get_projects(self):
        projects = set()
        folders = self.list_folders(self.root)
        if len(folders) == 0:
            return []
        for folder in folders[0]["child"]:
            projects.add(self._format_folder_name(folder['folder']))
        return list(projects)

    def get_labs(self, parent_folder: str):
        parent_folder = os.path.join(self.root, parent_folder.lower().replace(" ", "_"))
        labs = set()
        folders = self.list_folders(parent_folder)
        if len(folders) == 0:
            return []
        for folder in folders[0]["child"]:
            labs.add(self._format_folder_name(folder['folder']))
        return list(labs)

    def get_sections(self, project: str, lab: str):
        project = project.lower().replace(" ", "_")
        lab = lab.lower().replace(" ", "_")
        parent_folder = os.path.join(self.root, project, lab)
        sections = set()
        folders = self.list_folders(parent_folder)
        if len(folders) == 0:
            return [], None
        for folder in folders[0]["child"]:
            sections.add(self._format_folder_name(folder['folder']))
        return list(sections), parent_folder

    def get_chart(self, project: str, lab: str, section: str):
        project = project.lower().replace(" ", "_")
        lab = lab.lower().replace(" ", "_")
        section = section.lower().replace(" ", "_")
        parent_folder = os.path.join(self.root, project, lab, section)
        charts = set()
        folders = self.list_folders(parent_folder)
        if len(folders) == 0:
            return [], None
        for folder in folders[0]["child"]:
            charts.add(self._format_folder_name(folder['folder']))
        return list(charts), parent_folder


    def _format_folder_name(self, folder_name: str) -> str:
        return folder_name.replace("_", " ").capitalize()