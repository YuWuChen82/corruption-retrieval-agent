import os
def get_project_root() ->str:
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    return project_root

def get_abs_path(relative_path: str) -> str:
    return os.path.join(get_project_root(), relative_path)

if __name__ == '__main__':
    print(get_abs_path('test'))