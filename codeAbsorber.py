import os
import json
from datetime import datetime
from termcolor import colored
import hashlib
import fnmatch
import dotenv

from agentRequests import getArchitectureSummary, getCodeArchitectureExplanation
dotenv.load_dotenv()

def filter_code_files(folder):
    """Filters code files that match the extensions list.

    Args:
        folder (str): The folder path.

    Returns:
        list: A list of file paths of code files.
    """
    extensions = [
        ".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".py", ".rb", ".js", ".ts", ".jsx", ".tsx",
        ".pl", ".pm", ".t", ".m", ".cs", ".swift", ".go", ".r", ".kt", ".kts",
        ".rs", ".lua", ".php", ".php3", ".php4", ".php5", ".phtml",
        ".xhtml", ".css", ".scss", ".less", ".sass", ".clj", ".cljs", ".cljc",
        ".asd", ".lisp", ".lsp", ".groovy", ".gvy", ".gy", ".gsh", ".bash", ".sh", 
        ".zsh", ".ksh", ".csh", ".fish", ".sql", ".ps1", ".psm1",
        ".bat", ".cmd", ".yaml", ".yml", ".xml", ".json", ".ini", ".cfg", ".conf",
        ".rst", ".rest", ".txt", ".d", ".hs", ".lhs", ".ml", ".mli",".fs", ".fsi",
        ".fsx", ".fsscript", ".purs", ".nim", ".v", ".b", ".cob", ".cbl", ".p",
        ".pas", ".pl", ".scala", ".sc", ".sbt", ".tcl", ".ada", ".adb", ".ads",
        ".asm", ".S", ".s", ".styl", ".pde", ".coffee", ".ktm", ".ktr", ".dart",
        ".elm", ".erl", ".hrl", ".ex", ".exs", ".hbs", ".handlebars", ".gql",
        ".graphql", ".ipynb", ".j", ".jl", ".aj", ".cl", ".smali", ".au3", ".bb",
        ".bmx", ".java", ".re", ".veo", ".hx", ".sls", ".applescript", ".scpt",
        ".sml", ".spy", ".proto", ".latex", ".tex", ".ltx", ".sty", ".cls", ".asd", ".s", ".vala", ".vapi",
        ".g", ".diff", ".patch", ".eco", ".em", ".ensime", ".factor", ".fy", ".fancypack",
        ".litcoffee", ".pogo", ".qml", ".qbs", ".qmlproject", ".pro", ".pri", ".slim",
        ".lsp", ".lisp", ".cl", ".fast", ".mumps"]
    

    excluded_folders = ["node_modules", ".next"]
    excluded_files = [".cognitive-coder/architecture.json", "package-lock.json", "yarn-lock.json"]
    gitignore_files = []

    # Check for .gitignore file
    if os.path.isfile(os.path.join(folder, ".gitignore")):
        with open(os.path.join(folder, ".gitignore"), "r") as gitignore:
            gitignore_files = [line.strip() for line in gitignore.readlines()]

    code_files = []
    for root, dirs, files in os.walk(folder):
        # Remove excluded directories from dirs list
        dirs[:] = [dir for dir in dirs if dir not in excluded_folders]

        for file in files:
            file_name = os.path.join(root, file).replace("\\", "/")
            if (
                not any(to_exclude in file_name for to_exclude in excluded_files)
                and not any(
                    fnmatch.fnmatch(file_name, os.path.join(folder, gitignore_file))
                    for gitignore_file in gitignore_files
                )
                and any(file_name.endswith(extension) for extension in extensions)
            ):
                code_files.append(file_name)

    return code_files


def get_file_hash(file_path):
    """Generates the hash of a file.

    Args:
        file_path (str): The file path.

    Returns:
        str: The sha256 hash of the file.
    """
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()
    
def generate_code_architecture(repo_folder, update=False):
    """Generates code architecture for a repository.

    Args:
        repo_folder (str): The folder path.
        update (bool, optional): Update existing architecture.json. Defaults to False.
    """

    code_files = filter_code_files(repo_folder)
    cog_coder_folder = os.path.join(repo_folder, '.cognitive-coder')
    architecture_file = os.path.join(cog_coder_folder, 'architecture.json')
    
    code_architecture = {'files': {}}
    if os.path.exists(architecture_file):
        print(colored(f"Found an already existing cognitive-coder repository ", "magenta", attrs=["bold"]))
        with open(architecture_file, 'r') as f:
            code_architecture = json.load(f)
    else:
        print(colored(f"Ingesting new codebase...", "green", attrs=["bold"]))
    updated = False
    print(colored(f"Analyzing {len(code_files)} code files:", "green"))
    for index, file_path in enumerate(code_files, 1):
        print(colored(f"({index}/{len(code_files)}) Analyzing {file_path}", "cyan"))

        file_hash = get_file_hash(file_path)
        existing_file_data = code_architecture['files'].get(file_path)
        if existing_file_data and existing_file_data['hash'] == file_hash:
            print(colored(f"Skipping {file_path} as the file hash hasn't changed", "yellow"))
            continue
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                code = f.read()
            summary = getCodeArchitectureExplanation(code) if code else "Empty"
            last_scan = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            code_architecture['files'][file_path] = {'summary': summary, 'hash': file_hash, 'last_scan': last_scan}
            updated = True

        if not os.path.exists(cog_coder_folder):
            os.makedirs(cog_coder_folder)

        with open(architecture_file, 'w') as f:
            json.dump(code_architecture, f, indent=2)
            print(colored(f"Saved architecture for {file_path}", "white"))

    if updated or not code_architecture['overview']:
        summary_str = '\n'.join([f"{file_path}:\n{entry['summary']}\n" for file_path, entry in code_architecture['files'].items()])
        architecture_summary = getArchitectureSummary(summary_str)
        code_architecture['overview'] = architecture_summary
        with open(architecture_file, 'w') as f:
            json.dump(code_architecture, f, indent=2)
            print(colored("Saved architecture summary.", "green"))

    print(colored("Code architecture analysis completed.", "green", attrs=["bold"]))

# print(filter_code_files('D:\ShortCloud\Controller'))

print(generate_code_architecture("D:\ShortCloud\Controller"))