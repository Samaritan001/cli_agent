import ast
import logging
import os
from pathlib import Path

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

def get_full_headers():
    current_dir = Path.cwd()
    
    # Iterate through items in the directory
    for folder in current_dir.iterdir():
        # Check if it's a directory and ends with '_server'
        if folder.is_dir() and folder.name.endswith('_server'):
            folder_name = folder.name.split('_')
            source_file_prefix = '_'.join(folder_name[:-2])
            source_file_suffix = folder_name[-2]
            source_file_name = source_file_prefix + "." + source_file_suffix
            
            # Look for the source file inside that folder
            source_file_path = folder / source_file_name
            output_file_path = current_dir / "docs" / f"{source_file_prefix}.md"
            
            logging.info(f"--- Checking folder: {folder.name} ---")
            
            try:
                with open(source_file_path, "r") as file:
                    tree = ast.parse(file.read())
            
                try:
                    with open(output_file_path, "w") as f:
                        summary = ast.get_docstring(tree)
                        f.write(summary + "\n")
                        f.write(str(folder / (source_file_prefix + "_cli." + source_file_suffix)) + "\n")
                        logging.info(f"Server description written to docs/{source_file_prefix}.md")

                        for node in tree.body:
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                # Get arguments string: (a: int, b: str = "default")
                                args_str = ast.unparse(node.args)
                                
                                # Get return annotation: -> str
                                returns = ""
                                if node.returns:
                                    returns = f" -> {ast.unparse(node.returns)}"
                                
                                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                                sig = f"{prefix} {node.name}({args_str}){returns}:"
                                docstring = ast.get_docstring(node)
                                
                                header = f"Function: {node.name}\n{sig}\nDescription: {docstring}\n"
                                f.write("-"*30 + "\n" + header)
                    
                    logging.info(f"Tool documentation written to docs/{source_file_prefix}.md")

                except Exception as e:
                    logging.error(f"Error writing documentation {output_file_path}: {e}")
            
            except Exception as e:
                logging.error(f"Error reading file {source_file_path}: {e}")
            
def main():
    get_full_headers()

if __name__ == "__main__":
    main()


