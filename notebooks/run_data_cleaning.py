"""Execute cleaning in a new kernel using this exact Python interpreter."""
from pathlib import Path
import json
import sys
import tempfile
import argparse

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", choices=["01_data_cleaning.ipynb", "04_behaviour_features.ipynb"], default="01_data_cleaning.ipynb")
    args = parser.parse_args()
    path = root / "notebooks" / args.notebook
    notebook = nbformat.read(path, as_version=4)
    output = root / "outputs" / (path.stem + ".executed.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    # An isolated temporary spec avoids relying on a globally registered kernel.
    with tempfile.TemporaryDirectory(prefix="cleaning-kernel-") as directory:
        spec = Path(directory) / "cleaning"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Cleaning (current Python)", "language": "python",
        }))
        manager = KernelManager(transport="tcp" if sys.platform == "win32" else "ipc", kernel_name="cleaning", kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        client = NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            nbformat.write(notebook, output)
            for cell in notebook.cells:
                for item in cell.get("outputs", []):
                    if item.output_type == "stream":
                        print(item.text, end="")
    print(f"Fresh-kernel executed notebook: {output}")


if __name__ == "__main__":
    main()
