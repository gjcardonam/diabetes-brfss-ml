"""Convierte fuente_notebook.py (celdas marcadas con # %%) en proyecto_diabetes.ipynb."""
import re
from pathlib import Path
import nbformat

AQUI = Path(__file__).parent
texto = (AQUI / "fuente_notebook.py").read_text(encoding="utf-8")
bloques = re.split(r"^# %%(.*)$", texto, flags=re.M)
celdas = []
for marca, cuerpo in zip(bloques[1::2], bloques[2::2]):
    cuerpo = cuerpo.strip("\n")
    if "[markdown]" in marca:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in cuerpo.splitlines())
        celdas.append(nbformat.v4.new_markdown_cell(md))
    else:
        celdas.append(nbformat.v4.new_code_cell(cuerpo))
nb = nbformat.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, AQUI / "proyecto_diabetes.ipynb")
print(len(celdas), "celdas")
