# Detección de prediabetes y diabetes con indicadores de salud autorreportados (BRFSS 2015)

Proyecto de aula de **Modelos y Simulación de Sistemas II**, Departamento de Ingeniería de Sistemas,
Universidad de Antioquia (2026-2).

**Integrantes:** Gabriel Jaime Cardona Montoya, Yuliana Corrales Castaño, Rebeca Bedoya Gallego.

Se compara la capacidad de cinco modelos de aprendizaje supervisado (regresión logística, KNN,
Random Forest, perceptrón multicapa y SVM) para detectar prediabetes o diabetes a partir de 21
indicadores de salud de la encuesta BRFSS 2015, y se evalúa si es posible reducir la dimensión del
problema con selección de variables, PCA y UMAP.

- **Informe (PDF, plantilla IEEE):** [`informe/informe.pdf`](informe/informe.pdf)
- **Notebook reproducible:** [`notebooks/proyecto_diabetes.ipynb`](notebooks/proyecto_diabetes.ipynb)

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── datos/                       # se crea al ejecutar; la base se descarga sola de UCI
├── notebooks/
│   ├── proyecto_diabetes.ipynb  # análisis completo, con las salidas de la última ejecución
│   ├── fuente_notebook.py       # mismo código en texto plano (facilita revisar cambios)
│   └── construir_notebook.py    # regenera el .ipynb a partir de fuente_notebook.py
├── resultados/
│   └── resultados.json          # todos los números que usa el informe
└── informe/
    ├── informe.pdf              # reporte final
    ├── main.tex                 # fuente LaTeX (IEEEtran, compatible con Overleaf)
    ├── referencias.bib
    ├── generar_tex.py           # pasa resultados.json a tablas y valores del informe
    ├── generado/                # tablas y valores generados (no editar a mano)
    └── figuras/                 # figuras generadas por el notebook
```

## Cómo reproducir los resultados

Requisitos: Python 3.10 o superior.

```bash
git clone https://github.com/gjcardonam/diabetes-brfss-ml.git
cd diabetes-brfss-ml
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_diabetes.ipynb
```

Ejecute todas las celdas en orden (*Run All*). La primera vez, el notebook descarga la base
*CDC Diabetes Health Indicators* (UCI, id 891) con `ucimlrepo` y la guarda en `datos/`.
La ejecución completa tarda unos 20 minutos en un equipo de 4 núcleos; la búsqueda de
hiperparámetros de la SVM y las proyecciones UMAP son las etapas más lentas.

Al terminar, el notebook deja:

- las figuras del informe en `informe/figuras/`, y
- todos los resultados numéricos en `resultados/resultados.json`.

Todas las particiones y modelos usan la semilla 42, así que los resultados son reproducibles.

También se puede ejecutar sin abrir Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto_diabetes.ipynb
```

## Cómo regenerar el informe

```bash
cd informe
python generar_tex.py      # actualiza generado/ a partir de resultados/resultados.json
latexmk -pdf main.tex      # requiere una distribución LaTeX con IEEEtran
```

En Overleaf: suba el contenido de la carpeta `informe/` (incluida `generado/` y `figuras/`)
y compile `main.tex` con pdfLaTeX.

## Contenido del notebook

1. Descripción del problema y de la base de datos: variables, significado, codificación,
   datos faltantes.
2. Análisis exploratorio: distribución de la clase, IMC y valores atípicos, prevalencia por
   edad, ingreso, educación y salud general, correlaciones.
3. Estado del arte (resumen).
4. Entrenamiento y evaluación: partición, submuestreo, malla de hiperparámetros con validación
   cruzada, efecto de los hiperparámetros y resultados de entrenamiento, validación y prueba con
   intervalos de confianza.
5. Reducción de dimensión: análisis individual de variables, PCA y UMAP con los dos mejores
   modelos.
6. Resumen de resultados.

## Datos

*CDC Diabetes Health Indicators*, UCI Machine Learning Repository, DOI
[10.24432/C53919](https://doi.org/10.24432/C53919). Derivada de la encuesta BRFSS 2015 de los
CDC de Estados Unidos por A. Teboul
([Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)).
