"""Genera las tablas y los valores numéricos del informe a partir de resultados/resultados.json.

Uso (desde la carpeta informe/):  python generar_tex.py
Escribe generado/valores.tex y generado/tabla_*.tex, que main.tex incluye con \\input.
"""
import json
from pathlib import Path

AQUI = Path(__file__).parent
R = json.loads((AQUI.parent / "resultados" / "resultados.json").read_text(encoding="utf-8"))
SAL = AQUI / "generado"
SAL.mkdir(exist_ok=True)


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", "{,}")


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", "{,}") + "~\\%"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def ic(v):
    return f"{num(v['valor'])} ({num(v['ic95'][0])}--{num(v['ic95'][1])})"


def pm(v):
    return f"{num(v[0])}~$\\pm$~{num(v[1])}"


macros = {}
D, P = R["datos"], R["particion"]
macros.update({
    "nRegistros": miles(D["n_registros"]), "nVariables": str(D["n_variables"]),
    "prevalencia": pct(D["prevalencia"]), "prevalenciaNum": num(D["prevalencia"]),
    "nPositivos": miles(D["n_positivos"]), "nDuplicados": miles(D["duplicados"]),
    "nPrueba": miles(P["n_prueba"]), "nEntrenamiento": miles(P["n_entrenamiento"]),
    "nPool": miles(P["n_pool"]), "lineaBasePR": num(R["linea_base_auc_pr"]),
    "nBinarias": str(D["tipos"].get("Binaria", 0)), "nOrdinales": str(D["tipos"].get("Ordinal", 0)),
    "nNumericas": str(D["tipos"].get("Numérica", 0)),
})
E = R["eda"]
imc = E["imc"]
macros.update({
    "imcMediana": num(imc["mediana"], 0), "imcQuno": num(imc["q1"], 0), "imcQtres": num(imc["q3"], 0),
    "imcMin": num(imc["min"], 0), "imcMax": num(imc["max"], 0), "imcLimite": num(imc["limite_atipicos"], 1),
    "imcPctAtipicos": pct(imc["pct_atipicos"]), "imcMedianaSin": num(imc["mediana_sin"], 0),
    "imcMedianaCon": num(imc["mediana_con"], 0), "imcObesSin": pct(imc["pct_obesidad_sin"], 0),
    "imcObesCon": pct(imc["pct_obesidad_con"], 0),
    "edadMin": pct(E["prevalencia_edad"]["1"]), "edadMax": pct(max(E["prevalencia_edad"].values())),
    "edadUltimo": pct(E["prevalencia_edad"]["13"]),
    "ingresoDos": pct(E["prevalencia_ingreso"]["2"]), "ingresoOcho": pct(E["prevalencia_ingreso"]["8"]),
    "eduDos": pct(E["prevalencia_educacion"]["2"]), "eduSeis": pct(E["prevalencia_educacion"]["6"]),
    "saludUno": pct(E["prevalencia_salud_general"]["1"]), "saludCinco": pct(E["prevalencia_salud_general"]["5"]),
    "maxCorrPred": num(E["max_abs_correlacion_entre_predictoras"], 2),
})
M = R["modelos"]
for clave, nombre in [("LR", "Regresión logística"), ("KNN", "KNN"), ("RF", "Random Forest"),
                      ("MLP", "MLP"), ("SVM", "SVM")]:
    m = M[nombre]
    for met, etq in [("auc_roc", "Auc"), ("auc_pr", "Ap"), ("sensibilidad", "Sens"),
                     ("especificidad", "Esp"), ("precision", "Prec"), ("f1", "Funo"),
                     ("exactitud_balanceada", "Ba")]:
        macros[f"prueba{etq}{clave}"] = num(m["prueba"][met]["valor"])
        macros[f"val{etq}{clave}"] = num(m["validacion"][met][0])
        macros[f"de{etq}{clave}"] = num(m["validacion"][met][1])
        macros[f"ent{etq}{clave}"] = num(m["entrenamiento"][met][0])
macros["controlAuc"] = num(R["control_lr_completa"]["auc_roc"])
macros["controlN"] = miles(R["control_lr_completa"]["n"])
B = R["bloques"]
for clave, nombre in [("Social", "Socioeconómico y acceso"), ("Demo", "Demográfico"),
                      ("Estilo", "Estilo de vida"), ("Clinico", "Clínico"),
                      ("SocialDemo", "Social + demográfico"), ("Todas", "Todas")]:
    macros[f"bloqueRf{clave}"] = num(B[nombre]["auc_rf"])
    macros[f"bloqueLr{clave}"] = num(B[nombre]["auc_lr"])
Rd = R["reduccion"]
macros.update({
    "pcaKaiser": str(Rd["pca"]["k_kaiser"]), "pcaNoventa": str(Rd["pca"]["k_90"]),
    "pcaVarKaiser": pct(Rd["pca"]["varianza_kaiser"]),
    "pcaRedKaiser": pct(1 - Rd["pca"]["k_kaiser"] / 21, 0), "pcaRedNoventa": pct(1 - Rd["pca"]["k_90"] / 21, 0),
    "pcaPcUno": pct(Rd["pca"]["varianza_acumulada"][0]),
    "umapK": str(Rd["umap"]["k_elegido"]), "umapRed": pct(1 - Rd["umap"]["k_elegido"] / 21, 0),
    "nCandidatas": str(len(Rd["candidatas_eliminar"])),
    "nConservadas": str(Rd["seleccion"]["n_variables"]),
    "selRed": pct(1 - Rd["seleccion"]["n_variables"] / 21, 0),
    "mejorUno": Rd["mejores_modelos"][0], "mejorDos": Rd["mejores_modelos"][1],
})

with open(SAL / "valores.tex", "w", encoding="utf-8") as f:
    f.write("% Generado por generar_tex.py a partir de resultados/resultados.json. No editar a mano.\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- Tabla de variables
NOMBRE_TEX = lambda v: v.replace("_", "\\_")


def tex(t):
    for a, b in [("≥", "$\\geq$"), ("…", "\\ldots{}"), ("²", "$^2$"), ("<", "$<$")]:
        t = t.replace(a, b)
    return t
filas = []
for d in R["diccionario"]:
    rango = f"{d['minimo']}--{d['maximo']}"
    filas.append(f"\\texttt{{{NOMBRE_TEX(d['variable'])}}} & {tex(d['significado'])} & {d['tipo']} & "
                 f"{rango} & {tex(d['codificacion'])} \\\\")
(SAL / "tabla_variables.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- Tabla de mallas
ETQ_PARAM = {"C": "$C$", "n_neighbors": "$k$", "weights": "pesos", "max_depth": "prof. máx.",
             "min_samples_leaf": "mín. muestras por hoja", "hidden_layer_sizes": "capas ocultas",
             "alpha": "$\\alpha$", "gamma": "$\\gamma$"}
ETQ_VAL = {"uniform": "uniformes", "distance": "distancia", "None": "sin límite", "scale": "escala",
           "(32,)": "(32)", "(64, 32)": "(64, 32)", "0.0001": "$10^{-4}$", "0.01": "0,01", "0.001": "0,001",
           "0.1": "0,1", "1": "1", "10": "10"}
fijos = {"Regresión logística": "", "KNN": "", "Random Forest": "300 árboles; ",
         "MLP": "ReLU, Adam, parada temprana; ", "SVM": "kernel RBF; "}
filas = []
for nombre, m in M.items():
    malla = "; ".join(f"{ETQ_PARAM[p]} $\\in$ \\{{{'; '.join(ETQ_VAL.get(v, v) for v in vals)}\\}}"
                      for p, vals in m["malla"].items())
    elegido = ", ".join(f"{ETQ_PARAM[p]} = {ETQ_VAL.get(v, v)}" for p, v in m["mejores_parametros"].items())
    filas.append(f"{m['familia']} & {nombre} & {fijos[nombre]}{malla} & {elegido} \\\\")
(SAL / "tabla_malla.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- Tabla de resultados
METS = ["auc_roc", "auc_pr", "sensibilidad", "especificidad", "f1", "exactitud_balanceada"]
filas = []
for nombre, m in M.items():
    ent = " & ".join(pm(m["entrenamiento"][k]) for k in METS)
    val = " & ".join(pm(m["validacion"][k]) for k in METS)
    pru = " & ".join(f"{num(m['prueba'][k]['valor'])}" for k in METS)
    ics = " & ".join(f"\\scriptsize({num(m['prueba'][k]['ic95'][0])}--{num(m['prueba'][k]['ic95'][1])})"
                     for k in METS)
    filas.append(f"\\multirow{{4}}{{*}}{{{nombre}}} & Entrenamiento & {ent} \\\\")
    filas.append(f" & Validación & {val} \\\\")
    filas.append(f" & Prueba & {pru} \\\\")
    filas.append(f" & \\scriptsize IC 95~\\% & {ics} \\\\ \\midrule")
filas[-1] = filas[-1].replace(" \\midrule", "")
(SAL / "tabla_resultados.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- Tabla de bloques
filas = []
for nombre, b in B.items():
    filas.append(f"{nombre} & {b['n_variables']} & {num(b['auc_lr'])} & {num(b['auc_rf'])} \\\\")
(SAL / "tabla_bloques.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- Tabla de reducción
repr_ = [("Variables originales", Rd["referencia_21"], 21, "cv"),
         (f"Selección (sin {len(Rd['candidatas_eliminar'])} candidatas)", Rd["seleccion"],
          Rd["seleccion"]["n_variables"], "cv"),
         ("PCA, criterio de Kaiser", Rd["pca"]["kaiser"], Rd["pca"]["k_kaiser"], "cv"),
         ("PCA, 90~\\% de varianza", Rd["pca"]["var90"], Rd["pca"]["k_90"], "cv"),
         ("UMAP", Rd["umap"]["resultados"], Rd["umap"]["k_elegido"], "holdout")]
filas = []
for etiqueta, res, k, tipo in repr_:
    primera = True
    for nombre in Rd["mejores_modelos"]:
        r = res[nombre]
        if tipo == "cv":
            val = pm(r["validacion_auc"])
        else:
            val = num(r["validacion_auc_k"]) + "$^{\\dagger}$"
        p = r["prueba"]
        cab = (f"\\multirow{{2}}{{*}}{{\\parbox{{2.6cm}}{{{etiqueta}}}}} & \\multirow{{2}}{{*}}{{{k}}} & "
               f"\\multirow{{2}}{{*}}{{{pct(1 - k / 21, 0)}}}") if primera else " & & "
        filas.append(f"{cab} & {nombre} & {val} & {ic(p['auc_roc'])} & {num(p['auc_pr']['valor'])} & "
                     f"{num(p['exactitud_balanceada']['valor'])} \\\\")
        primera = False
    filas[-1] += " \\midrule"
filas[-1] = filas[-1].replace(" \\midrule", "")
(SAL / "tabla_reduccion.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")
print(f"{len(macros)} valores y 5 tablas en {SAL}")
