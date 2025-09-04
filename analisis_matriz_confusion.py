
def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def as_percent(value: float) -> str:
    """Formatea un valor [0,1] como porcentaje con 2 decimales."""
    return f"{value * 100:.2f}%"


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """
    Calcula métricas de una matriz de confusión binaria.

    Parámetros:
      - tp: verdaderos positivos
      - fp: falsos positivos
      - tn: verdaderos negativos
      - fn: falsos negativos

    Supuesto: la clase positiva es "maligno".

    Fórmulas usadas:
      - ACC = (TP + TN) / T
      - SEN/Recall/TPR = TP / (TP + FN)
      - SPE/TNR = TN / (TN + FP)
      - PPV (Precisión) = TP / (TP + FP)
      - NPV = TN / (TN + FN)
      - FDR = FP / (FP + TP)
      - FNR = FN / (TP + FN)
      - FPR = FP / (FP + TN)
      - Lift = [TP/(TP+FP)] / [ (TP+FN)/T ]
      - F1 = (2 * TP) / (2*TP + FP + FN)
    """
    # Total de observaciones en el conjunto de evaluación
    total: int = tp + fp + tn + fn

    # Exactitud: proporción de aciertos globales
    accuracy = safe_div(tp + tn, total)

    # Precisión (PPV): confiabilidad de una predicción positiva
    precision = safe_div(tp, tp + fp)  # PPV

    # Recall/Sensibilidad (TPR): cobertura de los positivos reales
    recall = safe_div(tp, tp + fn)  # Sensibilidad / TPR

    # Especificidad (TNR): cobertura de los negativos reales
    specificity = safe_div(tn, tn + fp)  # TNR

    # F1: media armónica entre precisión y recall (robusta ante desbalance)
    f1 = safe_div(2 * tp, (2 * tp) + fp + fn)

    # NPV: confiabilidad de una predicción negativa (TN / (TN + FN))
    npv = safe_div(tn, tn + fn)

    # Tasas de error por clase
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    # FDR (False Discovery Rate): FP / (FP + TP) = 1 - PPV
    fdr = safe_div(fp, fp + tp)

    # Lift: [TP/(TP+FP)] / [ (TP+FN)/T ] = PPV / Prevalence
    prevalence = safe_div(tp + fn, total)
    lift = safe_div(precision, prevalence)

    return {
        "total": float(total),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "prevalence": prevalence,
        "predicted_positive_rate": safe_div(tp + fp, total),
        "fdr": fdr,
        "lift": lift,
    }


def print_report(metrics: dict[str, float], as_table: bool = False) -> None:
    """Imprime las métricas calculadas.

    Parámetros:
      - metrics: diccionario devuelto por compute_metrics
      - as_table: si es True, muestra una tabla alineada con las métricas
                  en el mismo orden de la imagen compartida.
    """

    print("=== Conteos ===")
    print(f"Muestras totales: {int(metrics['total'])}")
    print(f"Verdaderos positivos (TP): {int(metrics['tp'])}")
    print(f"Falsos positivos    (FP): {int(metrics['fp'])}")
    print(f"Verdaderos negativos (TN): {int(metrics['tn'])}")
    print(f"Falsos negativos    (FN): {int(metrics['fn'])}")

    if as_table:
        print("\n=== Métricas (tabla) ===")
        rows = [
            ("Exactitud (ACC)", as_percent(metrics['accuracy'])),
            ("Sensibilidad (SEN) / Recall / TPR", as_percent(metrics['recall'])),
            ("Especificidad (SPE) / TNR", as_percent(metrics['specificity'])),
            ("Precisión / Valor Predictivo Positivo (PPV)", as_percent(metrics['precision'])),
            ("Valor Predictivo Negativo (NPV)", as_percent(metrics['npv'])),
            ("Tasa de descubrimiento falso (FDR)", as_percent(metrics['fdr'])),
            ("Tasa de falsos negativos (FNR)", as_percent(metrics['fnr'])),
            ("Tasa de falsos positivos (FPR)", as_percent(metrics['fpr'])),
            ("Índice de elevación (Lift)", f"{metrics['lift']:.3f}"),
            ("F1-Score", as_percent(metrics['f1'])),
        ]
        col1 = max(len(name) for name, _ in rows) + 2
        print("+" + "-" * (col1 + 14) + "+")
        for name, value in rows:
            print(f"| {name.ljust(col1)}{value.rjust(12)} |")
        print("+" + "-" * (col1 + 14) + "+")
    else:
        print("\n=== Métricas ===")
        print(f"Exactitud (ACC):                 {as_percent(metrics['accuracy'])}")
        print(f"Sensibilidad (SEN) / Recall / TPR: {as_percent(metrics['recall'])}")
        print(f"Especificidad (SPE) / TNR:       {as_percent(metrics['specificity'])}")
        print(f"Precisión / Valor Predictivo Positivo (PPV): {as_percent(metrics['precision'])}")
        print(f"Valor Predictivo Negativo (NPV): {as_percent(metrics['npv'])}")
        print(f"Tasa de descubrimiento falso (FDR): {as_percent(metrics['fdr'])}")
        print(f"Tasa de falsos negativos (FNR):  {as_percent(metrics['fnr'])}")
        print(f"Tasa de falsos positivos (FPR):  {as_percent(metrics['fpr'])}")
        print(f"Índice de elevación (Lift):      {metrics['lift']:.3f}")
        print(f"F1-Score:                        {as_percent(metrics['f1'])}")

    print("\nInterpretación breve:")


if __name__ == "__main__":

    TRUE_POSITIVES = 50
    FALSE_NEGATIVES = 17
    FALSE_POSITIVES = 23
    TRUE_NEGATIVES = 10

    results = compute_metrics(
        tp=TRUE_POSITIVES,
        fp=FALSE_POSITIVES,
        tn=TRUE_NEGATIVES,
        fn=FALSE_NEGATIVES,
    )
    print_report(results,True)

"""
### Interpretación
- Con TP=50, FP=23, TN=10, FN=17 (N=100), el modelo acierta el 60%. Detecta la mayoría de malignos, pero falla mucho en reconocer no malignos y en descartar con seguridad.

### ¿Qué significa cada métrica obtenida?
- Exactitud (ACC) 60.00%: aciertos totales sobre 100 casos. Es menor que predecir “todo maligno” (≈67%), señal de bajo desempeño global.
- Sensibilidad/Recall/TPR 74.63%: detecta ~3 de cada 4 malignos reales; se escapan 25.37% (FNR).
- Especificidad/TNR 30.30%: solo 3 de cada 10 no malignos son reconocidos; FPR 69.70% implica muchos falsos alarmas.
- Precisión/PPV 68.49%: de cada 100 predicciones “maligno”, ~68 son realmente malignos; FDR 31.51% son falsos.
- NPV 37.04%: una predicción “no maligno” es correcta solo ~37% del tiempo, por lo que no es segura para descartar.
- Lift 1.022: apenas 2.2% mejor que elegir positivos al ritmo de la prevalencia; mejora marginal.
- F1-Score 71.43%: equilibrio moderado entre precisión y sensibilidad, pero no compensa la baja especificidad/NPV.

### Calidad y confiabilidad
- Globalmente, la calidad es baja y no confiable clínicamente para “descartar” cáncer ni para evitar sobre-detecciones:
- Muy baja especificidad y NPV → muchas falsas alarmas y “no maligno” poco fiable.
- Sensibilidad aceptable pero con 1 de cada 4 malignos omitidos.
- Exactitud por debajo del baseline ingenuo (todo positivo).

### Qué mejorar
- Ajuste de umbral y curvas ROC/PR para priorizar sensibilidad alta con mejor especificidad.
"""