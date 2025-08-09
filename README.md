# 🤖 Análisis Predictivo de Evasión de Clientes - TelecomX LATAM (Parte 2)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=for-the-badge&logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24.2-F7931E.svg?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success.svg?style=for-the-badge)

_Un proyecto de Machine Learning de extremo a extremo para construir, evaluar e interpretar modelos predictivos que anticipan la evasión de clientes y guían las estrategias de retención._

[**🚀 Ver el Análisis**](#-ejecución-directa-en-colab) **|** [**📊 Principales Hallazgos**](#-principales-hallazgos-del-modelo) **|** [**💡 Recomendaciones**](#-recomendaciones-estratégicas) **|** [**🛠️ Metodología**](#️-metodología)

</div>

---

## 🎯 Descripción del Proyecto

Este proyecto es la segunda fase del análisis de **TelecomX LATAM**, centrado en el desarrollo de un pipeline de Machine Learning para **predecir la evasión de clientes (churn)**. A partir de los datos limpios de la Parte 1, construimos y comparamos múltiples modelos de clasificación para identificar a los clientes con mayor riesgo de cancelar.

El objetivo final no es solo predecir, sino **interpretar los resultados del mejor modelo** para entender *por qué* los clientes se van y proponer acciones de retención proactivas y basadas en datos.

### 🌟 Características Destacadas

-   ✅ **Pipeline de Machine Learning de Extremo a Extremo:** Desde el preprocesamiento hasta la evaluación y la interpretabilidad del modelo.
-   ✅ **Manejo Avanzado del Desbalance de Clases:** Uso de **SMOTE** (Synthetic Minority Over-sampling Technique) para entrenar modelos robustos y evitar el sesgo hacia la clase mayoritaria.
-   ✅ **Comparación Robusta de Modelos:** Evaluación y comparación de `Regresión Logística`, `Random Forest` y `HistGradientBoosting` utilizando métricas clave como **ROC-AUC** y **Recall**.
-   ✅ **Interpretabilidad del Modelo Avanzada:** Análisis profundo de los factores de churn utilizando **Permutation Importance** y **SHAP** (SHapley Additive exPlanations) para entender las predicciones.
-   ✅ **Recomendaciones de Negocio Accionables:** Estrategias de retención directamente derivadas de los resultados del modelo predictivo.
-   ✅ **Código Reproducible y Mejorado:** El notebook está documentado para una fácil ejecución y comprensión, incluyendo mejoras sobre el análisis original.

---

## 🚀 Inicio Rápido

### Prerrequisitos

-   Python 3.8+
-   Jupyter Notebook o Google Colab
-   Conexión a internet

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/frenrey3/Desafio-TelecomX-Churn-Analysis-Parte2.git
    cd Desafio-TelecomX-Churn-Analysis-Parte2
    ```

2.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: `requirements.txt` debería incluir `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `shap` y `xgboost`.*

3.  **Ejecuta el análisis:**
    Abre y ejecuta el notebook principal.
    ```bash
    jupyter notebook "analisis_predictivo_churn_telecomx_entrega.ipynb"
    ```

### 🔗 Ejecución Directa en Colab

No necesitas instalar nada. ¡Ejecuta el análisis completo directamente en tu navegador!

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frenrey3/Desafio-TelecomX-Churn-Analysis-Parte2/blob/main/analisis_predictivo_churn_telecomx_entrega.ipynb)
_**Nota:** Reemplaza la URL con el enlace final a tu notebook en GitHub._

---

## 📊 Dataset

| Característica | Detalle                                                                                                                                                             |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Fuente**     | `https://objectstorage.us-sanjose-1.oraclecloud.com/.../telecom_churn_cleaned.csv`                                                                                   |
| **Registros**  | 7,043 clientes                                                                                                                                                       |
| **Variables**  | 22 características procesadas                                                                                                                                         |
| **Formato**    | CSV → DataFrame de Pandas                                                                                                                                              |
| **Objetivo**   | `churn` (1 si el cliente canceló, 0 si no)                                                                                                                           |

---

## 📈 Principales Hallazgos del Modelo

Tras comparar varios algoritmos, el modelo de **Regresión Logística**, entrenado sobre datos balanceados con SMOTE, fue seleccionado como el de mejor rendimiento general.

1.  **🏆 Rendimiento del Modelo:** El modelo de Regresión Logística alcanzó un **ROC-AUC de 0.84** y un **Recall del 73%** para la clase "Churn". Esto significa que es altamente efectivo para identificar a los clientes que realmente están en riesgo de cancelar.

2.  **📉 Principal Factor de Retención: `tenure` (Antigüedad):** Es, con diferencia, el predictor más fuerte. A mayor antigüedad del cliente, menor es su probabilidad de evasión.

3.  **📝 Contratos a Largo Plazo:** Tener un contrato de `one_year` o, especialmente, de `two_year`, reduce drásticamente la probabilidad de churn. El contrato `month-to-month` es un fuerte indicador de riesgo.

4.  **🌐 Internet de Fibra Óptica:** De manera contraintuitiva, contratar `internetservice_fiber_optic` está positivamente correlacionado con el churn, sugiriendo que las expectativas de los clientes con este servicio premium (posiblemente en precio o calidad) no se están cumpliendo.

5.  **🛡️ Servicios de Soporte y Seguridad:** La ausencia de `onlinesecurity` y `techsupport` son factores importantes que aumentan la probabilidad de evasión.

---

## 🛠️ Metodología

El proyecto sigue un pipeline de Machine Learning estructurado para garantizar la robustez y reproducibilidad de los resultados.

```mermaid
graph LR
    A[🧹 Preprocesamiento y Encoding] --> B[⚖️ Manejo de Desbalance con SMOTE];
    B --> C[⚙️ Entrenamiento de Modelos];
    C --> D[📊 Evaluación y Selección];
    D --> E[🧠 Interpretabilidad con SHAP];
    E --> F[🚀 Recomendaciones Estratégicas];
