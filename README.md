# 🏡 End-to-End Project: Predicción de Precios de Vivienda

Este proyecto muestra el flujo completo (end-to-end) de un modelo de **Machine Learning para regresión**, entrenado para predecir precios de viviendas con base en sus características.  
Incluye el análisis de datos, creación del modelo, serialización (`.sav`) y código listo para producción.

---

## 🎯 Objetivo

- Predecir el precio de una propiedad a partir de características como número de habitaciones, superficie, ubicación, etc.
- Implementar un pipeline de ML completo y reproducible.
- Preparar el modelo para su despliegue en aplicaciones web o API.

---

## 📁 Estructura del proyecto

| Archivo                        | Descripción                                               |
|-------------------------------|------------------------------------------------------------|
| `end to end codigo.ipynb`     | Notebook con el flujo completo: análisis, modelos y resultados |
| `housing.csv`                 | Dataset con información de viviendas                      |
| `pipeline.sav`                | Pipeline de preprocesamiento + modelo serializado         |
| `modelLR.sav`                 | Modelo base de Regresión Lineal                          |
| `codigohousing.py`            | Código reutilizable para predicción o integración         |
| `requirements.txt`            | Librerías necesarias                                      |
| `logo.png`                    | Imagen de apoyo para visualizaciones o apps               |

---

## 🧰 Tecnologías utilizadas

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn
- Joblib (para guardar el pipeline)
- Jupyter Notebook

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/andrestorres29/end-to-end-project.git
cd end-to-end-project
