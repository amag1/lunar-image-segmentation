# Segmentación de imágenes de renders lunares

## Alumnos
- Andrés Maglione
- Yeumen Silva

## Importación del conjunto de datos
Por motivos de espacio, el conjunto de datos no se incluye en el repositorio. Para poder ejecutar el código es necesario descargarlo desde Kaggle a través del siguiente enlace: [Lunar Renders Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset).
También se ofrece un enlace alternativo para descargar el conjunto de datos desde Google Drive: [Lunar Renders Dataset - Google Drive](https://drive.google.com/drive/folders/1cNMRqZToX1X4S8pkTGWw2KEw8lR-ijIx?usp=sharing).

Una vez descargado el conjunto de datos, es necesario descomprimirlo en la carpeta `dataset/` del proyecto. Se espera que la estructura de carpetas sea la siguiente:

```
dataset
├── images
│   ├── clean
│   |   ├── clean0001.png
│   ├── ground
│   |   ├── ground0001.png
│   └── render
│       ├── render0001.png
```

## Estructura del proyecto
El proyecto cuenta con un script principal `main.py` que se encarga de ejecutar el flujo de trabajo comlpeto: el mismo llama a los tres posibles pipelines de segmentación (definidos en `/pipelines`) para cada una de las imágenes del dataset. Posteriormnete, se guarda la mejor imagen dentro de la carpeta `/dataset/images/resulting_masks`.

En la raíz del proyecto se encuentran los siguientes archivos y carpetas:

|Archivo/Carpeta|Descripción|
|---|---|
|`README.md`|Este archivo, que contiene la documentación del proyecto.|
|`main.py`|Script principal que ejecuta el flujo de trabajo completo.|
|`loss.py`|Archivo que contiene las funciones de pérdida utilizadas en los pipelines de segmentación.|
|`remove_red.py`|Archivo que contiene la función para eliminar la parte roja de las imágenes.|
|`eval.py`|Contiene una función evaluadora que permite medir la calidad de una función de segmentación a partir de una función de pérdida.|
|`metrics.py`| Debe ejecutarse luego de `main.py` para evaluar las métricas de las imágenes segmentadas.|
|`dataset/`|Carpeta que contiene el conjunto de datos.|
|`pipelines/`|Carpeta que contiene los pipelines de segmentación.|
|`notebooks/`|Carpeta que contiene los notebooks de prueba y experimentación.|
|`metrics_outputs/`|Carpeta que contiene los histogramas de las métricas obtenidas, junto con un CSV resumen de las mismas|
|`requirements.txt`|Archivo que contiene las dependencias del proyecto.|

## Ejecución del proyecto
Para ejecutar el proyecto, es necesario tener instalado Python 3.8 o superior y las dependencias del proyecto. Se recomienda crear un entorno virtual para evitar conflictos con otras dependencias.
Para instalar las dependencias, se puede utilizar el siguiente comando:

```bash
pip install -r requirements.txt
```
Una vez instaladas las dependencias, se puede ejecutar el script principal con el siguiente comando:

```bash
python3 main.py
```

Esto ejecutará el flujo de trabajo completo y generará las imágenes segmentadas en la carpeta `dataset/images/resulting_masks`.

## Formato de los resultados
Como se mencionó anteriormente, el script solo guarda la mejor máscara para cada imagen. Dentro del directorio `dataset/images/resulting_masks/` se podrán encontrar tres subdirectorios que corresponden a cada uno de los pipelines de segmentación implementados. Cada subdirectorio contendrá las imágenes segmentadas generadas por el pipeline correspondiente, con el siguiente formato:

```
dataset/images/resulting_masks/
├── bright
│   ├── result0001-lossXX.png
|   ├── result0002-lossXX.png
├── dark
│   ├── result0001-lossXX.png
|   ├── result0002-lossXX.png
├── flat
│   ├── result0001-lossXX.png
|   ├── result0002-lossXX.png
```

Donde `XX` representa el valor de la pérdida obtenido para cada imagen segmentada. El script guarda la mejor máscara para cada imagen, es decir, aquella que tiene el menor valor de pérdida.

Para más información sobre los pipelines de segmentación utilizados, consultar el informe final del proyecto, que los explica en detalle.

## Evaluación de las métricas
Para evaluar las métricas de las imágenes segmentadas, se debe ejecutar el script `metrics.py` una vez que se hayan generado las imágenes segmentadas. Este script calculará la precisión de las imágenes segmentadas y producirá histogramas de las métricas obtenidas. Para ejecutarlo, se puede utilizar el siguiente comando:

```bash
python3 metrics.py
```

## Notebooks de prueba y experimentación
En la carpeta `notebooks/` se encuentran varios notebooks que contienen pruebas y experimentos realizados durante el desarrollo del proyecto. Estos notebooks pueden ser útiles para entender mejor el funcionamiento de los pipelines de segmentación y las métricas utilizadas. Se recomienda revisar estos notebooks para obtener una comprensión más profunda del proyecto, aunque no son necesarios para la ejecución del flujo de trabajo completo.
