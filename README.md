# PROYECTOS-UPN

# Red Eléctrica Óptima – Python

## ¿Qué hace este proyecto?

Este proyecto simula la **conexión eléctrica óptima** entre una **central eléctrica** y **284 familias** usando Python.

El programa:

* Genera ubicaciones geográficas simuladas alrededor de una central
* Calcula la distancia entre todos los puntos
* Encuentra la forma más corta de conectar a todos usando un **Árbol de Expansión Mínima (MST)**
* Agrupa las familias en **clusters** para análisis geográfico
* Muestra un gráfico con la red resultante
* Calcula la longitud total del cable y su costo aproximado

---

## ¿Cómo funciona?

1. Se define una coordenada central
2. Se generan coordenadas aleatorias para las familias
3. Se crea un grafo donde los nodos son familias y la central
4. Las aristas representan distancias en kilómetros
5. Se calcula el MST para minimizar el cableado
6. Se visualiza la red y se muestran métricas básicas

---

## Librerías usadas

* numpy
* matplotlib
* networkx
* scikit-learn
* scipy

---

## Cómo ejecutar

```en la terminal añadir los siguientes comandos:
pip install numpy matplotlib networkx scikit-learn scipy
python main.py
```
o también puedes instalar las librerías de esta manera en el terminal: 

python -m pip install numpy matplotlib networkx scipy scikit-learn


---

## Resultado

* Un mapa con la red eléctrica optimizada
* Longitud total del cable (km)
* Costo estimado del cableado
* Información básica de la red
