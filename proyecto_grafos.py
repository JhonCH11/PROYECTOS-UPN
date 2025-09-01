import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
from sklearn.cluster import KMeans

# 1. Configuración inicial
centro = np.array([-14.814404703595391, -71.32839434748219])  # Coordenada central
num_familias = 284
radio_dispersion = 0.02  # ~2.2 km alrededor del centro (en grados decimales)

# 2. Generar coordenadas realistas para las familias
np.random.seed(42)
desviaciones = np.random.normal(loc=0, scale=radio_dispersion, size=(num_familias, 2))
coordenadas = centro + desviaciones

# 3. Crear grafo geográfico
G = nx.Graph()

# Añadir nodos (0 = central, 1-284 = familias)
G.add_node(0, pos=tuple(centro), tipo='central')
for i in range(1, num_familias + 1):
    G.add_node(i, pos=tuple(coordenadas[i-1]), tipo='familia')

# 4. Calcular distancias reales en kilómetros
for i in range(num_familias + 1):
    for j in range(i + 1, num_familias + 1):
        lat1, lon1 = G.nodes[i]['pos']
        lat2, lon2 = G.nodes[j]['pos']
        
        # Conversión a kilómetros (1° ≈ 111 km)
        dx = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))  # Ajuste por latitud
        dy = (lat2 - lat1) * 111
        dist_km = np.hypot(dx, dy)
        
        G.add_edge(i, j, weight=dist_km)

# 5. Calcular el Árbol de Expansión Mínima (MST)
mst = nx.minimum_spanning_tree(G)

# 6. Optimización adicional con clusters
kmeans = KMeans(n_clusters=15)
coordenadas_todas = np.vstack([centro, coordenadas])
clusters = kmeans.fit_predict(coordenadas_todas)

# 7. Visualización profesional
plt.figure(figsize=(16, 12))

# Dibujar el terreno base
ax = plt.gca()
ax.set_facecolor('#f0f0f0')  # Color de fondo tipo mapa

# Dibujar todas las familias
familias_x = [G.nodes[i]['pos'][1] for i in range(1, num_familias + 1)]
familias_y = [G.nodes[i]['pos'][0] for i in range(1, num_familias + 1)]
plt.scatter(familias_x, familias_y, c=clusters[1:], cmap='tab20', s=50, alpha=0.8, edgecolors='k', linewidths=0.5)

# Dibujar la central eléctrica
plt.scatter(centro[1], centro[0], s=300, c='gold', edgecolors='black', marker='*', linewidths=1.5, label='Central Eléctrica')

# Dibujar las conexiones del MST
for edge in mst.edges():
    punto1 = G.nodes[edge[0]]['pos']
    punto2 = G.nodes[edge[1]]['pos']
    plt.plot([punto1[1], punto2[1]], [punto1[0], punto2[0]], 
             color='#ff4444', linewidth=1.2, alpha=0.7)

# Añadir elementos de mapa
plt.colorbar(label='Clusters Geográficos', shrink=0.8)
plt.title('Red Eléctrica Óptima para Huancané Bajo\n' 
          f'Longitud total: {mst.size(weight="weight"):.1f} km | '
          f'Costo estimado: ${mst.size(weight="weight")*1000*25:,.0f} USD', 
          fontsize=14, pad=20)

plt.xlabel('Longitud Oeste', fontsize=12)
plt.ylabel('Latitud Sur', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Añadir escala gráfica
x_range = max(familias_x) - min(familias_x)
plt.plot([min(familias_x)+0.1*x_range, min(familias_x)+0.1*x_range+0.01], 
         [min(familias_y)+0.05, min(familias_y)+0.05], 
         color='black', linewidth=2)
plt.text(min(familias_x)+0.1*x_range+0.005, min(familias_y)+0.06, 
         '1 km', ha='center', fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()

# 8. Métricas detalladas
print(f'''
=== Métricas Clave ===
- Familias atendidas: {num_familias}
- Longitud total de cable: {mst.size(weight="weight"):.1f} km
- Costo estimado del cableado (USD $25/m): ${mst.size(weight="weight")*1000*25:,.0f}
- Clusters geográficos: 15
- Distancia máxima entre nodos: {max(dict(mst.degree(weight='weight')).values()):.1f} km
''')
