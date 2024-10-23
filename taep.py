import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)

# Definição de constantes
E, A = 290e9, 12700e-6  # Módulo de Elasticidade e Área das Barras
n, m = 7, 11  # Número de pontos e elementos

# Comprimentos das barras em um array
L = np.array([1, 2, np.sqrt(2), np.sqrt(2), 1, 1, np.sqrt(2), np.sqrt(5), np.sqrt(2), 3, 1])

# Vetores de forças e deslocamentos
forcas = np.zeros((n * 2, 1))
deslocamentos = np.zeros_like(forcas)

# Conexões entre os nós para cada elemento
elementos = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 2, 3],
                      [6, 7, 8, 9], [8, 9, 10, 11], [8, 9, 0, 1], [10, 11, 0, 1],
                      [10, 11, 12, 13], [12, 13, 0, 1], [8, 9, 2, 3]])

# Ângulos dos elementos em radianos
teta = np.deg2rad([0, 0, 225, 135, 180, 270, 135, 116.57, 225, 90, 90])

# Função para calcular a matriz de rigidez de um elemento
def matriz_rigidez(E, A, L, theta):
    c, s = np.cos(theta), np.sin(theta)
    k = (E * A) / L
    return k * np.array([[c**2, c*s, -c**2, -c*s],
                         [c*s, s**2, -c*s, -s**2],
                         [-c**2, -c*s, c**2, c*s],
                         [-c*s, -s**2, c*s, s**2]])

# Matrizes de rigidez locais para os elementos
matrizes_rigidez = [matriz_rigidez(E, A, L[i], teta[i]) for i in range(m)]

# Montagem da matriz global de rigidez
K = np.zeros((n * 2, n * 2))
for idx, elem in enumerate(elementos):
    K[np.ix_(elem, elem)] += matrizes_rigidez[idx]

# Definindo forças e deslocamentos fixos
forcas[5, 0] = -200000  # Força em C (d6)
deslocamentos[[0, 1, 12], 0] = 0  # Deslocamentos fixos: d1, d2, d13

# Redução das matrizes de rigidez e do vetor de forças
indices_fixos = [0, 1, 12]
K_reduzido = np.delete(np.delete(K, indices_fixos, axis=0), indices_fixos, axis=1)
F_reduzido = np.delete(forcas, indices_fixos, axis=0)

# Resolução dos deslocamentos
deslocamentos_reduzidos = np.linalg.solve(K_reduzido, F_reduzido)

# Reconstrução do vetor completo de deslocamentos
deslocamentos_completos = np.zeros((n * 2, 1))
deslocamentos_completos[np.setdiff1d(range(n * 2), indices_fixos)] = deslocamentos_reduzidos

# Separação dos deslocamentos em x e y
x_d, y_d = [deslocamentos_completos[i::2, 0] for i in range(2)]

# Cálculo das tensões em cada elemento
tensoes = np.array([(E/L[i]) * ((x_d[elementos[i][2] // 2] - x_d[elementos[i][0] // 2]) * np.cos(teta[i])
             + (y_d[elementos[i][3] // 2] - y_d[elementos[i][1] // 2]) * np.sin(teta[i])) / 1E6 for i in range(m)])

# Impressão dos resultados
print("Deslocamentos dos Nós (em metros):")
print(" Deslocamento em X | Deslocamento em Y")
print("--- | ----------------- | -----------------")
for i, (x, y) in enumerate(zip(x_d, y_d)):
    print(f"{x:.7f}           | {y:.7f}")

print("\nTensões:")
for i, t in enumerate(tensoes):
    tipo = "Compressão" if t < 0 else "Tração"
    print(f"{t:.2f} MPa ({tipo})")
import matplotlib.pyplot as plt

# Coordenadas dos nós
nodes = {
    'A': (0, 3),
    'B': (1, 3),
    'C': (3, 3),
    'D': (2, 2),
    'E': (1, 1),
    'F': (0, 0),
    'G': (1, 2)
}

# Elementos da treliça (conectando os nós)
elements = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), 
    ('A', 'E'), ('A', 'F'), ('E', 'F'), 
    ('B', 'E'), ('B', 'G'), ('E', 'G'), 
    ('D', 'G'), ('G', 'A'), ('D', 'B')
]

# Plotar a treliça original
fig, ax = plt.subplots()
for node1, node2 in elements:
    x_values = [nodes[node1][0], nodes[node2][0]]
    y_values = [nodes[node1][1], nodes[node2][1]]
    ax.plot(x_values, y_values, color='lightblue', marker='o', linewidth=2)  # Cor alterada para azul claro

# Rotular os nós
for node, (x, y) in nodes.items():
    ax.text(x, y, f' {node}', color='darkblue', fontsize=14, verticalalignment='bottom')  # Cor alterada

# Fazendo tudo novamente para a série deformada
deform_factor = 20  # Novo valor de deformação
nodes_deformed = {
    node + '.': (x + deform_factor * x_d[i], y + deform_factor * y_d[i])
    for i, (node, (x, y)) in enumerate(nodes.items())
}

elements_deformed = [
    (node1 + '.', node2 + '.') for node1, node2 in elements
]

# Plotar a treliça deformada
for node1_deformed, node2_deformed in elements_deformed:
    x_values_deformed = [nodes_deformed[node1_deformed][0], nodes_deformed[node2_deformed][0]]
    y_values_deformed = [nodes_deformed[node1_deformed][1], nodes_deformed[node2_deformed][1]]
    ax.plot(x_values_deformed, y_values_deformed, color='salmon', marker='s', linewidth=2)  # Cor alterada para salmão

# Rotular os nós deformados
for node, (x, y) in nodes_deformed.items():
    ax.text(x, y, f' {node}', color='darkred', fontsize=14, verticalalignment='top')  # Cor alterada

# Personalizar o gráfico
ax.set_title('Treliça Original e Deformada', fontsize=16)  # Título do gráfico
ax.set_xlabel('Coordenadas X', fontsize=14)  # Rótulo do eixo X
ax.set_ylabel('Coordenadas Y', fontsize=14)  # Rótulo do eixo Y
ax.grid()

# Legenda simplificada
ax.legend(['Original', 'Deformada'], loc='upper right', fontsize=12)  # Adiciona legenda


# Legenda com cores correspondentes
handles = [
    plt.Line2D([0], [0], color='lightblue', lw=2, label='Original'),
    plt.Line2D([0], [0], color='salmon', lw=2, label='Deformada')
]
ax.legend(handles=handles, loc='upper right', fontsize=12)  # Adiciona legenda

plt.show()