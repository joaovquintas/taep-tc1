# Análise de Treliças em Python

## Descrição

Este programa realiza a análise estrutural de uma treliça utilizando o método dos elementos finitos. Ele calcula os deslocamentos dos nós e as tensões em cada elemento da treliça sob a ação de forças aplicadas. O resultado é apresentado tanto numericamente quanto graficamente, mostrando a configuração original e deformada da treliça.

## Funcionalidades

- **Cálculo da Matriz de Rigidez**: O programa calcula a matriz de rigidez para cada elemento da treliça com base em suas propriedades (módulo de elasticidade, área da seção transversal e comprimento).
- **Montagem da Matriz Global de Rigidez**: As matrizes de rigidez locais são montadas em uma matriz global que representa toda a estrutura.
- **Resolução de Deslocamentos**: Deslocamentos nos nós são calculados resolvendo o sistema de equações resultante da matriz de rigidez e das forças aplicadas.
- **Cálculo de Tensões**: As tensões em cada elemento da treliça são calculadas e categorizadas como compressão ou tração.
- **Visualização Gráfica**: O programa gera um gráfico que mostra a treliça original e a sua configuração deformada, permitindo a visualização dos efeitos das forças aplicadas.

## Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

- `numpy`
- `pandas`
- `matplotlib`

Você pode instalá-las usando o seguinte comando:

```bash
pip install numpy pandas matplotlib
