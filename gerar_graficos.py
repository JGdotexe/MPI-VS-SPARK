import matplotlib.pyplot as plt

# 1. Preencha estes dados com os resultados da sua tabela!
processos = [1, 2, 4, 8]
tempos_mpi = [
    0.1431, # Substitua pelo seu tempo médio com 1 processo
    0.1136, # Substitua pelo seu tempo médio com 2 processos
    0.0874,# Substitua pelo seu tempo médio com 4 processos
    0.3757   # Substitua pelo seu tempo médio com 8 processos
]
tempo_spark = 2.18 # O tempo que você mediu para o Spark

# 2. Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(processos, tempos_mpi, marker='o', linestyle='-', label='MPI (mpi4py)')

# Adiciona uma linha horizontal para o Spark para comparação
plt.axhline(y=tempo_spark, color='r', linestyle='--', label=f'Spark (PySpark) - {tempo_spark:.2f}s')

# 3. Melhorar a aparência do gráfico
plt.title('Comparação de Desempenho: MPI vs. Spark\n(Multiplicação de Matriz 4000x4000 Localmente)')
plt.xlabel('Número de Processos')
plt.ylabel('Tempo de Execução (segundos)')
plt.xticks(processos) # Garante que o eixo X mostre 1, 2, 4, 8
plt.grid(True)
plt.legend()

# Inverte o eixo Y para que "melhor" (menos tempo) seja "mais alto" - opcional, mas legal
# plt.gca().invert_yaxis()

# 4. Salvar o gráfico em um arquivo
plt.savefig('comparacao_desempenho_local.png')
print("Gráfico 'comparacao_desempenho_local.png' foi salvo!")
