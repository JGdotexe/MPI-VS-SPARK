import sys
import pandas as pd
import matplotlib.pyplot as plt

def gerar_grafico(arquivo_resultados, arquivo_saida):
    """
    Lê um arquivo CSV com os resultados e gera um gráfico de comparação.
    """
    try:
        df = pd.read_csv(arquivo_resultados)
    except FileNotFoundError:
        print(f"Erro: Arquivo de resultados '{arquivo_resultados}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    # Calcula a média para cada grupo
    df_avg = df.groupby(['Framework', 'Processos'])['TempoExecucao_s'].mean().reset_index()

    # Separa os dados de MPI e Spark
    mpi_data = df_avg[df_avg['Framework'] == 'MPI'].sort_values('Processos')
    spark_data = df_avg[df_avg['Framework'] == 'Spark']

    # Pega o tempo médio do Spark
    spark_avg_time = spark_data['TempoExecucao_s'].iloc[0]
    spark_cores = spark_data['Processos'].iloc[0]

    # Cria o gráfico
    plt.figure(figsize=(12, 7))
    plt.plot(mpi_data['Processos'], mpi_data['TempoExecucao_s'], marker='o', linestyle='-', label='MPI (Média)')
    
    plt.axhline(y=spark_avg_time, color='r', linestyle='--', label=f'Spark ({spark_cores} cores) - Média: {spark_avg_time:.4f}s')

    # Melhora a aparência
    plt.title('Comparação de Desempenho: MPI vs. Spark\n(Multiplicação de Matriz no Cluster da Universidade)')
    plt.xlabel('Número de Processos')
    plt.ylabel('Tempo de Execução Médio (segundos)')
    plt.xticks(mpi_data['Processos'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.yscale('log') # Escala logarítmica é ótima para ver grandes diferenças
    
    # Salva o gráfico
    plt.savefig(arquivo_saida)
    print(f"Gráfico salvo com sucesso em '{arquivo_saida}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python gerar_grafico.py <arquivo_resultados.csv> <arquivo_saida.png>", file=sys.stderr)
        sys.exit(1)
    
    gerar_grafico(sys.argv[1], sys.argv[2])
