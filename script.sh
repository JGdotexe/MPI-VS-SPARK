#!/bin/bash

# --- Configurações do Experimento no CLUSTER ---
NUM_RUNS=3
MPI_PROCESSES="1 2 4 8 16 32"
MATRIX_SIZE=8192 # Aumentamos o tamanho para o cluster
RESULTS_FILE="resultados_cluster.csv"
GRAPH_FILE="comparacao_desempenho_cluster.png"

# Garante que o script pare se algum comando falhar
set -e

echo "======================================================"
echo "INICIANDO EXPERIMENTO DE COMPARAÇÃO NO CLUSTER"
echo "======================================================"
echo "Número de execuções por teste: $NUM_RUNS"
echo "Tamanho da Matriz: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Processos MPI a serem testados: $MPI_PROCESSES"
echo "------------------------------------------------------"

# Ativa o ambiente virtual, se não estiver ativo
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Ativando ambiente virtual Python..."
    source .venv/bin/activate
fi

# Cria o cabeçalho do arquivo de resultados
echo "Framework,Processos,TempoExecucao_s" > $RESULTS_FILE

# --- Teste do Spark ---
# No cluster, o Spark pode ser configurado de forma diferente.
# Este comando assume uma execução local[*] usando todos os cores disponíveis.
cores_spark=$(nproc)
echo "Iniciando testes com Spark (usando $cores_spark cores)..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Execução Spark $i de $NUM_RUNS..."
    # Executa o script e extrai o tempo
    output=$(spark-submit spark_matrix_vector.py $MATRIX_SIZE 2>&1)
    time=$(echo "$output" | grep "Tempo de execução:" | awk '{print $3}')
    echo "Spark,$cores_spark,$time" >> $RESULTS_FILE
done
echo "Testes do Spark concluídos."
echo "------------------------------------------------------"


# --- Testes do MPI ---
echo "Iniciando testes com MPI..."
for p in $MPI_PROCESSES; do
    echo "  Testando com $p processos MPI..."
    for i in $(seq 1 $NUM_RUNS); do
        echo "    Execução MPI ($p processos) $i de $NUM_RUNS..."
        # Executa o script. NÃO usamos --oversubscribe no cluster.
        output=$(mpirun -np $p python mpi_matrix_vector.py $MATRIX_SIZE)
        time=$(echo "$output" | grep "Tempo de execução:" | awk '{print $3}')
        echo "MPI,$p,$time" >> $RESULTS_FILE
    done
done
echo "Testes do MPI concluídos."
echo "------------------------------------------------------"

# --- Geração do Gráfico ---
echo "Gerando o gráfico de comparação..."
python gerar_grafico.py $RESULTS_FILE $GRAPH_FILE

echo "======================================================"
echo "EXPERIMENTO CONCLUÍDO!"
echo "Resultados salvos em: $RESULTS_FILE"
echo "Gráfico salvo em: $GRAPH_FILE"
echo "======================================================"
