import time
import numpy as np
from pyspark.sql import SparkSession

def matrix_vector_multiplication(spark, matrix, vector_broadcast):
    """
    Função para multiplicar uma matriz (distribuída como RDD) por um vetor.
    """
    # Paraleliza as linhas da matriz. Cada linha se torna um elemento do RDD.
    matrix_rdd = spark.sparkContext.parallelize(matrix)

    # Função que será aplicada a cada linha da matriz
    def multiply_row(row):
        # np.dot faz o produto escalar entre a linha e o vetor (que foi enviado a todos os nós)
        return np.dot(row, vector_broadcast.value)

    # Usa a transformação 'map' para aplicar a função a cada partição (linha) do RDD
    result_rdd = matrix_rdd.map(multiply_row)

    # A ação 'collect' traz o resultado de todos os workers para o driver (nó mestre)
    result_vector = result_rdd.collect()

    return result_vector

if __name__ == "__main__":
    # 1. Inicializa a Sessão Spark
    spark = SparkSession.builder.appName("MatrixVectorMultiplication").getOrCreate()

    # --- Definição do Problema ---
    MATRIX_SIZE = 4000 # Altere este valor para testar (1000, 2000, 4000, etc.)
    print(f"Iniciando multiplicação de matriz {MATRIX_SIZE}x{MATRIX_SIZE}...")

    # Cria uma matriz e um vetor com dados aleatórios
    # Em um cenário real, você leria isso de um arquivo
    matrix_A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    vector_x = np.random.rand(MATRIX_SIZE)

    # 2. Distribui o vetor para todos os nós
    # 'broadcast' é uma primitiva coletiva otimizada para enviar dados grandes a todos os workers.
    vector_x_broadcast = spark.sparkContext.broadcast(vector_x)

    # --- Execução e Medição de Tempo ---
    start_time = time.time()

    # 3. Executa a multiplicação
    result = matrix_vector_multiplication(spark, matrix_A, vector_x_broadcast)

    end_time = time.time()
    # --- Fim da Medição ---

    print(f"Multiplicação concluída.")
    # print("Vetor resultante (primeiros 10 elementos):", result[:10])
    print(f"Tempo de execução: {end_time - start_time:.4f} segundos")

    # 4. Encerra a Sessão Spark
    spark.stop()
