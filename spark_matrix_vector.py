import sys
import time
import numpy as np
from pyspark.sql import SparkSession

def matrix_vector_multiplication(spark, matrix, vector_broadcast):
    """
    Função para multiplicar uma matriz (distribuída como RDD) por um vetor.
    """
    matrix_rdd = spark.sparkContext.parallelize(matrix)
    def multiply_row(row):
        return np.dot(row, vector_broadcast.value)
    result_rdd = matrix_rdd.map(multiply_row)
    result_vector = result_rdd.collect()
    return result_vector

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: spark-submit spark_matrix_vector.py <tamanho_matriz>", file=sys.stderr)
        sys.exit(-1)

    MATRIX_SIZE = int(sys.argv[1])

    spark = SparkSession.builder.appName("MatrixVectorMultiplication").getOrCreate()
    
    # O print foi removido daqui para não poluir a saída que o .sh captura
    
    matrix_A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    vector_x = np.random.rand(MATRIX_SIZE)
    vector_x_broadcast = spark.sparkContext.broadcast(vector_x)

    start_time = time.time()
    result = matrix_vector_multiplication(spark, matrix_A, vector_x_broadcast)
    end_time = time.time()

    # A única linha de saída importante para o script .sh
    print(f"Tempo de execução: {end_time - start_time:.4f} segundos")

    spark.stop()
