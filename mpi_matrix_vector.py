import sys
import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 2:
    if rank == 0:
        print("Uso: mpirun -np <N> python mpi_matrix_vector.py <tamanho_matriz>", file=sys.stderr)
    sys.exit(-1)

MATRIX_SIZE = int(sys.argv[1])

matrix_A = None
vector_x = None

if rank == 0:
    matrix_A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype('float64')
    vector_x = np.random.rand(MATRIX_SIZE).astype('float64')
    
    # Validação crucial: Garante que a matriz pode ser dividida igualmente
    if MATRIX_SIZE % size != 0:
        print(f"Erro: O tamanho da matriz ({MATRIX_SIZE}) deve ser divisível pelo número de processos ({size}).", file=sys.stderr)
        comm.Abort() # Termina todos os processos MPI

    start_time = time.time()
else:
    start_time = None

if rank != 0:
    vector_x = np.empty(MATRIX_SIZE, dtype='float64')
comm.Bcast(vector_x, root=0)

rows_per_process = MATRIX_SIZE // size
local_matrix_rows = np.empty((rows_per_process, MATRIX_SIZE), dtype='float64')
comm.Scatter(matrix_A, local_matrix_rows, root=0)

local_result = np.dot(local_matrix_rows, vector_x)

full_result_vector = None
if rank == 0:
    full_result_vector = np.empty(MATRIX_SIZE, dtype='float64')

comm.Gather(local_result, full_result_vector, root=0)

if rank == 0:
    end_time = time.time()
    # A única linha de saída importante para o script .sh
    print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
