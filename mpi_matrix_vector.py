import time
import numpy as np
from mpi4py import MPI

# --- Inicialização do MPI ---
comm = MPI.COMM_WORLD  # O comunicador padrão, envolve todos os processos
rank = comm.Get_rank() # O ID deste processo (0 é o mestre)
size = comm.Get_size() # O número total de processos

# --- Definição do Problema ---
MATRIX_SIZE = 4000

# Variáveis que só o processo mestre precisa inicializar
matrix_A = None
vector_x = None

# ===============================================================
# ETAPA 1: O MESTRE (RANK 0) PREPARA E DISTRIBUI OS DADOS
# ===============================================================
if rank == 0:
    print(f"Executando com {size} processos MPI.")
    print(f"Iniciando multiplicação de matriz {MATRIX_SIZE}x{MATRIX_SIZE}...")
    
    # Cria a matriz e o vetor
    matrix_A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype('float64')
    vector_x = np.random.rand(MATRIX_SIZE).astype('float64')
    
    # Inicia a contagem do tempo AQUI, antes de qualquer comunicação
    start_time = time.time()
else:
    # Trabalhadores não precisam do tempo inicial
    start_time = None

# ===============================================================
# ETAPA 2: COMUNICAÇÃO COLETIVA
# ===============================================================

# 1. Broadcast do vetor: O mestre (rank 0) envia o vetor_x para TODOS os outros processos.
# Os trabalhadores precisam alocar espaço para receber o vetor.
if rank != 0:
    vector_x = np.empty(MATRIX_SIZE, dtype='float64')
comm.Bcast(vector_x, root=0)

# 2. Scatter das linhas da matriz: O mestre divide a matrix_A em pedaços
# e envia um pedaço para cada processo (incluindo ele mesmo).
rows_per_process = MATRIX_SIZE // size
local_matrix_rows = np.empty((rows_per_process, MATRIX_SIZE), dtype='float64')
comm.Scatter(matrix_A, local_matrix_rows, root=0)

# ===============================================================
# ETAPA 3: CÁLCULO LOCAL (TODOS OS PROCESSOS FAZEM ISSO)
# ===============================================================

# Cada processo multiplica seu pedaço da matriz pelo vetor completo
local_result = np.dot(local_matrix_rows, vector_x)

# ===============================================================
# ETAPA 4: O MESTRE REÚNE OS RESULTADOS
# ===============================================================

# Prepara um array no mestre para receber os resultados de todos
full_result_vector = None
if rank == 0:
    full_result_vector = np.empty(MATRIX_SIZE, dtype='float64')

# 3. Gather: Cada processo envia seu 'local_result' de volta para o mestre,
# que os organiza no 'full_result_vector'.
comm.Gather(local_result, full_result_vector, root=0)

# ===============================================================
# ETAPA 5: FINALIZAÇÃO (SÓ O MESTRE)
# ===============================================================
if rank == 0:
    end_time = time.time()
    print("Multiplicação concluída.")
    # print("Vetor resultante (primeiros 10 elementos):", full_result_vector[:10])
    print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
