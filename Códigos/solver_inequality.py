# Biblioteca
import numpy as np

##################### Parâmetros #######################

# Fator de desconto
v = 1/(1+0.05)

# Numéro de tentativas
n = 100000

# Seed
np.random.seed(79)

########################################################

#################################### Soluções ##########################################

# Lista para armazenar resultados
solucoes = []

# Loop
for _ in range(n):
    
    # Gera numeros aleatórios entre 0 e 1
    a0, a1, a2, b0, b1, b2 = np.random.rand(6)
    
    # Primeria condição
    cond1 = (a0 + a1 + a2) >= (b0 + b1 + b2)
    
    # Segunda condição
    cond2 = ((a0) + (pow(v, 1) * a1) + (pow(v, 2) * a2)) <= ((b0) + (pow(v, 1) * b1) + (pow(v, 2) * b2))
    
    # Condições para ter probabilidades mais reais
    cond3 = a0 > a1 > a2
    cond4 = b0 > b1 > b2
    
    # Caso as condições sejam verdadeiras
    if cond1 and cond2 and cond3 and cond4:

        # Adiciona os valores na lista de armazenamento
        solucoes.append((a0, a1, a2, b0, b1, b2))
#########################################################################################

# Caso não tenha encontrado soluções
if not solucoes:
    print("Vazio")

########################### Visualizar a primeira solução ###############################
else:
    # Valores
    print("-" * 30)
    a0, a1, a2, b0, b1, b2 = solucoes[0]
    print(f"a0 = {a0:.4f}, a1 = {a1:.4f}, a2 = {a2:.4f}, b0 = {b0:.4f}, b1 = {b1:.4f}, b2 = {b2:.4f}\n")

    # Primeira condição
    print(f"Condição 1: {a0:.4f} + {a1:.4f} + {a2:.4f} ≥ {b0:.4f} + {b1:.4f} + {b2:.4f}")
    print(f"Condição 1: {a0+a1+a2:.4f} ≥ {b0+b1+b2:.4f}\n")

    # Segunda condição
    print(f"Condição 2: ({a0:.4f}) + ({pow(v, 1):.4f} * {a1:.4f}) + ({pow(v, 2):.4f} * {a2:.4f}) ≤ ({b0:.4f}) + ({pow(v, 1):.4f} * {b1:.4f}) + ({pow(v, 2):.4f} * {b2:.4f})")
    print(f"Condição 2: {(a0) + (pow(v, 1) * a1) + (pow(v, 2) * a2):.4f} ≤ {(b0) + (pow(v, 1) * b1) + (pow(v, 2) * b2):.4f}")
    print("-" * 30)
#########################################################################################