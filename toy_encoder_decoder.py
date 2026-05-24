import math
import random

random.seed(42)

# --- 1) Beispiel-Trainingsdaten -------------------------------------------
X = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
]

# --- 2) Mathe-Helfer -------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def tanh(z):
    return math.tanh(z)

def zeros(r, c):
    return [[0.0 for _ in range(c)] for _ in range(r)]

def randn(r, c, scale=0.2):
    return [[random.gauss(0, scale) for _ in range(c)] for _ in range(r)]

def matmul(A, B):
    # A: (m x n), B: (n x p) => (m x p)
    m, n, p = len(A), len(A[0]), len(B[0])
    out = zeros(m, p)
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p):
                out[i][j] += aik * B[k][j]
    return out

def add_bias(A, b):
    return [[A[i][j] + b[0][j] for j in range(len(A[0]))] for i in range(len(A))]

def apply_fn(A, fn):
    return [[fn(v) for v in row] for row in A]

def transpose(A):
    return [list(col) for col in zip(*A)]

def sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def hadamard(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_mul(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

def sum_rows(A):
    cols = len(A[0])
    s = [0.0] * cols
    for row in A:
        for j in range(cols):
            s[j] += row[j]
    return [s]

# --- 3) Modell: Encoder + Decoder -----------------------------------------
# Encoder: 6 -> 2, Decoder: 2 -> 6
W_enc = randn(6, 2)
b_enc = zeros(1, 2)
W_dec = randn(2, 6)
b_dec = zeros(1, 6)

lr = 0.3
epochs = 3000
n = len(X)

# --- 4) Training -----------------------------------------------------------
for epoch in range(1, epochs + 1):
    # Vorwärts
    Z = add_bias(matmul(X, W_enc), b_enc)
    H = apply_fn(Z, tanh)
    O = add_bias(matmul(H, W_dec), b_dec)
    Y = apply_fn(O, sigmoid)

    # Loss (Binary Cross Entropy)
    eps = 1e-9
    loss = 0.0
    for i in range(n):
        for j in range(6):
            xij = X[i][j]
            yij = Y[i][j]
            loss += -(xij * math.log(yij + eps) + (1 - xij) * math.log(1 - yij + eps))
    loss /= n * 6

    # Rückwärts
    dO = zeros(n, 6)
    for i in range(n):
        for j in range(6):
            dO[i][j] = (Y[i][j] - X[i][j]) / n

    dW_dec = matmul(transpose(H), dO)
    db_dec = sum_rows(dO)

    dH = matmul(dO, transpose(W_dec))

    tanh_grad = zeros(n, 2)
    for i in range(n):
        for j in range(2):
            tanh_grad[i][j] = 1 - H[i][j] * H[i][j]
    dZ = hadamard(dH, tanh_grad)

    dW_enc = matmul(transpose(X), dZ)
    db_enc = sum_rows(dZ)

    # Update
    W_dec = sub(W_dec, scalar_mul(dW_dec, lr))
    b_dec = sub(b_dec, scalar_mul(db_dec, lr))
    W_enc = sub(W_enc, scalar_mul(dW_enc, lr))
    b_enc = sub(b_enc, scalar_mul(db_enc, lr))

    if epoch % 500 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# --- 5) Ausgabe ------------------------------------------------------------
print("\nRekonstruktion nach Training (Schwellwert 0.5):")
Y_bin = [[1 if v > 0.5 else 0 for v in row] for row in Y]

for i in range(n):
    original = "".join(str(v) for v in X[i])
    recon = "".join(str(v) for v in Y_bin[i])
    latent = [round(v, 3) for v in H[i]]
    print(f"Input: {original} -> Output: {recon} | Latent: {latent}")

print("\nRohwerte (erste 2 Beispiele):")
for i in range(2):
    raw = [round(v, 3) for v in Y[i]]
    print(f"Input {i}:  {X[i]}")
    print(f"Output {i}: {raw}")
