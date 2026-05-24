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

def print_latent_axis(samples, latents, width=61):
    min_v = min(latents)
    max_v = max(latents)
    span = max(max_v - min_v, 1e-9)

    axis = ["-"] * width
    zero_pos = int(round((0 - min_v) / span * (width - 1)))
    zero_pos = max(0, min(width - 1, zero_pos))
    axis[zero_pos] = "|"

    print("\nLatent Space (1D Koordinatensystem):")
    print(f"min={min_v:.3f} 0={0.0:.3f} max={max_v:.3f}")
    print("".join(axis))

    for i, (sample, z) in enumerate(zip(samples, latents)):
        pos = int(round((z - min_v) / span * (width - 1)))
        pos = max(0, min(width - 1, pos))
        line = [" "] * width
        line[pos] = "x"
        sample_bits = "".join(str(v) for v in sample)
        print(f"{i:02d} {''.join(line)} z={z:+.3f} sample={sample_bits}")

# --- 3) Modell: Encoder + Decoder -----------------------------------------
# Encoder: 6 -> 1, Decoder: 1 -> 6
W_enc = randn(6, 1)
b_enc = zeros(1, 1)
W_dec = randn(1, 6)
b_dec = zeros(1, 6)

lr = 0.3
epochs = 3000
n = len(X)

# --- 4) Training -----------------------------------------------------------
for epoch in range(1, epochs + 1):
    Z = add_bias(matmul(X, W_enc), b_enc)
    H = apply_fn(Z, tanh)
    O = add_bias(matmul(H, W_dec), b_dec)
    Y = apply_fn(O, sigmoid)

    eps = 1e-9
    loss = 0.0
    for i in range(n):
        for j in range(6):
            xij = X[i][j]
            yij = Y[i][j]
            loss += -(xij * math.log(yij + eps) + (1 - xij) * math.log(1 - yij + eps))
    loss /= n * 6

    dO = zeros(n, 6)
    for i in range(n):
        for j in range(6):
            dO[i][j] = (Y[i][j] - X[i][j]) / n

    dW_dec = matmul(transpose(H), dO)
    db_dec = sum_rows(dO)

    dH = matmul(dO, transpose(W_dec))

    tanh_grad = zeros(n, 1)
    for i in range(n):
        tanh_grad[i][0] = 1 - H[i][0] * H[i][0]
    dZ = hadamard(dH, tanh_grad)

    dW_enc = matmul(transpose(X), dZ)
    db_enc = sum_rows(dZ)

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
    latent = round(H[i][0], 3)
    print(f"Input: {original} -> Output: {recon} | Latent: {latent:+.3f}")

print_latent_axis(X, [h[0] for h in H])

print("\nRohwerte (erste 2 Beispiele):")
for i in range(2):
    raw = [round(v, 3) for v in Y[i]]
    print(f"Input {i}:  {X[i]}")
    print(f"Output {i}: {raw}")
