import numpy as np

def tridag(IF, L, A, B, C, D, V):
    BETA = np.zeros(101)
    GAMMA = np.zeros(101)
    BETA[IF] = B[IF]
    GAMMA[IF] = D[IF] / BETA[IF]
    IFP1 = IF + 1
    for I in range(IFP1, L + 1):
        BETA[I] = B[I] - A[I] * C[I - 1] / BETA[I - 1]
        GAMMA[I] = (D[I] - A[I] * GAMMA[I - 1]) / BETA[I]
    V[L] = GAMMA[L]
    LAST = L - IF
    for K in range(1, LAST + 1):
        I = L - K
        V[I] = GAMMA[I] - C[I] * V[I + 1] / BETA[I]

V = np.zeros((26, 101))
VNEW = np.zeros((26, 101))
T = np.zeros((26, 101))
TNEW = np.zeros((26, 101))
DTY = np.zeros(1081)
DTYBT = np.zeros(101)
XNU = np.zeros(101)
SKF = np.zeros(101)
A = np.zeros(101)
B = np.zeros(101)
C = np.zeros(101)
D = np.zeros(101)
E = np.zeros(101)
U = np.zeros((26, 101))
SKI = np.zeros(101)
C1 = np.zeros((26, 101))
CNEW = np.zeros((26, 101))
UNEW = np.zeros((26, 101))
DCY = np.zeros(101)
XSH = np.zeros(101)
DCYBT = np.zeros(101)
W = np.zeros((26, 101))
WNEW = np.zeros((26, 101))

EPS = 0.00001
Q = 2.0
M = 20
N = 56
RT = 0.2
R = 2
XMAX = 1
YMAX = 14
DTAU = 0.01
DN = 0.5
DM = 0.5
TAUMAX = 20
GR = 5
PR = 0.71
GM = 5.0
SC = 0.3
IFREQ = 10
MP1 = M + 1
NP1 = N + 1
FLOATM = M
FLOATN = N
DX = XMAX / FLOATM
DY = YMAX / FLOATN
DYSQ = DY * DY
DYSQPR = DYSQ * PR
DYSQSC = DYSQ * SC

print(f"M={M:2d}           N={N:3d}")
print(f"XMAX={XMAX:.2f}    YMAX={YMAX:.2f}")
print(f"DX={DX:.3f}        DY={DY:.3f}        DN={DN:.1f}        Q={Q:.3f}")
print(f"DM={DM:.1f}        PR={PR:.2f}        GR={GR:.2f}")
print(f"GM={GM:.2f}        SC={SC:.2f}        R={R:.2f}")
print(f"DTAU={DTAU:.3f}    TAUMAX={TAUMAX:.2f}    RT={RT:.1f}")

ICOUNT = 0
JCOUNT = 0

U.fill(0.0)
V.fill(0.0)
T.fill(0.0)
W.fill(0.0)
C1.fill(0.0)
CNEW.fill(0.0)
UNEW.fill(0.0)
VNEW.fill(0.0)
TNEW.fill(0.0)
WNEW.fill(0.0)

X = 0.0
for I in range(1, MP1):
    X += DX
    U[I, 0] = 1.0
    UNEW[I, 0] = 1.0
    W[I, 0] = 0.0
    WNEW[I, 0] = 0.0
    V[I, 0] = 0.0
    VNEW[I, 0] = 0.0
    T[I, 0] = 1.0
    TNEW[I, 0] = 1.0

TAU = 0.0
while TAU < TAUMAX:
    TAU += DTAU
    ICOUNT += 1
    X = 0.0
    for I in range(1, MP1):
        X += DX
        A[0] = 0.0
        B[0] = 1.0 + 0.5 * DTAU / DX * U[I, 0] + DTAU / DYSQSC
        C[0] = -DTAU / DYSQSC
        D[0] = C1[I, 0] + U[I, 0] * 0.5 * DTAU / DX * (CNEW[I - 1, 0] - C1[I, 0] + C1[I - 1, 0]) + DTAU * (C1[I, 1] - C1[I, 0] + 2 * DY) / DYSQSC
        for J in range(1, NP1):
            A[J] = -0.25 * DTAU / DY * V[I, J] - 0.5 * DTAU / DYSQSC
            B[J] = 1.0 + 0.5 * DTAU / DX * U[I, J] + DTAU / DYSQSC
            C[J] = 0.25 * DTAU / DY * V[I, J] - 0.5 * DTAU / DYSQSC
            D[J] = C1[I, J] + 0.5 * DTAU / DX * U[I, J] * (CNEW[I - 1, J] - C1[I, J] + C1[I - 1, J]) + 0.25 * DTAU / DY * V[I, J] * (C1[I, J - 1] - C1[I, J + 1]) + 0.5 * DTAU / DYSQSC * (C1[I, J - 1] - 2.0 * C1[I, J] + C1[I, J + 1])
        D[1] -= A[1] * TNEW[I, 0]
        tridag(1, N, A, B, C, D, E)
        for J in range(1, N + 1):
            CNEW[I, J] = E[J]

    for I in range(2, MP1):
        for J in range(2, N + 1):
            A[J] = -0.25 * DTAU / DY * V[I, J] - 0.5 * DTAU / DYSQPR
            B[J] = 1.0 + 0.5 * DTAU / DX * U[I, J] + DTAU / DYSQPR + (0.5 * R * DTAU / PR)
            C[J] = 0.25 * DTAU / DY * V[I, J] - 0.5 * DTAU / DYSQPR
            D[J] = (1.0 - (0.5 * R * DTAU / PR)) * T[I, J] + 0.5 / DX * DTAU * U[I, J] * (TNEW[I - 1, J] - T[I, J] + T[I - 1, J]) + 0.25 * DTAU / DY * V[I, J] * (T[I, J - 1] - T[I, J + 1]) + 0.5 * DTAU / DYSQPR * (T[I, J - 1] - 2.0 * T[I, J] + T[I, J + 1])
        D[2] -= A[2] * TNEW[I, 0]
        tridag(2, N, A, B, C, D, E)
        for J in range(2, N + 1):
            TNEW[I, J] = E[J]

    for I in range(1, MP1):
        for J in range(1, NP1):
            A[J] = (-0.25 * V[I, J] * DTAU / DY) - (0.5 * DTAU / DYSQ)
            B[J] = 1.0 + (0.5 * DTAU * U[I, J] / DX) + (DTAU / DYSQ)
            C[J] = (0.25 * DTAU * V[I, J] / DY) - (0.5 * DTAU / DYSQ)
            D[J] = W[I, J] + (0.5 * DTAU * U[I, J] / DX) * (WNEW[I - 1, J] - W[I, J] + W[I - 1, J]) + (0.25 * DTAU * V[I, J] / DY) * (W[I, J - 1] - W[I, J + 1]) + (0.5 * DTAU / DYSQ) * (W[I, J - 1] - 2.0 * W[I, J] + W[I, J + 1]) - (RT * U[I, J] * DTAU) - (Q * W[I, J] * DTAU)
        tridag(2, N, A, B, C, D, E)
        for J in range(2, N + 1):
            WNEW[I, J] = E[J]

    for I in range(2, MP1):
        for J in range(2, NP1):
            A[J] = -(0.25 * DTAU / DY * V[I, J] + 0.5 * DTAU / DYSQ)
            B[J] = 1.0 + 0.5 * DTAU / DX * U[I, J] + DTAU / DYSQ
            C[J] = 0.25 * DTAU / DY * V[I, J] - 0.5 * DTAU / DYSQ
            D[J] = U[I, J] + (0.5 * DTAU / DX * U[I, J]) * (UNEW[I - 1, J] - U[I, J] + U[I - 1, J]) + (0.25 * DTAU / DY * V[I, J] * (U[I, J - 1] - U[I, J + 1])) + (0.5 * DTAU / DYSQ * (U[I, J - 1] - 2.0 * U[I, J] + U[I, J + 1])) + DTAU * 0.5 * ((GR * (TNEW[I, J] + T[I, J])) + (GM * (C1[I, J] + CNEW[I, J]))) + DTAU * (RT * W[I, J] - Q * U[I, J])
        D[2] -= A[2] * UNEW[I, 0]
        tridag(2, N, A, B, C, D, E)
        for J in range(2, N + 1):
            UNEW[I, J] = E[J]

    for I in range(2, MP1):
        for J in range(2, NP1):
            VNEW[I, J] = VNEW[I, J - 1] - V[I, J] + V[I, J - 1] - 0.5 * DY * (UNEW[I, J] - UNEW[I - 1, J] + U[I, J] - U[I - 1, J] + UNEW[I, J - 1] - UNEW[I - 1, J - 1] + U[I, J - 1] - U[I - 1, J - 1]) / DX

    for I in range(1, MP1):
        for J in range(1, NP1):
            if (abs(UNEW[I, J] - U[I, J]) > EPS) or (abs(TNEW[I, J] - T[I, J]) > EPS) or (abs(WNEW[I, J] - W[I, J]) > EPS) or (abs(CNEW[I, J] - C1[I, J]) > EPS):
                break
        else:
            continue
        break
    else:
        JCOUNT = 1
        break

X = 0.0
for I in range(2, MP1):
    X += DX
    XSH[0] = 0.0
    DCY[I] = (-3.0 * C1[I, 5] + 16.0 * C1[I, 4] - 36.0 * C1[I, 3] + 48.0 * C1[I, 2] - 25.0 * C1[I, 1]) / (12.0 * DY)
    DCYBT[I] = DCY[I] / C1[I, 1]
    XSH[I] = -X * DCYBT[I]
    XNU[0] = 0.0
    DTY[I] = (-3.0 * T[I, 5] + 16.0 * T[I, 4] - 36.0 * T[I, 3] + 48.0 * T[I, 2] - 25.0 * T[I, 1]) / (12.0 * DY)
    DTYBT[I] = DTY[I] / T[I, 1]
    XNU[I] = -X * DTYBT[I]
    SKF[0] = 0.0
    SKF[I] = (-3.0 * U[I, 5] + 16.0 * U[I, 4] - 36.0 * U[I, 3] + 48.0 * U[I, 2] - 25.0 * U[I, 1]) / (12.0 * DY)
    SKI[I] = -SKF[I]

DTYBT[0] = 0.0
ANUI = 0.0
for I in range(1, M + 1):
    ANU = 2.0 * DX * (7.0 * DTYBT[I] + 32.0 * DTYBT[I + 1] + 12.0 * DTYBT[I + 2] + 32.0 * DTYBT[I + 3] + 7.0 * DTYBT[I + 4]) / 45.0
    ANUI += ANU
ANUL = -ANUI

ASHI = 0.0
for I in range(1, M + 1):
    ASH = 2.0 * DX * (7.0 * DCYBT[I] + 32.0 * DCYBT[I + 1] + 12.0 * DCYBT[I + 2] + 32.0 * DCYBT[I + 3] + 7.0 * DCYBT[I + 4]) / 45.0
    ASHI += ASH
ASHL = -ASHI

ASKF = 0.0
for I in range(1, M + 1):
    ASKFI = 2.0 * DX * (7.0 * SKF[I] + 32.0 * SKF[I + 1] + 12.0 * SKF[I + 2] + 32.0 * SKF[I + 3] + 7.0 * SKF[I + 4]) / 45.0
    ASKF += ASKFI
ASK1 = -ASKF

print(f"AT A TIME TAU={TAU:.3f}")
print("TEM T IS")
for K in range(1, MP1, 5):
    I = MP1 - K + 1
    print(T[I, :])
print("CONC. C IS")
for K in range(1, MP1, 5):
    I = MP1 - K + 1
    print(C1[I, :])
print("VEL W IS")
for K in range(1, MP1, 5):
    I = MP1 - K + 1
    print(W[I, :])
print("VEL U IS")
for K in range(1, MP1, 5):
    I = MP1 - K + 1
    print(U[I, :])
print("VEL V IS")
for K in range(1, MP1, 5):
    I = MP1 - K + 1
    print(V[I, ::5])
print(f"AVE SKIN FRICTION={ASK1:.5f}")
print(f"AVE NUSSLET NO={ANUL:.5f}")
print(f"AVE SHERWOOD NO={ASHL:.5f}")
print("LOCAL NUSSELET NO=", XNU[1:MP1])
print("LOCAL SKIN FRICTION=", SKI[1:MP1])
print("LOCAL SHEWOOD NO=", XSH[1:MP1])
