import numpy as np

def qr_householder_real(A):
    
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    
    for i in range(0, n - 1):

        alpha = np.linalg.norm(R[i:, i], 2)
        
        e1 = np.zeros_like(R[i:, [i]])
        e1[0] = 1
        
        u = R[i:, [i]] - alpha*e1
        v = u / np.linalg.norm(u, 2)
        
        Qn = np.identity(m - i) - (2 * np.dot(v, v.T))
        
        Qn = np.block([
            [np.eye(i), np.zeros((i, m - i))],
            [np.zeros((m - i, i)), Qn]
        ])
        
        R = np.dot(Qn, R)
        Q = np.dot(Q, Qn.T)
        
    return Q, R


def qr_householder_complex(A):
    
    m, n = A.shape
    R = A.astype(np.complex128).copy()
    Q = np.eye(m, dtype=np.complex128)
    
    for i in range(0, n - 1):

        if R[i:, i][0] != 0:
            alpha = -(R[i:, i][0])/(np.abs(R[i:, i][0]))*np.linalg.norm(R[i:, i], 2)
        else: 
            alpha = np.linalg.norm(R[i:, i], 2)

        e1 = np.zeros_like(R[i:, [i]])
        e1[0] = 1
        
        u = R[i:, [i]] - alpha*e1
        v = u / np.linalg.norm(u, 2)
        
        Qn = np.identity(m - i) - (2 * np.dot(v, v.conj().T))
        
        Qn = np.block([
            [np.eye(i),           np.zeros((i, m - i))],
            [np.zeros((m - i, i)), Qn]
        ])
        
        R = np.dot(Qn, R)
        Q = np.dot(Q, Qn.conj().T)
        
    return Q, R


A1 = np.array([[1+1j, -1, 0],
               [1, 0, 0],
               [0, 0, 1]]) @ np.array([[1, 1, 1],
                                       [0, 1, 1],
                                       [0, 0, -1]]) # Q @ R 

print(f'A1 = {A1}')

Q, R = qr_householder_complex(A1)

print(f'Q = {Q}', f'R = {R}', sep='\n')

print(f'A1 = {Q @ R}')
