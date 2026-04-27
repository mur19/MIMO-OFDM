import numpy as np
import matplotlib.pyplot as plt
import commpy as cp
import time
import concurrent.futures

from commpy.channelcoding import Trellis, conv_encode, viterbi_decode

trellis = Trellis(
    memory=np.array([6]),
    g_matrix=np.array([[0o133, 0o171]])
)


class QAMmodulation:
    def __init__(self, M, SNR_dB, subcarriers, OFDM_symbols, flag_code):
        self.flag_code = flag_code
        self.flag = True
        self.OFDM_symbols = OFDM_symbols
        self.subcarriers = subcarriers
        self.M = M
        self.SNR_dB = SNR_dB
        self.SNR = 10 ** (self.SNR_dB / 10)
        self.num_symbols = subcarriers * OFDM_symbols * 2
        self.num_bits = int(self.num_symbols * np.log2(M))
        self.QAM_signal = None
        self.num_err_min = 100
        self.recursion_limit = 700


    def qr_householder(self, A):
    
        m, n = A.shape
        R = A.astype(np.complex128).copy()
        Q = np.eye(m, dtype=np.complex128)
        
        for i in range(0, min(m, n)):

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
                [np.eye(i), np.zeros((i, m - i))],
                [np.zeros((m - i, i)), Qn]
            ])

            R = np.dot(Qn, R)
            Q = np.dot(Q, Qn.conj().T)
            
        return Q, R

    def bit_input(self):
        
        self.bits_input = np.random.randint(0, 2, self.num_bits)
        # self.bits_input_code = np.array(dot11a_codec.encode(self.bits_input))
        self.bits_input_code = conv_encode(self.bits_input, trellis, termination='truncated')
        self.bits_input_code = self.bits_input_code[:-12]

        if self.flag_code:

            if self.flag:
                
                self.OFDM_symbols = len(self.bits_input_code) // len(self.bits_input) * self.OFDM_symbols
                self.num_symbols_code = len(self.bits_input_code) // len(self.bits_input) * self.num_symbols
                self.num_bits_code = len(self.bits_input_code) // len(self.bits_input) * self.num_bits

                self.flag = False

        # # print(f'bits_input = {self.bits_input}')
        # self.bits_input_1 = self.bits_input_code[:int(self.num_bits_code/2)]
        # # print(f'bits_input_1 = {self.bits_input_1}')
        # self.bits_input_2 = self.bits_input_code[int(self.num_bits_code/2):]

    def modul(self):

        self.modem = cp.QAMModem(self.M)

        if self.flag_code:

            self.x = self.modem.modulate(self.bits_input_code)
            self.QAM_signal = np.array(list(set(self.x)))

        else:

            self.x = self.modem.modulate(self.bits_input)
            self.QAM_signal = np.array(list(set(self.x)))

        self.x_1 = self.x[:int(len(self.x)/2)]

        self.x_2 = self.x[int(len(self.x)/2):]

    def OFDM(self):
        X_OFDM_1 = self.x_1.reshape(self.OFDM_symbols, self.subcarriers)
        X_OFDM_2 = self.x_2.reshape(self.OFDM_symbols, self.subcarriers)
        X_OFDM = np.empty((self.OFDM_symbols, self.subcarriers, 2), dtype=complex)
        for i in range(self.OFDM_symbols):
            for j in range(self.subcarriers):
                for k in range(2):
                    if k == 0:
                        X_OFDM[i, j, k] = X_OFDM_1[i, j]
                    elif k == 1:
                        X_OFDM[i, j, k] = X_OFDM_2[i, j]
        return X_OFDM
    
    def add_pilots(self, X_OFDM):

        ones_list = np.array([[1, 1]] * self.subcarriers)
        twoes_list = np.array([[1, 2]] * self.subcarriers) 

        ind_ones = [i + 1 for i in range(self.OFDM_symbols)]
        ind_twoes = [2*i + 2 for i in range(self.OFDM_symbols)]

        X_OFDM = np.insert(X_OFDM, ind_ones, ones_list, axis=0)
        X_OFDM = np.insert(X_OFDM, ind_twoes, twoes_list, axis=0)

        return X_OFDM 
    
    def idft(self, X_OFDM_pilots):
        for i in range(3*self.OFDM_symbols):
            for k in range(2):
                X_OFDM_pilots[i, :, k] = np.fft.ifft(X_OFDM_pilots[i, :, k])
        return X_OFDM_pilots
    
    def dft(self, X_OFDM_pilots):
        for i in range(3*self.OFDM_symbols):
            for k in range(2):
                X_OFDM_pilots[i, :, k] = np.fft.fft(X_OFDM_pilots[i, :, k])
        return X_OFDM_pilots

    def power_of_signal(self, y1):
        P_signal = 2/3 * (self.M - 1)
        # P_signal = np.mean(np.abs(self.x) ** 2)
        # P_signal = np.mean(np.abs(np.unique(self.x)) ** 2)
        return P_signal

    def RMSD(self, y1):  # root mean square deviation
        return np.sqrt(self.power_of_signal(y1) / (2 * self.SNR))

    def convolution_cp(self, r, h):
        s = []
        x_cp = r[-(len(h) - 1) :].tolist() + r.tolist()
        for k in range(self.subcarriers + len(h) - 1):
            y_k = 0
            m = 0
            while k >= m and m <= len(h) - 1:
                y_k += h[m] * x_cp[k - m]
                m += 1
            s.append(y_k)
        return np.array(s)

    def convolution_cp_MIMO(self, x_t_pilots):
        h = np.zeros((2, 2, len(self.impulse_response(diagram=False))), dtype=complex)
        y = np.zeros_like(x_t_pilots)
        for i in range(self.OFDM_symbols):
            for l in range(2):
                for m in range(2):
                    h[l, m] = self.impulse_response(diagram=False)
            for n in range(3):
                for k in range(2):
                    q = self.convolution_cp(x_t_pilots[3*i + n, :, 0], h[k, 0])[len(h[0, 0])-1:]
                    q_old = q
                    p = self.convolution_cp(x_t_pilots[3*i + n, :, 1], h[k, 1])[len(h[0, 0])-1:]
                    p_old = p
                    y[3*i + n, :, k] = q + p
        return y

    def AWGN(self, x_t_vector):
        sigma = self.RMSD(x_t_vector)
        re_n = np.array(np.random.normal(0, sigma, len(x_t_vector)))
        im_n = np.array(np.random.normal(0, sigma, len(x_t_vector)))
        awgn = re_n + im_n * 1j
        return awgn

    def power_of_noise(self):
        noise = self.AWGN()
        P_noise = np.mean(np.abs(noise) ** 2)
        return P_noise

    def zf_equalize(self, H, r):

        y_zf = np.zeros_like(r, dtype=H.dtype)
        
        for i in range(self.OFDM_symbols):
            for j in range(self.subcarriers):

                a = H[2*i, j, 0]      
                b = H[2*i, j, 1]      
                c = H[2*i + 1, j, 0]  
                d = H[2*i + 1, j, 1]  
                
                det = a * d - b * c
                
                y_zf[i, j, 0] = (d * r[i, j, 0] - b * r[i, j, 1]) / det
                y_zf[i, j, 1] = (-c * r[i, j, 0] + a * r[i, j, 1]) / det
        
        return y_zf

    def mmse_equalize(self, H, Y, mmse_solver):

        if mmse_solver == 'classic':

            y_mmse = np.zeros_like(Y, dtype=complex)
            
            for i in range(self.OFDM_symbols):
                for j in range(self.subcarriers):

                    H_mat = np.array([[H[2*i, j, 0], H[2*i, j, 1]],
                                    [H[2*i+1, j, 0], H[2*i+1, j, 1]]])
                    
                    Y_vec = np.array([Y[i, j, 0], Y[i, j, 1]])
                    
                    H_H = H_mat.conj().T
                    G = H_H @ H_mat + np.eye(2)/self.SNR

                    X_vec = np.linalg.solve(G, H_H @ Y_vec)
                    
                    y_mmse[i, j, 0] = X_vec[0]
                    y_mmse[i, j, 1] = X_vec[1]

        elif mmse_solver == 'householder':

            y_mmse = np.zeros_like(Y, dtype=complex)
            
            for i in range(self.OFDM_symbols):
                for j in range(self.subcarriers):

                    H_mat = np.array([[H[2*i, j, 0], H[2*i, j, 1]],
                                    [H[2*i+1, j, 0], H[2*i+1, j, 1]]])

                    I = np.eye(2, dtype='complex')

                    Y_vec = np.array([Y[i, j, 0], Y[i, j, 1]]).reshape(-1, 1)

                    H_house = np.vstack([H_mat, I/np.sqrt(self.SNR)], dtype='complex')

                    Y_house = np.vstack([Y_vec, np.zeros((2, 1))], dtype='complex')

                    Q, R = self.qr_householder(H_house)

                    z = np.conj(Q).T @ Y_house

                    R_up = R[:2, :]

                    z_1 = z[:2, :]

                    X_vec = np.linalg.solve(R_up, z_1)

                    y_mmse[i, j, 0] = X_vec[0, 0]
                    y_mmse[i, j, 1] = X_vec[1, 0]
        
        return y_mmse


    # def ml_equalize(self, H, Y):

    #     y_ml = np.zeros_like(Y, dtype=complex)

    #     x, y = np.meshgrid(self.QAM_signal, self.QAM_signal, indexing='ij')

    #     X = np.column_stack((x.ravel(), y.ravel()))

    #     y = np.zeros((2, self.M**2), dtype=complex)

    #     for i in range(self.OFDM_symbols):
    #         for j in range(self.subcarriers):
    #             moduls = np.zeros(self.M**2)
    #             for k in range(self.M**2):

    #                 H_mat = np.array([
    #             [H[2*i, j, 0], H[2*i, j, 1]],
    #             [H[2*i+1, j, 0], H[2*i+1, j, 1]]
    #         ])

    #                 y[:, k] = Y[i, j, :] - H_mat @ X[k]

    #                 moduls[k] = np.linalg.norm(y[:, k])

    #             idx = np.argmin(moduls)
    #             y_ml[i, j, :] = X[idx]

    #     return y_ml


    def ml_equalize(self, H, Y):

        x1, x2 = np.meshgrid(self.QAM_signal, self.QAM_signal, indexing="ij")
        X = np.stack((x1.ravel(), x2.ravel()), axis=1)

        H_mat = np.zeros((self.OFDM_symbols, self.subcarriers, 2, 2), dtype=complex)

        H_mat[:, :, 0, 0] = H[0::2, :, 0]
        H_mat[:, :, 0, 1] = H[0::2, :, 1]
        H_mat[:, :, 1, 0] = H[1::2, :, 0]
        H_mat[:, :, 1, 1] = H[1::2, :, 1]

        HX = np.einsum('ijab,kb->ijak', H_mat, X)

        diff = Y[:, :, :, None] - HX

        metric = np.sum(np.abs(diff)**2, axis=2)

        idx = np.argmin(metric, axis=2)

        y_ml = X[idx]

        return y_ml

    def demodulation(self, equalized_signal, modem):
        return self.modem.demodulate(equalized_signal, demod_type="hard")
    
    def llr(self, equalized_signal, key, H):

        bits_gray = []
        symbols_gray = []

        constellation = self.modem.constellation

        for i in range(len(constellation)):
            symbol = constellation[i]

            bits = self.modem.demodulate(np.array([symbol]), 'hard')
            
            bits_gray.append(bits)
            symbols_gray.append(symbol)

        # equalized_signal = equalized_signal.reshape(-1)
        # equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())

        sigma2 = self.power_of_signal(equalized_signal) / (2 * self.SNR)

        k_qam = int(np.log2(self.M))

        # if key == 'ml':

        #     llr = np.zeros((self.OFDM_symbols, self.subcarriers, 2, 4))

        #     for i in range(self.OFDM_symbols):
        #         for j in range(self.subcarriers):
        #             H_mat = np.array([ 
        #                 [H[2*i, j, 0], H[2*i, j, 1]],
        #                 [H[2*i+1, j, 0], H[2*i+1, j, 1]]
        #             ])
        #             # print(H_mat)
        #             llr_vect_1 = np.array([])
        #             llr_vect_2 = np.array([])
        #             for q in range(k_qam): # q \in [0;3]
        #                 bits_1, bits_2 = np.array([]), np.array([])
        #                 llr_0_1, llr_1_1 = np.array([]), np.array([])
        #                 llr_0_2, llr_1_2 = np.array([]), np.array([])
        #                 for p in range(self.M): # p \in [0;15]
        #                     if bits_gray[p][q] == 0:
        #                         for l in range(self.M): # l \in [0;15]
        #                             y_H = H_mat @ np.array([symbols_gray[p], symbols_gray[l]])
        #                             # llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n]) ** 2 / sigma2))
        #                             llr_0_1 = np.append(llr_0_1, np.exp(-np.linalg.norm(equalized_signal[i, j, :] - y_H) ** 2 / sigma2))
        #                             y_H = H_mat @ np.array([symbols_gray[l], symbols_gray[p]])
        #                             # llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n]) ** 2 / sigma2))
        #                             llr_0_2 = np.append(llr_0_2, np.exp(-np.linalg.norm(equalized_signal[i, j, :] - y_H) ** 2 / sigma2))
        #                     elif bits_gray[p][q] == 1:
        #                         for l in range(self.M): # l \in [0;15]
        #                             y_H = H_mat @ np.array([symbols_gray[p], symbols_gray[l]])
        #                             # llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n]) ** 2 / sigma2))
        #                             llr_1_1 = np.append(llr_1_1, np.exp(-np.linalg.norm(equalized_signal[i, j, :] - y_H) ** 2 / sigma2))
        #                             y_H = H_mat @ np.array([symbols_gray[l], symbols_gray[p]])
        #                             # llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n]) ** 2 / sigma2))
        #                             llr_1_2 = np.append(llr_1_2, np.exp(-np.linalg.norm(equalized_signal[i, j, :] - y_H) ** 2 / sigma2))

                            

        #                 divide_llr_1 = np.log(np.sum(llr_1_1) / np.sum(llr_0_1))
        #                 divide_llr_2 = np.log(np.sum(llr_1_2) / np.sum(llr_0_2))
                
        #                 if divide_llr_1 == np.inf:
        #                     divide_llr_1 = 10000
        #                 elif divide_llr_1 == -np.inf:
        #                     divide_llr_1 = -10000

        #                 if divide_llr_2 == np.inf:
        #                     divide_llr_2 = 10000
        #                 elif divide_llr_2 == -np.inf:
        #                     divide_llr_2 = -10000
                        
        #                 llr_vect_1 = np.append(llr_vect_1, divide_llr_1)
        #                 llr_vect_2 = np.append(llr_vect_2, divide_llr_2)
                    
        #             llr[i, j, 0, :], llr[i, j, 1, :] = llr_vect_1, llr_vect_2

        #     llr = llr.reshape(-1)
        #     llr = np.array(llr[0::2].tolist() + llr[1::2].tolist())


        if key == 'ml':

            llr = np.zeros((self.OFDM_symbols, self.subcarriers, 2*k_qam))

            bits = list(map(lambda x: format(x, f'0{int(np.log2(self.M))}b'), list(np.arange(0, self.M))))


            bits_x, bits_y = np.meshgrid(bits, bits)
            bits_x = bits_x.reshape(-1, 1)
            bits_y = bits_y.reshape(-1, 1)

            bits_grid = np.concatenate((bits_x, bits_y), axis=1)

            bits_for_llr_str = np.array(
                list(map(lambda x, y: x + y, bits_grid[:, 0], bits_grid[:, 1]))
            )

            bits_for_llr_int = np.array(
                [list(map(lambda x: int(x), bits_for_llr_str[b])) for b in range(len(bits_for_llr_str))]
            )

            qam_x, qam_y = np.meshgrid(constellation, constellation)
            qam_x = qam_x.reshape(-1, 1)
            qam_y = qam_y.reshape(-1, 1)

            qam_grid = np.concatenate((qam_x, qam_y), axis=1)
            # x, y = np.meshgrid(bits, bits, indexing='ij')
            # BITS = np.column_stack((x.ravel(), y.ravel()))
            
            y = np.zeros((2, self.M**2), dtype=complex)

            for i in range(self.OFDM_symbols):
                for j in range(self.subcarriers):

                    if i%2 == 0:
                        H_mat = np.array([ 
                            [H[2*i, j, 0], H[2*i, j, 1]],
                            [H[2*i+1, j, 0], H[2*i+1, j, 1]]
                        ])
                    else:
                        H_mat = np.array([ 
                            [H[2*i-1, j, 0], H[2*i-1, j, 1]],
                            [H[2*i, j, 0], H[2*i, j, 1]]
                        ])
                    # llr_vect = np.array([])
                    # llr_vect_2 = np.array([])

                    for k in range(2*k_qam):

                        llr_0, llr_1 = np.array([]), np.array([])

                        for q in range(self.M**2):

                            if bits_for_llr_int[q][k] == 0:

                                llr_0 = np.append(llr_0, np.exp(-np.sum(np.abs(equalized_signal[i, j, :]
                                 - H_mat @ qam_grid[q])**2) / sigma2))
                            
                            elif bits_for_llr_int[q][k] == 1:
                                
                                llr_1 = np.append(llr_1, np.exp(-np.sum(np.abs(equalized_signal[i, j, :] 
                                 - H_mat @ qam_grid[q])**2) / sigma2))
                    
                        divide_llr = np.log(np.sum(llr_1) / np.sum(llr_0))
                        
                        if divide_llr == np.inf:
                            divide_llr = 10000
                        elif divide_llr == -np.inf:
                            divide_llr = -10000
                        
                        # llr = np.append(llr, divide_llr)
                        llr[i, j, k] = divide_llr

            llr = llr.reshape(self.OFDM_symbols, self.subcarriers, 2, k_qam)

            llr = llr.transpose(0, 2, 1, 3)

            llr = llr.reshape(-1)
                        
                    

        elif key == 'zf' or key == 'mmse_classic' or key == 'mmse_householder':

            llr = np.array([])

            equalized_signal = equalized_signal.reshape(-1)
            equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())
            
        # sigma2 = np.var(equalized_signal - self.x)
            for i in range(len(equalized_signal)):
                for q in range(int(np.log2(self.M))): # q \in [0, 3]

                    llr_0, llr_1 = np.array([]), np.array([])

                    for p in range(self.M): # p \in [0, 15]
                        if bits_gray[p][q] == 0:
                            llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i] - symbols_gray[p])**2 / sigma2))
                        elif bits_gray[p][q] == 1:
                            llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i] - symbols_gray[p])**2 / sigma2))
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        divide_llr = np.log(np.sum(llr_1) / np.sum(llr_0))
                    
                    if divide_llr == np.inf:
                        divide_llr = 10000
                    elif divide_llr == -np.inf:
                        divide_llr = -10000
                    
                    llr = np.append(llr, divide_llr)

        return llr
    


    def BER_new(self, listt, H, num_err, num_bits, count, type_demod):
        
        key, equalized_signal = listt

        while num_err <= self.num_err_min:

            count += 1

            if count >= self.recursion_limit:
                break
            
            # equalized_signal = equalized_signal.reshape(-1)
            # equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())

            if self.flag_code:

                if type_demod == 'hard':

                    equalized_signal = equalized_signal.reshape(-1)
                    equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())

                    bits_output_code = self.demodulation(equalized_signal, self.modem)
                    # bits_output_decode = np.array(dot11a_codec.decode(bits_output_code))
                    bits_output_decode = viterbi_decode(
                        bits_output_code,
                        trellis,
                        tb_depth=20,
                        decoding_type='hard'
                    )

                    num_err += np.sum((self.bits_input + bits_output_decode) % 2)

                elif type_demod == 'soft':

                    llr = self.llr(equalized_signal, key, H)
                    # bits_output_decode = np.array(dot11a_codec.decode(bits_output_code))
                    bits_output_decode = viterbi_decode(
                        llr,
                        trellis,
                        tb_depth=20,
                        decoding_type='unquantized'
                    )

                    num_err += np.sum((self.bits_input + bits_output_decode) % 2)

            else:

                if type_demod == 'hard':

                    equalized_signal = equalized_signal.reshape(-1)
                    equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())

                    bits_output = self.demodulation(equalized_signal, self.modem)
                    num_err += np.sum((self.bits_input + bits_output) % 2)
                # elif type_demod == 'soft':
                #     bits_output = self.llr(equalized_signal)
                #     num_err += np.sum((self.bits_input + bits_output) % 2)

            num_bits += self.num_bits

            if num_err <= self.num_err_min:

                H, Y_OFDM = self.start()

                if key == 'zf':
                    y_ml = []
                    y_zf = self.zf_equalize(H, Y_OFDM)
                    y_mmse_classic = []
                    y_mmse_householder = []
                elif key == 'mmse_classic':
                    y_ml = []
                    y_zf = []
                    y_mmse_classic = self.mmse_equalize(H, Y_OFDM, mmse_solver='classic')
                    y_mmse_householder = []
                elif key == 'mmse_householder':
                    y_ml = []
                    y_zf = []
                    y_mmse_classic = []
                    y_mmse_householder = self.mmse_equalize(H, Y_OFDM, mmse_solver='householder')
                elif key == 'ml':
                    y_ml = []
                    y_zf = []
                    y_mmse_classic = []
                    y_mmse_householder = []

                data = {'zf': y_zf, 'mmse_classic': y_mmse_classic, 'mmse_householder': y_mmse_householder, 'ml': Y_OFDM}

                if key == 'zf':
                    num_err, num_bits = self.BER_new(list(data.items())[0], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > self.num_err_min:
                        return num_err, num_bits
                elif key == 'mmse_classic':
                    num_err, num_bits = self.BER_new(list(data.items())[1], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > self.num_err_min:
                        return num_err, num_bits
                elif key == 'mmse_householder':
                    num_err, num_bits = self.BER_new(list(data.items())[2], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > self.num_err_min:
                        return num_err, num_bits
                elif key == 'ml':
                    num_err, num_bits = self.BER_new(list(data.items())[3], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > self.num_err_min:
                        return num_err, num_bits
                    
        # print(f'количество ошибок для {key}, code = {self.flag_code}: {num_err}')

        return num_err, num_bits

    def EVM(self, equalized_signal):
        equalized_signal = equalized_signal.reshape(-1)
        equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())
        return np.mean(np.abs(self.x - equalized_signal) ** 2) # np.sum(np.abs(self.x - equalized_signal) ** 2) / len(self.x)
    
    def del_pilots(self, r):
        for i in range(self.OFDM_symbols):
            r = np.delete(r, i + 1, axis=0)
        return r
    
    def impulse_response(self, diagram=False):
        tau_us = np.array([0, 3, 5, 6, 8])  # мкс
        power_dB = np.array([0, -8, -17, -21, -25])

        power = 10 ** (power_dB / 10)

        amplitude = np.sqrt(power)

        phase = np.zeros_like(tau_us, dtype=complex)
        N = len(phase)
        for k in range(N):
            phase[k] = 2 * np.pi * np.random.randint(0, N - 1) / N

        h = amplitude * np.exp(1j * phase)
        # print(h)
        if diagram:
            plt.title("Импульсная характеристика рассматриваемого канала", fontsize=16)
            plt.stem(tau_us, amplitude, linefmt="black")
            plt.xlabel("Отсчёт, мкс", fontsize=14)
            plt.ylabel("Амплитуда", fontsize=14)
            plt.show()

        return h
    
    def start(self):

        self.bit_input()
        self.modul()

        X_OFDM = self.OFDM()

        X_OFDM_pilots = self.add_pilots(X_OFDM)

        x_t_pilots = self.idft(X_OFDM_pilots) * np.sqrt(self.subcarriers)
        # x_t = self.del_pilots(x_t_pilots)

        x_t_vector_pilots = x_t_pilots.reshape(-1)
        # x_t_vector = x_t.reshape(-1)

        y_clear_t_pilots = self.convolution_cp_MIMO(x_t_pilots) 
        # y_clear_t = self.del_pilots(y_clear_t_pilots)

        awgn = self.AWGN(x_t_vector_pilots).reshape(3*self.OFDM_symbols, self.subcarriers, 2)

        mask = np.arange(awgn.shape[0]) % 3 != 0
        awgn[mask, :, :] = 0

        y_t_pilots = y_clear_t_pilots + awgn #/ np.sqrt(self.subcarriers)
        # y_t = self.del_pilots(y_t_pilots)

        Y_OFDM_pilots = self.dft(y_t_pilots) / np.sqrt(self.subcarriers)



        indices_to_keep = np.arange(0, Y_OFDM_pilots.shape[0], 3)

        all_indices = np.arange(Y_OFDM_pilots.shape[0])
        indices_to_remove = np.delete(all_indices, indices_to_keep)

        H_sum = Y_OFDM_pilots[indices_to_remove, :, :]

        Y_OFDM = Y_OFDM_pilots[indices_to_keep, :, :]


        H = np.zeros_like(H_sum)

        even_idx = np.arange(0, 2*self.OFDM_symbols, 2)
        odd_idx = np.arange(1, 2*self.OFDM_symbols, 2)

        H[even_idx, :, 0] = 2 * H_sum[even_idx, :, 0] - H_sum[odd_idx, :, 0]
        H[even_idx, :, 1] = -H_sum[even_idx, :, 0] + H_sum[odd_idx, :, 0]

        H[odd_idx, :, 0] = 2 * H_sum[even_idx, :, 1] - H_sum[odd_idx, :, 1]
        H[odd_idx, :, 1] = -H_sum[even_idx, :, 1] + H_sum[odd_idx, :, 1]

        return H, Y_OFDM





def one_iteration(SNR_dB):

    results = {}

# # zf без кодера
#     qam = QAMmodulation(
#         M=16,
#         SNR_dB=SNR_dB,
#         subcarriers=7,
#         OFDM_symbols=20,
#         flag_code=False
#     )

#     H, Y_OFDM = qam.start()

#     y_zf = qam.zf_equalize(H, Y_OFDM)
#     y_mmse_classic = []

#     data = {'zf': y_zf, 'mmse': y_mmse_classic}

#     num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0],
#                                             H, 
#                                             num_err=0, 
#                                             num_bits=0, 
#                                             count=0, 
#                                             type_demod='hard')
#     print(f'zf без кодера: num_err = {num_err_zf}, num_bits = {num_bits_zf}')

#     results['ber_zf'] = num_err_zf / num_bits_zf

# # zf с кодером
#     qam = QAMmodulation(
#         M=16,
#         SNR_dB=SNR_dB,
#         subcarriers=7,
#         OFDM_symbols=20,
#         flag_code=True
#     )

#     H, Y_OFDM = qam.start()

#     y_zf = qam.zf_equalize(H, Y_OFDM)
#     y_mmse = []

#     data = {'zf': y_zf, 'mmse': y_mmse}

#     num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0], 
#                                             H,
#                                             num_err=0, 
#                                             num_bits=0, 
#                                             count=0, 
#                                             type_demod='hard')
#     print(f'zf с кодером: num_err = {num_err_zf}, num_bits = {num_bits_zf}')

#     results['ber_zf_with_code'] = num_err_zf / num_bits_zf
    
# # zf с кодером soft
#     qam = QAMmodulation(
#         M=16,
#         SNR_dB=SNR_dB,
#         subcarriers=7,
#         OFDM_symbols=20,
#         flag_code=True
#     )

#     H, Y_OFDM = qam.start()

#     y_zf = qam.zf_equalize(H, Y_OFDM)
#     y_mmse = []

#     data = {'zf': y_zf, 'mmse': y_mmse}

#     num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0], 
#                                             H,
#                                             num_err=0, 
#                                             num_bits=0, 
#                                             count=0, 
#                                             type_demod='soft')
#     print(f'zf с кодером soft: num_err = {num_err_zf}, num_bits = {num_bits_zf}')

#     results['ber_zf_with_code_soft'] = num_err_zf / num_bits_zf

# mmse без кодера classic
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=False
    )

    H, Y_OFDM = qam.start()

    y_zf = qam.zf_equalize(H, Y_OFDM)
    y_mmse_classic = qam.mmse_equalize(H, Y_OFDM, mmse_solver='classic')
    y_mmse_householder = qam.mmse_equalize(H, Y_OFDM, mmse_solver='householder')

    # results['evm_zf'] = qam.EVM(y_zf)
    results['evm_mmse_classic'] = qam.EVM(y_mmse_classic)
    results['evm_mmse_householder'] = qam.EVM(y_mmse_householder)

    data = {'zf': y_zf, 'mmse_classic': y_mmse_classic}


    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
    print(f'mmse без кодера classic: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_classic'] = num_err_mmse / num_bits_mmse


# mmse с кодером classic
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=True
    )

    H, Y_OFDM = qam.start()

    y_zf = []
    y_mmse_classic = qam.mmse_equalize(H, Y_OFDM, mmse_solver='classic')

    data = {'zf': y_zf, 'mmse_classic': y_mmse_classic}

    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H, 
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
    print(f'mmse с кодером classic: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_with_code_classic'] = num_err_mmse / num_bits_mmse

# mmse с кодером soft classic
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=True
    )

    H, Y_OFDM = qam.start()

    y_zf = []
    y_mmse_classic = qam.mmse_equalize(H, Y_OFDM, mmse_solver='classic')

    data = {'zf': y_zf, 'mmse_classic': y_mmse_classic}

    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H, 
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='soft')
    print(f'mmse с кодером soft classic: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_with_code_soft_classic'] = num_err_mmse / num_bits_mmse


# mmse без кодера householder
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=False
    )

    H, Y_OFDM = qam.start()

    y_zf = qam.zf_equalize(H, Y_OFDM)
    y_mmse_householder = qam.mmse_equalize(H, Y_OFDM, mmse_solver='householder')

    data = {'zf': y_zf, 'mmse_householder': y_mmse_householder}

    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
    print(f'mmse без кодера householder: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_householder'] = num_err_mmse / num_bits_mmse

# mmse с кодером householder
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=True
    )

    H, Y_OFDM = qam.start()

    y_zf = []
    y_mmse_householder = qam.mmse_equalize(H, Y_OFDM, mmse_solver='householder')

    data = {'zf': y_zf, 'mmse_householder': y_mmse_householder}

    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H, 
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
    print(f'mmse с кодером householder: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_with_code_householder'] = num_err_mmse / num_bits_mmse

# mmse с кодером soft householder
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR_dB,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=True
    )

    H, Y_OFDM = qam.start()

    y_zf = []
    y_mmse_householder = qam.mmse_equalize(H, Y_OFDM, mmse_solver='householder')

    data = {'zf': y_zf, 'mmse_householder': y_mmse_householder}

    num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                H, 
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='soft')
    print(f'mmse с кодером soft householder: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')

    results['ber_mmse_with_code_soft_householder'] = num_err_mmse / num_bits_mmse


# # ml без кодера
#         qam = QAMmodulation(
#             M=16,
#             SNR_dB=SNR_dB,
#             subcarriers=7,
#             OFDM_symbols=20,
#             flag_code=False
#         )

#         H, Y_OFDM = qam.start()

#         y_ml = qam.ml_equalize(H, Y_OFDM)
#         y_zf = []
#         y_mmse = []

#         data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

#         num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2], 
#                                                 H,
#                                                 num_err=0, 
#                                                 num_bits=0, 
#                                                 count=0, 
#                                                 type_demod='hard')
#         print(f'ml без кодера: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
#         BER_ml_list.append(num_err_ml / num_bits_ml)

# # ml с кодером
#         qam = QAMmodulation(
#             M=16,
#             SNR_dB=SNR_dB,
#             subcarriers=7,
#             OFDM_symbols=20,
#             flag_code=True
#         )

#         H, Y_OFDM = qam.start()

#         y_ml = qam.ml_equalize(H, Y_OFDM)
#         y_zf = []
#         y_mmse = []

#         data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

#         num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2],
#                                                 H,
#                                                 num_err=0, 
#                                                 num_bits=0, 
#                                                 count=0, 
#                                                 type_demod='hard')
#         print(f'ml с кодером: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
#         BER_ml_list_code.append(num_err_ml / num_bits_ml)

# # ml с кодером soft
#         qam = QAMmodulation(
#             M=16,
#             SNR_dB=SNR_dB,
#             subcarriers=7,
#             OFDM_symbols=20,
#             flag_code=True
#         )

#         H, Y_OFDM = qam.start()

#         y_ml = qam.ml_equalize(H, Y_OFDM)
#         y_zf = []
#         y_mmse = []

#         data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

#         num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2],
#                                                 H,
#                                                 num_err=0,
#                                                 num_bits=0,
#                                                 count=0,
#                                                 type_demod='soft')
#         print(f'ml с кодером soft: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
#         BER_ml_list_code_soft.append(num_err_ml / num_bits_ml)

    return results


if __name__ == '__main__':

    SNR_dB_set = np.arange(0, 21, 5)
    N_avg = 5
    MAX_WORKERS = 4 # Оптимально для ваших 6 ядер

    # final_ber_zf = []
    # final_ber_zf_code = []
    # final_ber_zf_code_soft = []

    final_ber_mmse_classic = []
    final_ber_mmse_code_classic = []
    final_ber_mmse_code_soft_classic = []

    final_ber_mmse_householder = []
    final_ber_mmse_code_householder = []
    final_ber_mmse_code_soft_householder = []

    final_evm_zf = []
    final_evm_mmse_classic = []
    final_evm_mmse_householder = []

    start_time = time.perf_counter()

    for snr_db in SNR_dB_set:
        print(f"Запуск параллельного расчета для SNR = {snr_db} dB...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Распределяем N_avg задач между ядрами
            # Передаем список из N_avg элементов, каждый из которых равен текущему snr
            futures = [executor.submit(one_iteration, snr_db) for _ in range(N_avg)]
            
            # Собираем результаты по мере готовности
            batch_results = []
            for future in concurrent.futures.as_completed(futures):
                batch_results.append(future.result())

        # Усредняем результаты этого SNR

        # final_ber_zf.append(np.mean([r['ber_zf'] for r in batch_results]))
        # final_ber_zf_code.append(np.mean([r['ber_zf_with_code'] for r in batch_results]))
        # final_ber_zf_code_soft.append(np.mean([r['ber_zf_with_code_soft'] for r in batch_results]))

        final_ber_mmse_classic.append(np.mean([r['ber_mmse_classic'] for r in batch_results]))
        final_ber_mmse_code_classic.append(np.mean([r['ber_mmse_with_code_classic'] for r in batch_results]))
        final_ber_mmse_code_soft_classic.append(np.mean([r['ber_mmse_with_code_soft_classic'] for r in batch_results]))

        final_ber_mmse_householder.append(np.mean([r['ber_mmse_householder'] for r in batch_results]))
        final_ber_mmse_code_householder.append(np.mean([r['ber_mmse_with_code_householder'] for r in batch_results]))
        final_ber_mmse_code_soft_householder.append(np.mean([r['ber_mmse_with_code_soft_householder'] for r in batch_results]))

        # final_evm_zf.append(np.mean([r['evm_zf'] for r in batch_results]))
        final_evm_mmse_classic.append(np.mean([r['evm_mmse_classic'] for r in batch_results]))
        final_evm_mmse_householder.append(np.mean([r['evm_mmse_householder'] for r in batch_results]))


    end_time = time.perf_counter()

    print(f'время выполнения: {round(end_time - start_time, 2)} секунд = {round((end_time - start_time) / 60, 2)} минут = {round((end_time - start_time) / 3600, 2)} часов')


    qam = QAMmodulation(M=16, SNR_dB=0, subcarriers=7, OFDM_symbols=20, flag_code=True)


    # plt.figure(figsize=(10, 6))

    # plt.suptitle(f'реализаций: {N_avg}, ошибок: {qam.num_err_min}, глубина рекурсии: {qam.recursion_limit}')

    # # BER:

    # # ZF
    # plt.xlabel("SNR_dB", fontsize=12)
    # plt.ylabel("BER", fontsize=12)
    # plt.xlim()
    # plt.ylim()
    # plt.semilogy(SNR_dB_set, final_ber_zf, color="red", label="zf")
    # plt.semilogy(SNR_dB_set, final_ber_zf_code, color="red", linestyle='--', label="zf_code")
    # plt.semilogy(SNR_dB_set, final_ber_zf_code_soft, color="red", linestyle='-.', label="zf_code_soft")
    # plt.legend(loc='upper right', fontsize=8)

    # plt.savefig('ber_zf_multiprocessing')

    # plt.tight_layout()


    plt.figure(figsize=(10, 6))

    plt.suptitle(f'реализаций: {N_avg}, ошибок: {qam.num_err_min}, глубина рекурсии: {qam.recursion_limit}')

    # MMSE
    plt.subplot(1, 2, 1)
    plt.xlabel("SNR_dB", fontsize=12)
    plt.ylabel("BER", fontsize=12)
    plt.xlim()
    plt.ylim()
    # classic
    plt.semilogy(SNR_dB_set, final_ber_mmse_classic, color="green", label="mmse_classic")
    plt.semilogy(SNR_dB_set, final_ber_mmse_code_classic, color="green", linestyle='--', label="mmse_code_classic")
    plt.semilogy(SNR_dB_set, final_ber_mmse_code_soft_classic, color="green", linestyle=':', label="mmse_code_soft_classic")
    # householder
    plt.semilogy(SNR_dB_set, final_ber_mmse_householder, color="black", label="mmse_householder")
    plt.semilogy(SNR_dB_set, final_ber_mmse_code_householder, color="black", linestyle='--', label="mmse_code_householder")
    plt.semilogy(SNR_dB_set, final_ber_mmse_code_soft_householder, color="black", linestyle=':', label="mmse_code_soft_householder")

    plt.legend(loc='lower left', fontsize=8)

    # ML
    # plt.subplot(1, 3, 2)
    # plt.xlabel("SNR_dB", fontsize=12)
    # plt.ylabel("BER", fontsize=12)
    # plt.xlim()
    # plt.ylim()
    # plt.semilogy(SNR_dB_set, final_ber_ml, color="magenta", label="ml")
    # plt.semilogy(SNR_dB_set, final_ber_ml_code, color="magenta", linestyle='--', label="ml_code")
    # plt.semilogy(SNR_dB_set, final_ber_ml_code_soft, color="magenta", linestyle='-.', label="ml_code_soft")
    # plt.legend(loc='upper right', fontsize=8)

    # EVM:
    plt.subplot(1, 2, 2)
    plt.xlabel("SNR_dB", fontsize=12)
    plt.ylabel("|EVM_classic - EVM_householder|", fontsize=12)
    plt.xlim()
    plt.ylim()
    # plt.semilogy(SNR_dB_set, final_evm_zf, color="red", label="zf")
    plt.plot(SNR_dB_set, np.abs(np.array(final_evm_mmse_classic) - np.array(final_evm_mmse_householder)), color="green", label="|EVM_classic - EVM_householder|")
    # plt.semilogy(SNR_dB_set, final_evm_mmse_householder, color="green", linestyle=':', label="mmse_householder")
    plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    plt.savefig('ber_mmse_and_evm_multiprocessing')

    plt.show()
