import numpy as np
import matplotlib.pyplot as plt
import commpy as cp

from commpy.channelcoding import Trellis, conv_encode, viterbi_decode
# from viterbi import Viterbi

# dot11a_codec = Viterbi(7, [0o133, 0o171])

trellis = Trellis(
    memory=np.array([6]),        # constraint length - 1
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
        # self.bits_input = np.random.randint(0, 2, self.num_bits)
        # self.bits_input_1 = self.bits_input[:int(self.num_bits/2)]
        # self.bits_input_2 = self.bits_input[int(self.num_bits/2):]
        # self.modem = cp.QAMModem(self.M)
        # self.x = self.modem.modulate(self.bits_input)
        # self.x_1 = self.x[:int(len(self.x)/2)]
        # self.x_2 = self.x[int(len(self.x)/2):]


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
        # print(f'modem = {self.modem}')
        if self.flag_code:
            self.x = self.modem.modulate(self.bits_input_code)
            self.QAM_signal = np.array(list(set(self.x)))
        else:
            self.x = self.modem.modulate(self.bits_input)
            self.QAM_signal = np.array(list(set(self.x)))
        # print(f'x = {self.x}')
        self.x_1 = self.x[:int(len(self.x)/2)]
        # print(f'x_1 = {self.x_1}')
        self.x_2 = self.x[int(len(self.x)/2):]
        # print(f'x_2 = {self.x_2}')

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
    
        # def convolution_cp_matrix(self, x_t):  # возвращает матрицу с cp
    #     y_clear_t = []
    #     for i in range(x_t.shape[0]):
    #         if i%2 == 0:
    #             h = self.impulse_response(diagram=False)
    #         OFDM_i = x_t[i]
    #         y_clear_t = np.append(y_clear_t, self.convolution_cp(OFDM_i, h))
    #     return y_clear_t.reshape(x_t.shape[0], self.subcarriers + len(h) - 1)

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
                
                y_zf[i, j, 0] = (d * r[i, j, 0] - b * r[i, j, 1]) / (det + 1e-10)
                y_zf[i, j, 1] = (-c * r[i, j, 0] + a * r[i, j, 1]) / (det + 1e-10)
        
        return y_zf

    def mmse_equalize(self, H, Y):

        y_mmse = np.zeros_like(Y, dtype=complex)
        
        for i in range(self.OFDM_symbols):
            for j in range(self.subcarriers):

                H_mat = np.array([[H[2*i, j, 0], H[2*i, j, 1]],
                                [H[2*i+1, j, 0], H[2*i+1, j, 1]]])
                
                Y_vec = np.array([Y[i, j, 0], Y[i, j, 1]])
                
                H_H = H_mat.conj().T  # эрмитово сопряжение
                G = H_H @ H_mat + np.eye(2)/self.SNR
                
                # решаем систему G * X_mmse = H_H * Y_vec
                X_vec = np.linalg.solve(G, H_H @ Y_vec)
                
                y_mmse[i, j, 0] = X_vec[0]
                y_mmse[i, j, 1] = X_vec[1]
        
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

    # все кандидаты (M^2,2)
        x1, x2 = np.meshgrid(self.QAM_signal, self.QAM_signal, indexing="ij")
        X = np.stack((x1.ravel(), x2.ravel()), axis=1)

        # матрицы канала (OFDM, sub, 2,2)
        H_mat = np.zeros((self.OFDM_symbols, self.subcarriers, 2, 2), dtype=complex)

        H_mat[:, :, 0, 0] = H[0::2, :, 0]
        H_mat[:, :, 0, 1] = H[0::2, :, 1]
        H_mat[:, :, 1, 0] = H[1::2, :, 0]
        H_mat[:, :, 1, 1] = H[1::2, :, 1]

        HX = np.einsum('ijab,kb->ijak', H_mat, X) # H*X для всех кандидатов

        diff = Y[:, :, :, None] - HX

        metric = np.sum(np.abs(diff)**2, axis=2)

        idx = np.argmin(metric, axis=2)

        y_ml = X[idx]

        return y_ml

    def demodulation(self, equalized_signal, modem):
        return self.modem.demodulate(equalized_signal, demod_type="hard")
    
    def llr(self, equalized_signal, key, H):

        llr = np.array([])

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

        k = int(np.log2(self.M))

        if key == 'ml':
            for i in range(self.OFDM_symbols):
                for j in range(self.subcarriers):
                    H_mat = np.array([ # [][]
                        [H[2*i, j, 0], H[2*i, j, 1]],
                        [H[2*i+1, j, 0], H[2*i+1, j, 1]]
                    ])
                    # print(H_mat)
                    for n in range(2):
                        for q in range(k): # q \in [0, 3]
                            llr_0, llr_1 = np.array([]), np.array([])
                            for p in range(self.M): # p \in [0, 15]
                                if bits_gray[p][q] == 0:
                                    for l in range(self.M):
                                        y_H = H_mat @ np.array([symbols_gray[p], symbols_gray[l]])
                                        llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n])))
                                elif bits_gray[p][q] == 1:
                                    for l in range(self.M):
                                        y_H = H_mat @ np.array([symbols_gray[p], symbols_gray[l]])
                                        llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i + n*j] - y_H[n])))

                            divide_llr = np.log(np.sum(llr_1) / np.sum(llr_0))
                    
                            if divide_llr == np.inf:
                                divide_llr = 10000
                            elif divide_llr == -np.inf:
                                divide_llr = -10000
                            
                            llr = np.append(llr, divide_llr)

        elif key == 'zf' or key == 'mmse':
        # sigma2 = np.var(equalized_signal - self.x)
            for i in range(len(equalized_signal)):
                for q in range(int(np.log2(self.M))): # q \in [0, 3]

                    llr_0, llr_1 = np.array([]), np.array([])

                    for p in range(self.M): # p \in [0, 15]
                        if bits_gray[p][q] == 0:
                            # llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i, j, k] - symbols_gray[p])**2 / (2*self.RMSD(equalized_signal)**2)))
                            llr_0 = np.append(llr_0, np.exp(-np.abs(equalized_signal[i] - symbols_gray[p])**2 / sigma2))
                        elif bits_gray[p][q] == 1:
                            # llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i, j, k] - symbols_gray[p])**2 / (2*self.RMSD(equalized_signal)**2)))
                            llr_1 = np.append(llr_1, np.exp(-np.abs(equalized_signal[i] - symbols_gray[p])**2 / sigma2))
                    
                    divide_llr = np.log(np.sum(llr_1) / np.sum(llr_0))
                    
                    if divide_llr == np.inf:
                        divide_llr = 10000
                    elif divide_llr == -np.inf:
                        divide_llr = -10000
                    
                    llr = np.append(llr, divide_llr)

        # bits_llr = np.where(llr < 0, 1, 0)

        return llr # bits_llr

    # def llr(self, equalized_signal):

    #     constellation = self.modem.constellation
    #     M = self.M
    #     k = int(np.log2(M))

    #     # ---- формируем bits_gray и symbols_gray ----
    #     symbols_gray = np.array(constellation)

    #     bits_gray = np.array([
    #         self.modem.demodulate(np.array([symbol]), 'hard')
    #         for symbol in symbols_gray
    #     ])  # shape: (M, k)

    #     # ---- шум ----
    #     sigma2 = self.power_of_signal(equalized_signal) / (2 * self.SNR)

    #     # ---- расстояния до всех точек созвездия ----
    #     # shape: (N, M)
    #     dist2 = np.abs(equalized_signal[:, None] - symbols_gray[None, :])**2

    #     # ---- экспоненты ----
    #     # shape: (N, M)
    #     metrics = np.exp(-dist2 / sigma2)

    #     # ---- LLR ----
    #     llr_list = []

    #     for q in range(k):
    #         mask0 = bits_gray[:, q] == 0  # shape: (M,)
    #         mask1 = bits_gray[:, q] == 1

    #         # суммы по соответствующим символам
    #         sum0 = np.sum(metrics[:, mask0], axis=1)  # shape: (N,)
    #         sum1 = np.sum(metrics[:, mask1], axis=1)

    #         divide_llr = np.log(sum1 / (sum0 + 1e-10))

    #         # обработка inf как у тебя
    #         divide_llr[np.isposinf(divide_llr)] = 10000
    #         divide_llr[np.isneginf(divide_llr)] = -10000

    #         llr_list.append(divide_llr)

    #     # собираем в один вектор (как у тебя через append)
    #     llr = np.stack(llr_list, axis=1).reshape(-1)

    #     return llr
    
    def BER_new(self, listt, H, num_err, num_bits, count, type_demod):
        
        key, equalized_signal = listt

        num_err_min = 10

        while num_err <= num_err_min:

            count += 1

            if count >= 400:
                break
            
            equalized_signal = equalized_signal.reshape(-1)
            equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())

            if self.flag_code:
                if type_demod == 'hard':
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
                    bits_output = self.demodulation(equalized_signal, self.modem)
                    num_err += np.sum((self.bits_input + bits_output) % 2)
                # elif type_demod == 'soft':
                #     bits_output = self.llr(equalized_signal)
                #     num_err += np.sum((self.bits_input + bits_output) % 2)

            num_bits += self.num_bits

            if num_err <= num_err_min:

                H, Y_OFDM = self.start()

                if key == 'zf':
                    y_ml = []
                    y_zf = self.zf_equalize(H, Y_OFDM)
                    y_mmse = []
                elif key == 'mmse':
                    y_ml = []
                    y_zf = []
                    y_mmse = self.mmse_equalize(H, Y_OFDM)
                elif key == 'ml':
                    y_ml = self.ml_equalize(H, Y_OFDM)
                    y_zf = []
                    y_mmse = []

                data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

                if key == 'zf':
                    num_err, num_bits = self.BER_new(list(data.items())[0], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > num_err_min:
                        return num_err, num_bits
                elif key == 'mmse':
                    num_err, num_bits = self.BER_new(list(data.items())[1], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > num_err_min:
                        return num_err, num_bits
                elif key == 'ml':
                    num_err, num_bits = self.BER_new(list(data.items())[2], H, num_err=num_err, num_bits=num_bits, count=count, type_demod=type_demod)
                    if num_err > num_err_min:
                        return num_err, num_bits
                    
        # print(f'количество ошибок для {key}, code = {self.flag_code}: {num_err}')

        return num_err, num_bits

    def EVM(self, equalized_signal):
        equalized_signal = equalized_signal.reshape(-1)
        equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())
        return np.sum(np.abs(self.x - equalized_signal) ** 2) / len(self.x)
    
    def del_pilots(self, r):
        for i in range(qam.OFDM_symbols):
            r = np.delete(r, i + 1, axis=0)
        return r
    
    def impulse_response(self, diagram=False):
        tau_us = np.array([0, 3, 5, 6, 8])  # мкс
        power_dB = np.array([0, -8, -17, -21, -25])

        # Мощность в линейных единицах
        power = 10 ** (power_dB / 10)

        # Амплитуды
        amplitude = np.sqrt(power)

        # Фазы с равномерным распределением
        phase = np.zeros_like(tau_us, dtype=complex)
        N = len(phase)
        for k in range(N):
            phase[k] = 2 * np.pi * np.random.randint(0, N - 1) / N

        # Импульсная характеристика
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

        x_t_pilots = self.idft(X_OFDM_pilots) * np.sqrt(qam.subcarriers)
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

SNR_set = [1, 3, 5, 10, 12, 15, 17, 20]

for SNR in SNR_set:
    qam = QAMmodulation(
        M=16,
        SNR_dB=SNR,
        subcarriers=7,
        OFDM_symbols=20,
        flag_code=True
    )
    
    H, Y_OFDM = qam.start()

    y_zf = qam.zf_equalize(H, Y_OFDM)
    y_mmse = qam.mmse_equalize(H, Y_OFDM)
    y_ml = qam.ml_equalize(H, Y_OFDM)

    Y_OFDM = Y_OFDM.reshape(-1)
    Y_OFDM = np.array(Y_OFDM[0::2].tolist() + Y_OFDM[1::2].tolist())

    llr = qam.llr(Y_OFDM, 'ml', H)
    # bits_output_decode = np.array(dot11a_codec.decode(bits_llr))
    bits_output_decode = viterbi_decode(
                        llr,
                        trellis,
                        tb_depth=20,
                        decoding_type='unquantized'
                    )

    # llr = qam.llr(y_zf)
    # bits_output_decode = np.array(dot11a_codec.decode(llr))

    err = np.sum((qam.bits_input + bits_output_decode) % 2) / qam.num_bits
    print(f'with SNR = {SNR} dB BER = {round(err, 4)}')
pass

# constellation = qam.modem.constellation

# bits_gray = []
# symbols_gray = []

# for i in range(len(constellation)):
#     symbol = constellation[i]
#     bits = qam.modem.demodulate(np.array([symbol]), 'hard')
    
#     bits_gray.append(bits)
#     symbols_gray.append(symbol)

# q = []

# for i in range(16):
#     q.append(qam.modem.constellation[i])
# print(q)


# plt.figure(figsize=(10, 6))

# plt.subplot(1, 3, 1)
# plt.title(f"Modulated signal (QAM{qam.M}), {qam.num_symbols}symbols")
# plt.xlim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
# plt.ylim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
# plt.xlabel("I", fontsize=12)
# plt.ylabel("Q", fontsize=12)
# plt.scatter(np.real(qam.x), np.imag(qam.x), color="red")

# plt.subplot(1, 3, 2)
# plt.title(f"Signal after channel with AWGN (SNR={qam.SNR_dB} dB)", fontsize=12)
# plt.xlabel("I", fontsize=12)
# plt.ylabel("Q", fontsize=12)
# plt.xlim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
# plt.ylim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x)))
# plt.scatter(np.real(Y_OFDM), np.imag(Y_OFDM), color="black", s=0.5)
# plt.scatter(np.real(qam.x), np.imag(qam.x), color="red", s=20)

# plt.subplot(1, 3, 3)
# plt.title("Equalized signal", fontsize=12)
# plt.xlabel("I", fontsize=12)
# plt.ylabel("Q", fontsize=12)
# plt.xlim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
# plt.ylim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
# plt.scatter(np.real(y_zf), np.imag(y_zf), s=0.5, color="blue")
# plt.scatter(np.real(y_mmse), np.imag(y_mmse), s=0.5, color="green")
# plt.scatter(np.real(qam.x), np.imag(qam.x), color="red", s=20)
# plt.legend(labels=["ZF", "MMSE"], fontsize=10)

# plt.tight_layout()
# plt.show()


SNR_dB_set = np.arange(0, 21, 2)

ber_zf, ber_zf_code, ber_zf_code_soft, evm_zf = [], [], [], []
ber_mmse, ber_mmse_code, ber_mmse_soft, ber_mmse_code_soft, evm_mmse = [], [], [], [], []
ber_ml, ber_ml_code, ber_ml_soft, ber_ml_code_soft = [], [], [], []

N_avg = 4

for SNR_dB in SNR_dB_set:

    BER_zf_list, BER_zf_list_code, BER_zf_list_code_soft, EVM_zf_list = [], [], [], []
    BER_mmse_list, BER_mmse_list_code, BER_mmse_list_soft, BER_mmse_list_code_soft, EVM_mmse_list = [], [], [], [], []
    BER_ml_list, BER_ml_list_code, BER_ml_list_soft, BER_ml_list_code_soft = [], [], [], []

    print(f'считается SNR = {SNR_dB} dB')

    for _ in range(N_avg):

# zf без кодера
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=False
        )

        H, Y_OFDM = qam.start()

        y_zf = qam.zf_equalize(H, Y_OFDM)
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse}

        num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0],
                                                H, 
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
        print(f'zf без кодера: num_err = {num_err_zf}, num_bits = {num_bits_zf}')
        BER_zf_list.append(num_err_zf / num_bits_zf)

# zf с кодером
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_zf = qam.zf_equalize(H, Y_OFDM)
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse}

        num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0], 
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
        print(f'zf с кодером: num_err = {num_err_zf}, num_bits = {num_bits_zf}')
        BER_zf_list_code.append(num_err_zf / num_bits_zf)

# zf с кодером soft
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_zf = qam.zf_equalize(H, Y_OFDM)
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse}

        num_err_zf, num_bits_zf = qam.BER_new(list(data.items())[0], 
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='soft')
        print(f'zf с кодером soft: num_err = {num_err_zf}, num_bits = {num_bits_zf}')
        BER_zf_list_code_soft.append(num_err_zf / num_bits_zf)

# mmse без кодера
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=False
        )

        H, Y_OFDM = qam.start()

        y_zf = qam.zf_equalize(H, Y_OFDM)
        y_mmse = qam.mmse_equalize(H, Y_OFDM)

        data = {'zf': y_zf, 'mmse': y_mmse}

        EVM_zf_list.append(qam.EVM(y_zf))
        EVM_mmse_list.append(qam.EVM(y_mmse))

        num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                    H,
                                                    num_err=0, 
                                                    num_bits=0, 
                                                    count=0, 
                                                    type_demod='hard')
        print(f'mmse без кодера: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')
        BER_mmse_list.append(num_err_mmse / num_bits_mmse)

# mmse с кодером
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_zf = []
        y_mmse = qam.mmse_equalize(H, Y_OFDM)

        data = {'zf': y_zf, 'mmse': y_mmse}

        num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                    H, 
                                                    num_err=0, 
                                                    num_bits=0, 
                                                    count=0, 
                                                    type_demod='hard')
        print(f'mmse с кодером: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')
        BER_mmse_list_code.append(num_err_mmse / num_bits_mmse)

# mmse с кодером soft
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_zf = []
        y_mmse = qam.mmse_equalize(H, Y_OFDM)

        data = {'zf': y_zf, 'mmse': y_mmse}

        num_err_mmse, num_bits_mmse = qam.BER_new(list(data.items())[1], 
                                                    H, 
                                                    num_err=0, 
                                                    num_bits=0, 
                                                    count=0, 
                                                    type_demod='soft')
        print(f'mmse с кодером soft: num_err = {num_err_mmse}, num_bits = {num_bits_mmse}')
        BER_mmse_list_code_soft.append(num_err_mmse / num_bits_mmse)



# ml без кодера
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=False
        )

        H, Y_OFDM = qam.start()

        y_ml = qam.ml_equalize(H, Y_OFDM)
        y_zf = []
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

        num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2], 
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
        print(f'ml без кодера: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
        BER_ml_list.append(num_err_ml / num_bits_ml)

# ml с кодером
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_ml = qam.ml_equalize(H, Y_OFDM)
        y_zf = []
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

        num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2],
                                                H,
                                                num_err=0, 
                                                num_bits=0, 
                                                count=0, 
                                                type_demod='hard')
        print(f'ml с кодером: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
        BER_ml_list_code.append(num_err_ml / num_bits_ml)

# ml с кодером soft
        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=20,
            flag_code=True
        )

        H, Y_OFDM = qam.start()

        y_ml = qam.ml_equalize(H, Y_OFDM)
        y_zf = []
        y_mmse = []

        data = {'zf': y_zf, 'mmse': y_mmse, 'ml': y_ml}

        num_err_ml, num_bits_ml = qam.BER_new(list(data.items())[2],
                                                H,
                                                num_err=0,
                                                num_bits=0,
                                                count=0,
                                                type_demod='soft')
        print(f'ml с кодером soft: num_err = {num_err_ml}, num_bits = {num_bits_ml}')
        BER_ml_list_code_soft.append(num_err_ml / num_bits_ml)



    ber_zf.append(np.mean(np.array(BER_zf_list)))
    ber_zf_code.append(np.mean(np.array(BER_zf_list_code)))
    ber_zf_code_soft.append(np.mean(np.array(BER_zf_list_code_soft)))

    evm_zf.append(np.mean(np.array(EVM_zf_list)))



    ber_mmse.append(np.mean(np.array(BER_mmse_list)))
    ber_mmse_code.append(np.mean(np.array(BER_mmse_list_code)))
    ber_mmse_code_soft.append(np.mean(np.array(BER_mmse_list_code_soft)))

    evm_mmse.append(np.mean(np.array(EVM_mmse_list)))



    ber_ml.append(np.mean(np.array(BER_ml_list)))
    ber_ml_code.append(np.mean(np.array(BER_ml_list_code)))
    ber_ml_code_soft.append(np.mean(np.array(BER_ml_list_code_soft)))

ber_zf = np.array(ber_zf)
ber_zf_code = np.array(ber_zf_code)
ber_zf_code_soft = np.array(ber_zf_code_soft)
evm_zf = np.array(evm_zf)

ber_mmse = np.array(ber_mmse)
ber_mmse_code = np.array(ber_mmse_code)
ber_mmse_code_soft = np.array(ber_mmse_code_soft)
evm_mmse = np.array(evm_mmse)

ber_ml = np.array(ber_ml)
ber_ml_code = np.array(ber_ml_code)
ber_ml_code_soft = np.array(ber_ml_code_soft)


plt.figure(figsize=(10, 6))

plt.suptitle(f'реализаций: {N_avg}')

# BER:

# ZF
plt.subplot(2, 2, 1)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("BER", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, ber_zf, color="red")
plt.semilogy(SNR_dB_set, ber_zf_code, color="red", linestyle='--')
plt.semilogy(SNR_dB_set, ber_zf_code_soft, color="red", linestyle='-.')
plt.legend(labels=["zf", "zf_code", "zf_code_soft"], fontsize=8)

# MMSE
plt.subplot(2, 2, 2)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("BER", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, ber_mmse, color="green")
plt.semilogy(SNR_dB_set, ber_mmse_code, color="green", linestyle='--')
plt.semilogy(SNR_dB_set, ber_mmse_code_soft, color="green", linestyle='-.')
plt.legend(labels=["mmse", "mmse_code", "mmse_code_soft"], fontsize=8)

# ML
plt.subplot(2, 2, 3)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("BER", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, ber_ml, color="magenta")
plt.semilogy(SNR_dB_set, ber_ml_code, color="magenta", linestyle='--')
plt.semilogy(SNR_dB_set, ber_ml_code_soft, color="magenta", linestyle='-.')
plt.legend(labels=["ml", "ml_code", "ml_code_soft"], fontsize=8)

# EVM:

plt.subplot(2, 2, 4)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("EVM", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, evm_zf, color="red")
plt.semilogy(SNR_dB_set, evm_mmse, color="green")
plt.legend(labels=["zf", "mmse"], fontsize=8)

plt.tight_layout()

plt.savefig('llr without ml')

plt.show()
