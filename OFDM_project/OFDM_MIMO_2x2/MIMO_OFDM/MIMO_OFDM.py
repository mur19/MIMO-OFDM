import numpy as np
import matplotlib.pyplot as plt
import commpy as cp


class QAMmodulation:
    def __init__(self, M, SNR_dB, subcarriers, OFDM_symbols):
        self.OFDM_symbols = OFDM_symbols
        self.subcarriers = subcarriers
        self.M = M
        self.SNR_dB = SNR_dB
        self.num_symbols = subcarriers * OFDM_symbols * 2
        self.num_bits = int(self.num_symbols * np.log2(M))
        self.bits_input = np.random.randint(0, 2, self.num_bits)
        self.bits_input_1 = self.bits_input[:int(self.num_bits/2)]
        self.bits_input_2 = self.bits_input[int(self.num_bits/2):]
        self.modem = cp.QAMModem(self.M)
        self.x = self.modem.modulate(self.bits_input)
        self.x_1 = self.x[:int(len(self.x)/2)]
        self.x_2 = self.x[int(len(self.x)/2):]
        self.SNR = 10 ** (self.SNR_dB / 10)

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
            # h = np.array([[self.impulse_response(diagram=False) for _ in range(2)] for _ in range(2)])
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

    def mmse_equalize(self, H, Y):

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
        
        return y_mmse

    def demodulation(self, equalized_signal, modem):
        return self.modem.demodulate(equalized_signal, demod_type="hard")

    def BER(self, equalized_signal):
        equalized_signal = equalized_signal.reshape(-1)
        equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())
        demod = self.demodulation(equalized_signal, self.modem)
        return np.sum((self.bits_input + demod) % 2) / self.num_bits

    def EVM(self, equalized_signal):
        equalized_signal = equalized_signal.reshape(-1)
        equalized_signal = np.array(equalized_signal[0::2].tolist() + equalized_signal[1::2].tolist())
        return np.sum(np.abs(self.x - equalized_signal) ** 2) / self.num_symbols
    
    def del_pilots(self, r):
        for i in range(qam.OFDM_symbols):
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



qam = QAMmodulation(
    M=16,
    SNR_dB=20,
    subcarriers=7,
    OFDM_symbols=40,
)

# x1 = qam.x_1
# x2 = qam.x_2
X_OFDM = qam.OFDM()

X_OFDM_pilots = qam.add_pilots(X_OFDM)

x_t_pilots = qam.idft(X_OFDM_pilots) * np.sqrt(qam.subcarriers)
x_t = qam.del_pilots(x_t_pilots)

x_t_vector_pilots = x_t_pilots.reshape(-1)
x_t_vector = x_t.reshape(-1)

y_clear_t_pilots = qam.convolution_cp_MIMO(x_t_pilots) 
y_clear_t = qam.del_pilots(y_clear_t_pilots)

awgn = qam.AWGN(x_t_vector_pilots).reshape(3*qam.OFDM_symbols, qam.subcarriers, 2)
# awgn_zeros = np.zeros_like(awgn)

for i in range(3*qam.OFDM_symbols):
    if i%3 != 0:
        awgn[i] = 0 + 0j


# awgn_zeros[]

y_t_pilots = y_clear_t_pilots + awgn
y_t = qam.del_pilots(y_t_pilots)

Y_OFDM_pilots = qam.dft(y_t_pilots) / np.sqrt(qam.subcarriers)


indices_to_keep = np.arange(0, Y_OFDM_pilots.shape[0], 3)

all_indices = np.arange(Y_OFDM_pilots.shape[0])
indices_to_remove = np.delete(all_indices, indices_to_keep)

H_sum = Y_OFDM_pilots[indices_to_remove, :, :]

Y_OFDM = Y_OFDM_pilots[indices_to_keep, :, :]



H = np.zeros_like(H_sum)

even_idx = np.arange(0, 2*qam.OFDM_symbols, 2)
odd_idx = np.arange(1, 2*qam.OFDM_symbols, 2)

H[even_idx, :, 0] = 2 * H_sum[even_idx, :, 0] - H_sum[odd_idx, :, 0]
H[even_idx, :, 1] = -H_sum[even_idx, :, 0] + H_sum[odd_idx, :, 0]

H[odd_idx, :, 0] = 2 * H_sum[even_idx, :, 1] - H_sum[odd_idx, :, 1]
H[odd_idx, :, 1] = -H_sum[even_idx, :, 1] + H_sum[odd_idx, :, 1]



y_zf = qam.zf_equalize(H, Y_OFDM)
y_mmse = qam.mmse_equalize(H, Y_OFDM)


plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title(f"Modulated signal (QAM{qam.M}), {qam.num_symbols}symbols")
plt.xlim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.ylim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.scatter(np.real(qam.x_1), np.imag(qam.x_1), color="red")

plt.subplot(1, 3, 2)
plt.title(f"Signal after channel with AWGN (SNR={qam.SNR_dB} dB)", fontsize=12)
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.xlim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.ylim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.scatter(np.real(Y_OFDM), np.imag(Y_OFDM), color="black", s=0.5)
plt.scatter(np.real(qam.x_1), np.imag(qam.x_1), color="red", s=20)

plt.subplot(1, 3, 3)
plt.title("Equalized signal", fontsize=12)
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.xlim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.ylim(-2 * max(np.abs(qam.x_1)), 2 * max(np.abs(qam.x_1)))
plt.scatter(np.real(y_zf), np.imag(y_zf), s=0.5, color="blue")
plt.scatter(np.real(y_mmse), np.imag(y_mmse), s=0.5, color="green")
plt.scatter(np.real(qam.x_1), np.imag(qam.x_1), color="red", s=20)
plt.legend(labels=["ZF", "MMSE"], fontsize=10)

plt.tight_layout()
plt.show()


SNR_dB_set = np.arange(0, 30, 2)

ber_zf, evm_zf = [], []
ber_mmse, evm_mmse = [], []

N_avg = 100

for SNR_dB in SNR_dB_set:

    BER_zf_list, EVM_zf_list, BER_mmse_list, EVM_mmse_list = [], [], [], []

    for _ in range(N_avg):

        qam = QAMmodulation(
            M=16,
            SNR_dB=SNR_dB,
            subcarriers=7,
            OFDM_symbols=50,
        )

        # x1 = qam.x_1
        # x2 = qam.x_2
        X_OFDM = qam.OFDM()

        X_OFDM_pilots = qam.add_pilots(X_OFDM)

        x_t_pilots = qam.idft(X_OFDM_pilots) * np.sqrt(qam.subcarriers)
        x_t = qam.del_pilots(x_t_pilots)

        x_t_vector_pilots = x_t_pilots.reshape(-1)
        x_t_vector = x_t.reshape(-1)

        y_clear_t_pilots = qam.convolution_cp_MIMO(x_t_pilots) 
        y_clear_t = qam.del_pilots(y_clear_t_pilots)

        awgn = qam.AWGN(x_t_vector_pilots).reshape(3*qam.OFDM_symbols, qam.subcarriers, 2)

        for i in range(3*qam.OFDM_symbols):
            if i%3 != 0:
                awgn[i] = 0 + 0j

        y_t_pilots = y_clear_t_pilots + awgn
        y_t = qam.del_pilots(y_t_pilots)

        Y_OFDM_pilots = qam.dft(y_t_pilots) / np.sqrt(qam.subcarriers)



        indices_to_keep = np.arange(0, Y_OFDM_pilots.shape[0], 3)

        all_indices = np.arange(Y_OFDM_pilots.shape[0])
        indices_to_remove = np.delete(all_indices, indices_to_keep)

        H_sum = Y_OFDM_pilots[indices_to_remove, :, :]

        Y_OFDM = Y_OFDM_pilots[indices_to_keep, :, :]



        H = np.zeros_like(H_sum)

        even_idx = np.arange(0, 2*qam.OFDM_symbols, 2)
        odd_idx = np.arange(1, 2*qam.OFDM_symbols, 2)

        H[even_idx, :, 0] = 2 * H_sum[even_idx, :, 0] - H_sum[odd_idx, :, 0]
        H[even_idx, :, 1] = -H_sum[even_idx, :, 0] + H_sum[odd_idx, :, 0]

        H[odd_idx, :, 0] = 2 * H_sum[even_idx, :, 1] - H_sum[odd_idx, :, 1]
        H[odd_idx, :, 1] = -H_sum[even_idx, :, 1] + H_sum[odd_idx, :, 1]




        y_zf = qam.zf_equalize(H, Y_OFDM)
        y_mmse = qam.mmse_equalize(H, Y_OFDM)

        BER_zf_list.append(qam.BER(y_zf))
        EVM_zf_list.append(qam.EVM(y_zf))

        BER_mmse_list.append(qam.BER(y_mmse))
        EVM_mmse_list.append(qam.EVM(y_mmse))

    ber_zf.append(np.mean(np.array(BER_zf_list)))
    evm_zf.append(np.mean(np.array(EVM_zf_list)))

    ber_mmse.append(np.mean(np.array(BER_mmse_list)))
    evm_mmse.append(np.mean(np.array(EVM_mmse_list)))

ber_zf = np.array(ber_zf)
evm_zf = np.array(evm_zf)

ber_mmse = np.array(ber_mmse)
evm_mmse = np.array(evm_mmse)


plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("", fontsize=12)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("BER", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, ber_zf, color="red")
plt.semilogy(SNR_dB_set, ber_mmse, color="green")
plt.legend(labels=["zf", "mmse"], fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("", fontsize=12)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("EVM", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, evm_zf, color="red")
plt.semilogy(SNR_dB_set, evm_mmse, color="green")
plt.legend(labels=["zf", "mmse"], fontsize=8)
plt.grid(True)

plt.savefig('MIMO_OFDM')

plt.tight_layout()
plt.show()
