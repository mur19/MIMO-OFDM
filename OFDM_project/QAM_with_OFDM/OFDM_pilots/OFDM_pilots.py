import numpy as np
import matplotlib.pyplot as plt
import commpy as cp


class QAMmodulation:
    def __init__(self, M, SNR_dB, subcarriers, OFDM_symbols):
        self.OFDM_symbols = OFDM_symbols
        self.subcarriers = subcarriers
        self.M = M
        self.SNR_dB = SNR_dB
        self.num_symbols = subcarriers * OFDM_symbols
        self.num_bits = int(subcarriers * OFDM_symbols * np.log2(M))
        self.bits_input = np.random.randint(0, 2, self.num_bits)
        self.modem = cp.QAMModem(self.M)
        self.x = self.modem.modulate(self.bits_input)
        self.SNR = 10 ** (self.SNR_dB / 10)

    def OFDM(self):
        X_OFDM = self.x.reshape(self.OFDM_symbols, self.subcarriers)
        return X_OFDM
    
    def add_pilots(self, X_OFDM):
        ones = np.ones(self.subcarriers)
        for i in range(self.OFDM_symbols):
            X_OFDM = np.insert(X_OFDM, 2 * i + 1, ones, axis=0)
        return X_OFDM # матрица 2*OFDM x sub

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

    def convolution_cp_matrix(self, x_t):  # возвращает матрицу с cp
        y_clear_t = []
        for i in range(x_t.shape[0]):
            if i%2 == 0:
                h = self.impulse_response(diagram=False)
            OFDM_i = x_t[i]
            y_clear_t = np.append(y_clear_t, self.convolution_cp(OFDM_i, h))
        return y_clear_t.reshape(x_t.shape[0], self.subcarriers + len(h) - 1)

    def AWGN(self, x_t_vector):
        sigma = self.RMSD(x_t_vector)
        re_n = np.array(np.random.normal(0, sigma, int(self.num_symbols)))
        im_n = np.array(np.random.normal(0, sigma, int(self.num_symbols)))
        awgn = re_n + im_n * 1j
        return awgn

    def power_of_noise(self):
        noise = self.AWGN()
        P_noise = np.mean(np.abs(noise) ** 2)
        return P_noise

    def zf_equalize(self, r):
        y_zf = []
        for i in range(self.OFDM_symbols):
            for j in range(self.subcarriers):
                y_zf.append(r[i*self.subcarriers+j]/H[i, j])
        return np.array(y_zf)

    def mmse_equalize(self, r):
        y_mmse = []
        for i in range(self.OFDM_symbols):
            for j in range(self.subcarriers):
                y_mmse.append(
                    r[i * self.subcarriers + j]
                    * np.conj(H[i, j])
                    / (np.abs(H[i, j]) ** 2 + 1 / self.SNR)
                )
        return np.array(y_mmse)

    def demodulation(self, equalized_signal, modem):
        return self.modem.demodulate(equalized_signal, demod_type="hard")

    def BER(self, equalized_signal):
        return np.sum((self.bits_input + self.demodulation(equalized_signal, self.modem)) % 2) / self.num_bits

    def EVM(self, equalized_signal):
        return np.sum(np.abs(self.x - equalized_signal) ** 2) / self.num_symbols
    
    def del_pilots(self, r):
        for i in range(qam.OFDM_symbols):
            r = np.delete(r, i + 1, axis=0)
        return r

    def impulse_response(self, diagram):
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
    SNR_dB=3,
    subcarriers=7,
    OFDM_symbols=50,
)


X_OFDM = qam.OFDM()

X_OFDM_pilots = qam.add_pilots(X_OFDM)

x_t_pilots = np.fft.ifft(X_OFDM_pilots) * np.sqrt(qam.subcarriers)  # матрица строки которой = ifft(OFDM символ)
x_t = qam.del_pilots(x_t_pilots)

x_t_vector_pilots = x_t_pilots.reshape(-1)  # перевод в вектор
x_t_vector = x_t.reshape(-1)

y_clear_t_pilots = qam.convolution_cp_matrix(x_t_pilots)  # матрица с первыми (len(h) - 1) столбцами cp
y_clear_t = qam.del_pilots(y_clear_t_pilots)

awgn = qam.AWGN(x_t_vector)  # вектор шума OFDM_symbols x subcarriers

awgn = awgn.reshape(qam.OFDM_symbols, qam.subcarriers)

zeros = np.zeros(qam.subcarriers)

for i in range(qam.OFDM_symbols):
    awgn = np.insert(awgn, 2 * i + 1, zeros, axis=0)

y_t_pilots = y_clear_t_pilots[:, len(qam.impulse_response(diagram=False)) - 1 :] + awgn#/np.sqrt(qam.subcarriers)
y_t = qam.del_pilots(y_t_pilots)

Y_OFDM_pilots = np.fft.fft(y_t_pilots) / np.sqrt(qam.subcarriers)
Y_OFDM = Y_OFDM_pilots

H = np.array([], dtype=complex)

for i in range(qam.OFDM_symbols):
    H = np.hstack((H, Y_OFDM[i + 1]))
    Y_OFDM = np.delete(Y_OFDM, i + 1, axis=0)

H = H.reshape(qam.OFDM_symbols, qam.subcarriers)
#print(H)

y = Y_OFDM.reshape(-1)

y_zf = qam.zf_equalize(y)
y_mmse = qam.mmse_equalize(y)

# time modeling

F = 105000
fc = 3*10**5
fs = 15000
T = 1/fs
fd = 15*10**6
td = 1/fd

Tds = np.arange(0, qam.OFDM_symbols*T, td)
tds = np.arange(0, T, td)

if len(Tds) != len(tds) * qam.OFDM_symbols:
    Tds = Tds[:len(tds) * qam.OFDM_symbols]

s, sn = np.array([]), np.array([])

for k in range(qam.OFDM_symbols):
    t_i = 0
    for _ in range(len(tds)):
        signal = 0
        signal_noise = 0
        for n in range(qam.subcarriers):
            signal += np.real(x_t[k, n] * np.exp(1j * 2 * np.pi * (fc + n * fs) * t_i))
            signal_noise += np.real(y_t[k, n] * np.exp(1j * 2 * np.pi * (fc + n * fs) * t_i))
        s = np.append(s, signal)
        sn = np.append(sn, signal_noise)
        t_i += td


plt.figure(figsize=(10, 6))

plt.title(f'сигнал в канале (полоса = {F/10**3} кГц, расстояние между поднесущими = {fs/10**3} кГц, несущая = {fc/10**3} кГц)')
plt.xlabel("t (мс)", fontsize=12)
plt.ylabel("s(t)", fontsize=12)
plt.plot(Tds * 10**3, np.array(s), color="red")
# plt.plot(Tds * 10**3, np.array(sn), color="black")
plt.legend(labels=["в начале канала без шума", "на приеме с шумом"])
plt.show()



plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title(f"Modulated signal (QAM{qam.M}), {qam.num_symbols}symbols")
plt.xlim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.ylim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.scatter(np.real(qam.x), np.imag(qam.x), color="red")

plt.subplot(1, 3, 2)
plt.title(f"Signal after channel with AWGN (SNR={qam.SNR_dB} dB)", fontsize=12)
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.xlim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.ylim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.scatter(np.real(y), np.imag(y), color="black", s=0.5)
plt.scatter(np.real(qam.x), np.imag(qam.x), color="red", s=20)

plt.subplot(1, 3, 3)
plt.title("Equalized signal", fontsize=12)
plt.xlabel("I", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.xlim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.ylim(-2 * max(np.abs(qam.x)), 2 * max(np.abs(qam.x)))
plt.scatter(np.real(y_zf), np.imag(y_zf), s=0.5, color="blue")
plt.scatter(np.real(y_mmse), np.imag(y_mmse), s=0.5, color="green")
plt.scatter(np.real(qam.x), np.imag(qam.x), color="red", s=20)
plt.legend(labels=["ZF", "MMSE"], fontsize=8)

plt.tight_layout()
plt.show()


SNR_dB_set = np.arange(0, 30, 2)

ber_zf, evm_zf = [], []
ber_mmse, evm_mmse = [], []

N_avg = 6

for SNR_dB in SNR_dB_set:

    BER_zf_list, EVM_zf_list, BER_mmse_list, EVM_mmse_list = [], [], [], []

    for _ in range(N_avg):

        qam = QAMmodulation(M=16, SNR_dB=SNR_dB, subcarriers=7, OFDM_symbols=50)

        X_OFDM = qam.OFDM()

        X_OFDM_pilots = qam.add_pilots(X_OFDM)

        x_t_pilots = np.fft.ifft(X_OFDM_pilots) * np.sqrt(qam.subcarriers)  # матрица строки которой = ifft(OFDM символ)
        x_t = qam.del_pilots(x_t_pilots)

        x_t_vector_pilots = x_t_pilots.reshape(-1)  # перевод в вектор
        x_t_vector = x_t.reshape(-1)

        y_clear_t_pilots = qam.convolution_cp_matrix(x_t_pilots)  # матрица с первыми (len(h) - 1) столбцами cp
        y_clear_t = qam.del_pilots(y_clear_t_pilots)

        awgn = qam.AWGN(x_t_vector)  # вектор шума OFDM_symbols x subcarriers

        awgn = awgn.reshape(qam.OFDM_symbols, qam.subcarriers)

        zeros = np.zeros(qam.subcarriers)

        for i in range(qam.OFDM_symbols):
            awgn = np.insert(awgn, 2 * i + 1, zeros, axis=0)

        y_t_pilots = y_clear_t_pilots[:, len(qam.impulse_response(diagram=False)) - 1 :] + awgn#/np.sqrt(qam.subcarriers)
        y_t = qam.del_pilots(y_t_pilots)

        Y_OFDM_pilots = np.fft.fft(y_t_pilots) / np.sqrt(qam.subcarriers)
        Y_OFDM = Y_OFDM_pilots

        H = np.array([], dtype=complex)

        for i in range(qam.OFDM_symbols):
            H = np.hstack((H, Y_OFDM[i + 1]))
            Y_OFDM = np.delete(Y_OFDM, i + 1, axis=0)

        H = H.reshape(qam.OFDM_symbols, qam.subcarriers)

        y = Y_OFDM.reshape(-1)

        y_zf = qam.zf_equalize(y)
        y_mmse = qam.mmse_equalize(y)

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

plt.subplot(1, 2, 2)
plt.title("", fontsize=12)
plt.xlabel("SNR_dB", fontsize=12)
plt.ylabel("EVM", fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, evm_zf, color="red")
plt.semilogy(SNR_dB_set, evm_mmse, color="green")
plt.legend(labels=["zf", "mmse"], fontsize=8)

plt.tight_layout()
plt.show()
