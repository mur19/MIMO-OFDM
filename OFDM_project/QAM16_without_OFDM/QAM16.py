import numpy as np
import matplotlib.pyplot as plt
import commpy as cp

class QAMmodulation:
    
    def __init__(self, M, SNR_dB, num_symbols, H):
        self.M = M
        self.SNR_dB = SNR_dB    
        self.num_symbols = num_symbols
        self.num_bits = int(num_symbols*np.log2(M))
        self.H = H
        self.bits_input = np.random.randint(0, 2, self.num_bits) 
        self.modem = cp.QAMModem(self.M)  
        self.x = self.modem.modulate(self.bits_input)     
        self.SNR = 10 ** (self.SNR_dB/10)


    def power_of_signal(self):
        P_signal = np.mean(np.abs(np.array(list(set(self.x))))**2)
        return P_signal
    
    
    def RMSD(self): #root mean square deviation
        return np.sqrt(self.power_of_signal()/(2*self.SNR))
    

    def AWGN(self):
        sigma = self.RMSD()
        re_n = np.array(np.random.normal(0, sigma, int(self.num_symbols)))
        im_n = np.array(np.random.normal(0, sigma, int(self.num_symbols)))
        awgn = re_n + im_n * 1j
        return awgn


    def power_of_noise(self):
        noise = self.AWGN()
        P_noise = np.mean(np.abs(noise)**2)
        return P_noise


    def zf_equalize(self, r):
        return r/self.H


    def mmse_equalize(self, r):
        return r*np.conj(self.H)/(np.abs(self.H)**2+1/self.SNR)


    def demodulation(self, equalized_signal, modem):
        return self.modem.demodulate(equalized_signal, demod_type='hard')


    def BER(self, equalized_signal):
        return np.sum((self.bits_input + self.demodulation(equalized_signal, self.modem)) % 2)/self.num_bits


    def EVM(self, equalized_signal):
        return np.sum(np.abs(self.x - equalized_signal)**2)/self.num_symbols
    


qam = QAMmodulation(M=16, SNR_dB=15, num_symbols=15000, H = 1+1j)

y_clear = qam.H*qam.x

SNR = 10 ** (qam.SNR_dB/10)

y = y_clear + qam.AWGN()

y_zf = qam.zf_equalize(y)
y_mmse = qam.mmse_equalize(y)

plt.figure(figsize=(10,6))

plt.subplot(2, 2, 1)
plt.title(f'Modulated signal (QAM{qam.M}), {qam.num_symbols}symbols')
plt.xlim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.ylim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.scatter(np.real(qam.x), np.imag(qam.x), color='red')

plt.subplot(2, 2, 2)
plt.title(f'Signal after channel (H={qam.H})', fontsize=12)
plt.xlim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.ylim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.scatter(np.real(y_clear), np.imag(y_clear), color='black', s=1)
plt.scatter(np.real(qam.x), np.imag(qam.x), color='red')

plt.subplot(2, 2, 3)
plt.title(f'Signal after channel with AWGN (SNR={qam.SNR_dB} dB)', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.ylim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.scatter(np.real(y), np.imag(y), color='black', s=1)
plt.scatter(np.real(qam.x), np.imag(qam.x), color='red', s=20)

plt.subplot(2, 2, 4)
plt.title('Equalized signal', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.ylim(-2*max(np.abs(qam.x)), 2*max(np.abs(qam.x)))
plt.scatter(np.real(y_zf), np.imag(y_zf), s=1, color='blue')
plt.scatter(np.real(y_mmse), np.imag(y_mmse), s=1, color='green')
plt.scatter(np.real(qam.x), np.imag(qam.x), color='red', s=20)
plt.legend(labels=['ZF', 'MMSE'], fontsize=8)

plt.tight_layout()
plt.show()


SNR_dB_set = np.arange(-20, 6, 0.2)

ber_zf, evm_zf = [], []
ber_mmse, evm_mmse = [], []

for SNR_dB in SNR_dB_set:

    qam = QAMmodulation(M=16, SNR_dB=SNR_dB, num_symbols=10000, H = -0.5 - 1j)

    y_clear = qam.H*qam.x

    SNR = 10 ** (qam.SNR_dB/10)

    y = y_clear + qam.AWGN()

    y_zf = qam.zf_equalize(y)
    y_mmse = qam.mmse_equalize(y)

    ber_zf.append(qam.BER(y_zf))
    evm_zf.append(qam.EVM(y_zf))

    ber_mmse.append(qam.BER(y_mmse))
    evm_mmse.append(qam.EVM(y_mmse))

    
ber_zf = np.array(ber_zf)
evm_zf = np.array(evm_zf)
ber_mmse = np.array(ber_mmse)
evm_mmse = np.array(evm_mmse)


plt.figure()

plt.subplot(1, 2, 1)
plt.title('', fontsize=12)
plt.xlabel('SNR_dB', fontsize=12)
plt.ylabel('BER', fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, ber_zf, color='red')
plt.semilogy(SNR_dB_set, ber_mmse, color='green')
plt.legend(labels=['zf', 'mmse'], fontsize=8)

plt.subplot(1, 2, 2)
plt.title('', fontsize=12)
plt.xlabel('SNR_dB', fontsize=12)
plt.ylabel('EVM', fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB_set, evm_zf, color='red')
plt.semilogy(SNR_dB_set, evm_mmse, color='green')
plt.legend(labels=['zf', 'mmse'], fontsize=8)

plt.tight_layout()
plt.show()