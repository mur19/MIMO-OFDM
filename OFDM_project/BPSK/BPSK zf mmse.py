import numpy as np
import matplotlib.pyplot as plt


def BPSK():
    x = []
    for i in bit_input:
        if i == 0:
            x.append(-1 + 0j)
        else:
            x.append(1 + 0j) 
    return np.array(x)


def AWGN(sigma):
    awgn = np.array(np.random.normal(0, sigma, num_symbols)) + np.array(np.random.normal(0, sigma, num_symbols)) * 1j
    return awgn


def power_of_noise():
    P_noise = np.mean(np.abs(noise)**2)
    return P_noise


def power_of_signal():
    P_signal = np.mean(np.abs(x)**2)
    return P_signal


def zf_equalize(r):
    return r/H


def mmse_equalize(r):
    return r*np.conj(H)/(np.abs(H)**2+1/SNR)


def demodulation(equalized_signal):
    bit_output = []
    for i in equalized_signal:
        if np.real(i) > 0:
            bit_output.append(1)
        else:
            bit_output.append(0)
    return np.array(bit_output)


def BER(equalized_signal):
    return np.sum((bit_input + demodulation(equalized_signal)) % 2)/len(bit_input)


def EVM(equalized_signal):
    return np.sum(np.abs(x - equalized_signal)**2)/num_symbols


num_symbols = 10000

bit_input = np.random.randint(0, 2, num_symbols) 

x = BPSK()

H = -0.5 - 1j

y_clear = H*x

SNR_dB = 3

SNR = 10 ** (SNR_dB/10)

sigma = np.sqrt(power_of_signal()/(2*SNR))

noise = AWGN(sigma)

P_signal = power_of_signal()
P_noise = power_of_noise()

y = H*x + noise

y_zf = zf_equalize(y)
y_mmse = mmse_equalize(y)

print(f'BER_zf( SNR = {SNR_dB} dB ) = {BER(y_zf)*100}%')
print(f'BER_mmse( SNR = {SNR_dB} dB ) = {BER(y_mmse)*100}%')



plt.figure(figsize=(10,6))

plt.subplot(2, 2, 1)
plt.title('modulated signal', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim(-1.5, 1.5)
plt.ylim(-2, 2)
plt.scatter(np.real(x), np.imag(x), color='red', s=20)



plt.subplot(2, 2, 2)
plt.title(f'signal after channel (H={H})', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim(-1.5, 1.5)
plt.ylim(-2, 2)
plt.scatter(np.real(y_clear), np.imag(y_clear), color='black', s=10)
plt.scatter(np.real(x), np.imag(x), color='red', s=20)



plt.subplot(2, 2, 3)
plt.title(f'signal after channel with AWGN (SNR={SNR_dB} dB)', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim()
plt.ylim()
plt.scatter(np.real(y), np.imag(y), color='black', s=1)
plt.scatter(np.real(x), np.imag(x), color='red', s=20)



plt.subplot(2, 2, 4)
plt.title('equalized signal', fontsize=12)
plt.xlabel('I', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.xlim()
plt.ylim(-2,2)
plt.scatter(np.real(y_zf), np.imag(y_zf), s=1, color='blue')
plt.scatter(np.real(y_mmse), np.imag(y_mmse), s=1, color='green')
plt.scatter(np.real(x), np.imag(x), color='red', s=20)
plt.legend(labels=['ZF', 'MMSE'], fontsize=8)

plt.tight_layout()
plt.show()


SNR_dB = np.arange(-20, 6, 0.2)

ber_zf, evm_zf = [], []
ber_mmse, evm_mmse = [], []

for SNR_dB_i in SNR_dB:

    num_symbols = 10000

    bit_input = np.random.randint(0, 2, num_symbols) 

    x = BPSK()

    H = -0.5 - 1j

    y_clear = H*x

    SNR = 10 ** (SNR_dB_i/10)

    sigma = np.sqrt(power_of_signal()/(2*SNR))

    noise = AWGN(sigma)

    P_signal = power_of_signal()
    P_noise = power_of_noise()

    y = H*x + noise

    y_zf = zf_equalize(y)
    y_mmse = mmse_equalize(y)

    ber_zf.append(BER(y_zf))
    evm_zf.append(EVM(y_zf))

    ber_mmse.append(BER(y_mmse))
    evm_mmse.append(EVM(y_mmse))

    
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
plt.semilogy(SNR_dB, ber_zf, color='red')
plt.semilogy(SNR_dB, ber_mmse, color='green')
plt.legend(labels=['zf', 'mmse'], fontsize=8)

plt.subplot(1, 2, 2)
plt.title('', fontsize=12)
plt.xlabel('SNR_dB', fontsize=12)
plt.ylabel('EVM', fontsize=12)
plt.xlim()
plt.ylim()
plt.semilogy(SNR_dB, evm_zf, color='red')
plt.semilogy(SNR_dB, evm_mmse, color='green')
plt.legend(labels=['zf', 'mmse'], fontsize=8)

plt.tight_layout()
plt.show()

