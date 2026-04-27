
# Краткое описание проекта

Проект представляет собой моделирование канала с Рэлеевским замиранием и АБГШ в условиях MIMO-OFDM передачи

# Структура директорий проекта


```
OFDM_project
├─ BPSK
├─ Householder
├─ OFDM_MIMO_2x2
│  ├─ MIMO_OFDM
│  ├─ MIMO_OFDM_viterbi
│  ├─ mmse_householder
│  └─ OFDM_multiprocessing
├─ QAM16_without_OFDM
├─ QAM_with_OFDM
└─ README.md

```

1. **BPSK** - реализация BPSK модуляции и расчет характеристик BER и EVM для Zero-Forcing (далее ZF) и Minimum Mean Square Error (далее MMSE) эквализаций 
2. **Householder** - QR-разложение матрицы $A \in \mathbb{C}^{m \times n}$
3. **OFDM_MIMO_2x2** - переход к от SISO к MIMO 2x2
4. **QAM16_without_OFDM** - реализация QAM16 модуляции и расчет характеристик BER и EVM для ZF и MMSE эквализаций
5. **QAM_with_OFDM** - реализация QAM16 модуляции и расчет характеристик BER и EVM для ZF и MMSE эквализаций
