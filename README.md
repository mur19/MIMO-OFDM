
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
2. **Householder** - QR-разложение матрицы $A \in \mathbb{C}^{m \times n}$ с использованием преобразования Хаусхолдера (метод используется для решения задачи MMSE-эквализации)
3. **OFDM_MIMO_2x2** - переход к от SISO к MIMO 2x2
4. **QAM16_without_OFDM** - реализация QAM16 модуляции и расчет характеристик BER и EVM для ZF и MMSE эквализаций
5. **QAM_with_OFDM** - реализация QAM16 модуляции и расчет характеристик BER и EVM для ZF и MMSE эквализаций с OFDM

# Математическая модель

## 1. Модель канала

Для каждой поднесущей OFDM сигнал в частотной области описывается базовым уравнением:

$$
y = Hx+n
$$

где:

 $y \in \mathbb{C}^{2 \times 1}$ - принятый сигнал

 $x \in \mathbb{C}^{2 \times 1}$ - переданный сигнал
 
 $H \in \mathbb{C}^{2 \times 2}$ - передаточная функция канала
 
 $n_i \sim \mathcal{N}(0, \sigma_n^2)$ - АБГШ

 ## 2. Эквализация

 ### Zero-Forcing (ZF)

 цель: обращение матрицы канала
 
 решение :

 $$
 \hat{x} = (H^{H}H)^{-1}H^Hy = H^{-1}y
 $$

 последнее равенство имеет место быть если $H$ - квадратная матрица

 ### Minimum Mean Square Error (MMSE)

 #### Классическое решение

 цель: минимизация среднеквадратичной ошибки

 $$
 J(W) = E[\Vert x-\hat{x}\Vert ^2] = E[\Vert x-Wy\Vert ^2]
 $$
 
 решение:

 $$
 \hat{x} = W_{MMSE}y = \left(H^HH + \dfrac{1}{SNR_{linear}}I \right)^{-1}H^Hy
 $$

 #### Метод преобразований Хаусхолдера

подробнее про метод и его применение к MMSE описал [здесь](https://www.overleaf.com/read/tzmffpbrjfcc#282156)



