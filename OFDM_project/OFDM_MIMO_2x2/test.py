import time
from concurrent.futures import ProcessPoolExecutor

# 1. Функция, которую мы хотим распараллелить
def heavy_computation(n):
    # Имитируем тяжелые вычисления (например, как ваш расчет BER)
    # Просто подождем 1 секунду
    print(n, end=' ')
    for _ in range(199990000):
        x = n * n  # Некоторая "тяжелая" операция
    # time.sleep(2) 
    return n * n

if __name__ == "__main__":
    # Список входных данных (например, ваши итерации N_avg)
    numbers = [1, 2, 3, 4, 5]
    
    print(f"Запуск на процессоре с {numbers} задачами...")

    # --- ВАРИАНТ 1: ПОСЛЕДОВАТЕЛЬНО (как у вас сейчас) ---
    start = time.perf_counter()
    results_sync = []
    for num in numbers:
        results_sync.append(heavy_computation(num))
        
    end = time.perf_counter()
    print(f"Последовательно: {end - start:.2f} сек. Результат: {results_sync}")

    # --- ВАРИАНТ 2: ПАРАЛЛЕЛЬНО (multiprocessing) ---
    start = time.perf_counter()
    
    # Создаем "пул" из 4 процессов
    with ProcessPoolExecutor(max_workers=2) as executor:
        # executor.map автоматически распределяет список 'numbers' 
        # между свободными процессами
        results_parallel = list(executor.map(heavy_computation, numbers))
        
    end = time.perf_counter()
    print(f"Параллельно: {end - start:.2f} сек. Результат: {results_parallel}")