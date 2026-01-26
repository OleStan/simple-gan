# Повний посібник з конвеєра: Генерація профілів вихрових струмів за допомогою GAN

**Версія**: 1.0  
**Дата**: 24 січня 2026  
**Мета**: Генерувати реалістичні профілі електропровідності (σ) і магнітної проникності (μ) за глибиною для вихрострумового контролю

---

## Зміст
1. [Огляд](#огляд)  
2. [Конвеєр генерації даних](#конвеєр-генерації-даних)  
3. [Початковий підхід WGAN](#початковий-підхід-wgan)  
4. [Покращений підхід WGAN](#покращений-підхід-wgan)  
5. [Навчання та чекпоїнти](#навчання-та-чекпоїнти)  
6. [Порівняння та оцінювання](#порівняння-та-оцінювання)  
7. [Майбутні покращення](#майбутні-покращення)

---

## Огляд

### Формулювання задачі
Вихрострумовий контроль потребує реалістичних профілів властивостей матеріалу, що змінюються з глибиною. Потрібно генерувати:
- **σ (сигма)**: електропровідність [См/м], діапазон 1e6–6e7
- **μ (мю)**: відносна магнітна проникність, діапазон 1–100

Профілі описують зміну властивостей від поверхні (r=0) до глибини (r=r_max).

### Архітектура конвеєра
```
┌─────────────────────────────────────────────────────────────┐
│                     ПОВНИЙ КОНВЕЄР                           │
├─────────────────────────────────────────────────────────────┤
│  1. ГЕНЕРАЦІЯ ДАНИХ (eddy_current_data_generator)            │
│     ├─ Roberts Sequence → квазі-випадкова вибірка            │
│     ├─ Profile Functions → σ(r) та μ(r)                      │
│     ├─ Дискретизація → 50 шарів                              │
│     └─ Output: X_raw.npy [N × 100]                           │
│                                                             │
│  2. ПОЧАТКОВИЙ WGAN (train_dual_wgan.py)                     │
│     ├─ MLP dual-head generator                               │
│     ├─ Fully connected critic                                │
│     └─ Pure adversarial loss                                 │
│                                                             │
│  3. ПОКРАЩЕНИЙ WGAN (train_improved_wgan.py)                 │
│     ├─ 1D Convolutional generator                            │
│     ├─ Convolutional critic                                  │
│     ├─ Physics-informed loss                                 │
│     └─ Quality metrics tracking                              │
│                                                             │
│  4. ПОРІВНЯННЯ (compare_approaches.py)                       │
│     ├─ Statistical analysis                                  │
│     ├─ Quality metrics                                       │
│     └─ Visual comparison                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Конвеєр генерації даних

### Модуль: `eddy_current_data_generator`
Розташування: `eddy_current_data_generator/`

### Компоненти

#### 1) Послідовність Робертса (`core/roberts_sequence.py`)
**Призначення**: квазі-випадкова вибірка у 4D просторі параметрів.

**Ключова функція**: `generate_roberts_plan(N, bounds)`
```python
# Generates N samples in 4D space: (σ₁, σ₂, μ₁, μ₂)
# Uses Roberts sequence for better space coverage than random sampling
```
**Параметри**:
- `N`: кількість зразків (напр., 2000)
- `bounds`: діапазони для σ₁, σ₂, μ₁, μ₂

**Чому Roberts sequence?**
- краще “заповнення” простору, ніж uniform random
- менше кластеризації
- більш різноманітне покриття параметрів

#### 2) Функції профілів (`core/material_profiles.py`)
**Призначення**: опис того, як σ і μ змінюються з глибиною.

**Типи профілів**:
- **Linear**: `p(r) = p₁ + (p₂ - p₁) × (r/r_max)`
- **Exponential**: `p(r) = p₁ + (p₂ - p₁) × (1 - exp(-a×r/r_max))`
- **Power**: `p(r) = p₁ + (p₂ - p₁) × (r/r_max)^b`
- **Sigmoid**: `p(r) = p₁ + (p₂ - p₁) / (1 + exp(-c×(r/r_max - g)))`

**Ключова функція**: `make_profile(r, kind, p1, p2, r_max, shape_params)`

#### 3) Дискретизація (`core/dataset_builder.py`)
**Призначення**: перетворити неперервні профілі у дискретні шари.

**Ключова функція**: `build_dataset(config)`

**Кроки**:
1. Згенерувати N наборів параметрів (Roberts sequence)
2. Для кожного набору:
   - створити σ(r) і μ(r) із випадковими типами профілю
   - дискретизувати до K=50 шарів
   - конкатенувати у вектор ознак довжини 100
3. Зберегти `X_raw.npy` [N × 100]

**Конфігурація (`DatasetConfig`)**:
```python
{
    'N': 2000,
    'K': 50,
    'r_max': 1e-3,
    'bounds': {...},
    'profile_types': [...]
}
```

### Вихідні файли
- `X_raw.npy`: сирі дані профілів [2000 × 100]
- `normalization_params.json`: статистика для денормалізації
- `metadata.json`: конфігурація датасету

---

## Початковий підхід WGAN

### Файли: `train_dual_wgan.py`, `wgan_dual_profiles.py`

### Архітектура
- **Генератор**: DualHeadGenerator (спільний енкодер + дві “голови” для σ і μ)
- **Критик**: повнозв’язна мережа, вихід — скалярна оцінка (WGAN)

(Схеми шарів і кодові фрагменти в оригіналі залишаються без змін — це технічні специфікації.)

### Функції втрат (WGAN‑GP)
```python
L_C = -E[C(x_real)] + E[C(x_fake)] + λ_gp × GP
L_G = -E[C(G(z))]
GP  = E[(||∇C(x̂)||₂ - 1)²]
```

### Нормалізація
```python
σ_norm = 2 × (σ - σ_min) / (σ_max - σ_min) - 1
μ_norm = 2 × (μ - μ_min) / (μ_max - μ_min) - 1
```

### Переваги
- стабільне навчання (WGAN‑GP)
- вловлює кореляції σ–μ (dual-head)
- коректна реалізація градієнтного штрафу

### Обмеження
- MLP ігнорує “послідовні” (просторові) зв’язки між шарами
- немає фізичних обмежень гладкості/реалістичності
- слабке оцінювання (лише GAN loss)

---

## Покращений підхід WGAN

### Файли: `train_improved_wgan.py`, `wgan_improved.py`

### Основні нововведення
1) **1D згорткова архітектура** (генератор і критик) для кращої просторової узгодженості профілю  
2) **Physics‑informed loss**: гладкість + штраф за вихід за межі  
3) **Метрики якості**: smoothness, monotonicity, diversity  
4) **Посилений моніторинг** і диференційовані LR (critic вищий)

### Physics‑informed loss
```python
L_G = L_adversarial + λ_physics × L_physics
L_physics = λ_smooth × L_smoothness + λ_bounds × L_bounds
```

### Очікувані покращення (приклад)
| Метрика | Початковий | Покращений | Приріст |
|--------|------------|------------|--------|
| Smoothness | ~0.75 | ~0.92 | +23% |
| KS (σ) | ~0.15 | ~0.08 | -47% |
| KS (μ) | ~0.12 | ~0.06 | -50% |
| Diversity | ~2.5 | ~3.2 | +28% |
| Monotonicity | ~0.15 | ~0.45 | +200% |

---

## Навчання та чекпоїнти

### Запуск навчання
```bash
python train_dual_wgan.py
python train_improved_wgan_resumable.py --epochs 500 --checkpoint_freq 50
```

### Відновлення навчання
```bash
python train_improved_wgan_resumable.py   --resume results/improved_wgan_XXX/checkpoints/checkpoint_latest.pth   --output_dir results/improved_wgan_XXX   --epochs 500
```

**Відновлюється**: ваги моделей, стани оптимізаторів, історія навчання, номер епохи.

---

## Порівняння та оцінювання

### Файл: `compare_approaches.py`

**Запуск**:
```bash
python compare_approaches.py <original_dir> <improved_dir>
```

**Компоненти**:
- генерація зразків (напр., 500)
- KS‑тест, різниця середніх/STD, покриття діапазону
- фізична правдоподібність (гладкість, монотонність, градієнти)
- візуалізації (оверлеї, гістограми, бар‑чарти метрик)

---

## Майбутні покращення

### Короткострокові (1–2 місяці)
- **Conditional generation** (керування типом профілю)
- **Validation monitoring** (рання зупинка / контроль перенавчання)
- **Hyperparameter tuning** (λ_physics, λ_smooth, lr_ratio)

### Середньострокові (3–6 місяців)
- **Attention** (фокус на критичних глибинах)
- **Multi‑scale** (coarse→fine)
- **Curriculum learning** (від простого до складного)

### Довгострокові (6–12 місяців)
- **Інтеграція фізичного forward‑model** (замкнений цикл із симуляцією)
- **VAE‑GAN hybrid**
- **Active learning pipeline**

---

## Підсумок

✅ Надійний конвеєр генерації даних (4 типи профілів)  
✅ Початковий WGAN (стабільний baseline)  
✅ Покращений WGAN (з фізичними обмеженнями + кращою архітектурою)  
✅ Інструменти порівняння та оцінювання  
✅ Чекпоїнти та відновлення навчання

---

## Швидка довідка

### Команди
```bash
python train_dual_wgan.py
python train_improved_wgan_resumable.py --epochs 500
python compare_approaches.py <original_dir> <improved_dir>
```

### Структура файлів
```
eddy_current_data_generator/
├── core/
│   ├── roberts_sequence.py
│   ├── material_profiles.py
│   └── dataset_builder.py

wgan_dual_profiles.py
train_dual_wgan.py

wgan_improved.py
train_improved_wgan_resumable.py

compare_approaches.py
```

**Кінець посібника**
