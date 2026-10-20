"""Builds comparison.ipynb from the cells below (run: uv run python make_notebook.py).

The notebook is the deliverable; this script keeps its source readable and
reviewable. Execute the notebook with:

    uv run jupyter nbconvert --to notebook --execute --inplace comparison.ipynb --ExecutePreprocessor.timeout=3600

``uv run python make_notebook.py --keep-outputs`` rebuilds the notebook and keeps
the outputs of the code cells whose source did not change (for edits of the text).
"""

import sys
from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    r"""
# trosna-kalman в сравнении с эталонными реализациями фильтра Калмана

Ноутбук сравнивает библиотеку [trosna-kalman](https://github.com/89605502155/trosna-kalman)
(фильтр Калмана и адаптивный выбор интервала мониторинга памяти ВМ по записке
А. О. Ферубко и О. Д. Казакова) с эталонными реализациями:

* **листинг A.1 записки** — исходная реализация метода на Python;
* **[filterpy](https://pypi.org/project/filterpy/)** — библиотека к книге R. Labbe
  «Kalman and Bayesian Filters in Python» (источник [7] записки);
* **[pykalman](https://pypi.org/project/pykalman/)** — фильтр, сглаживатель и EM-алгоритм;
* **[statsmodels](https://pypi.org/project/statsmodels/)** — модели в пространстве
  состояний с фильтром на Cython и оценкой параметров методом максимального правдоподобия;
* **[simdkalman](https://pypi.org/project/simdkalman/)** — векторизованный фильтр для
  множества рядов.

Четыре эксперимента проверяют разные свойства:

1. **Согласие и оптимальность** на линейной гауссовой модели с известными параметрами.
2. **Адаптивный опрос** при скачках нагрузки и одинаковом числе опросов — задача из раздела 4 записки.
3. **Офлайн-анализ журнала**: оценка параметров шума и сглаживание по всей записи.
4. **Производительность** на одном ряде, на пакете рядов и на модели с векторным состоянием.

Все случайные данные порождаются с фиксированными зёрнами, поэтому результаты
воспроизводимы. Таблицы результатов сохраняются в папку `results/`, рисунки —
в `figures/` в натуральную ширину 165 мм (на две колонки журнальной полосы):
PDF со встроенными шрифтами, TIFF 600 dpi и PNG для просмотра. Рисунки
рассчитаны на чёрно-белую печать: на них нет сетки и фона, а серии различаются
не только цветом, но и типом линии, маркером или подписью.
"""
)

code(
    r'''
import importlib.metadata as metadata
import io
import math
import os
import platform
import sys
import time
import warnings
from functools import partial
from pathlib import Path

# filterpy 1.4.5 (2018) содержит строки с устаревшими escape-последовательностями
warnings.filterwarnings("ignore", category=SyntaxWarning)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import simdkalman
import statsmodels.api as sm
from filterpy.kalman import KalmanFilter as FilterPyKalman
from filterpy.kalman import rts_smoother
from PIL import Image
from pykalman import KalmanFilter as PyKalman
from scipy import stats

from trosna_kalman import (
    AdaptiveKalmanMonitor,
    HeuristicMonitor,
    KalmanFilter,
    RandomWalkKalmanFilter,
    estimate_measurement_variance,
    estimate_noise_parameters,
)
from trosna_kalman.simulation import (
    adaptive_factory,
    evaluate,
    generate_scenario,
    make_scenarios,
    paper_config,
    simulate,
)
from trosna_kalman.sklearn import KalmanDenoiser

FIGURES = Path("figures")
RESULTS = Path("results")
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("Python", sys.version.split()[0], "|", platform.platform())
print("Процессор:", platform.processor() or platform.machine(), "|", os.cpu_count(), "логических ядер")
for package in ["trosna-kalman", "filterpy", "pykalman", "statsmodels", "simdkalman", "numpy", "scipy", "matplotlib"]:
    print(f"{package:14s} {metadata.version(package)}")
'''
)

code(
    r'''
# Цвета и типы линий закреплены за реализациями во всех рисунках. В чёрно-белой
# печати цвета сливаются, поэтому линии различаются ещё и штрихом.
COLORS = {
    "trosna-kalman": "#2a78d6",
    "filterpy": "#eb6834",
    "pykalman": "#1baf7a",
    "statsmodels": "#eda100",
    "simdkalman": "#e87ba4",
    "листинг A.1": "#008300",
    "эвристика": "#6250d6",
    "истина": "#52514e",
    "замеры": "#b9b8b2",
}
LINESTYLES = {
    "trosna-kalman": "-",
    "filterpy": (0, (5, 1.5, 1, 1.5)),
    "statsmodels": (0, (1, 2)),
    "pykalman": (0, (3, 1, 1, 1, 1, 1)),
    "истина": (0, (6, 2.5)),
}
WIDTH = 165 / 25.4  # ширина рисунка на две колонки полосы, дюймы
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,  # шрифты TrueType встраиваются в PDF
        "savefig.facecolor": "white",
    }
)
MB = 1000.0  # ГБ → МБ


def comma(value, digits=1):
    """Число с десятичной запятой, как принято в русском тексте."""
    return f"{value:.{digits}f}".replace(".", ",")


def plural(n, one, few, many):
    """Согласовать существительное с числом: 1 опрос, 2 опроса, 5 опросов."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


class DecimalComma(mticker.Formatter):
    """Подписи шкалы с десятичной запятой поверх обычного форматтера matplotlib."""

    def __init__(self, inner):
        self.inner = inner

    def set_axis(self, axis):
        super().set_axis(axis)
        self.inner.set_axis(axis)

    def set_locs(self, locs):
        super().set_locs(locs)
        self.inner.set_locs(locs)

    def __call__(self, x, pos=None):
        return self.inner(x, pos).replace(".", ",")

    def get_offset(self):
        return self.inner.get_offset().replace(".", ",")


def save(fig, name):
    """Сохранить рисунок в натуральную величину: PDF, TIFF 600 dpi (RGB) и PNG для просмотра."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, (mticker.ScalarFormatter, mticker.FuncFormatter, mticker.LogFormatter)):
                axis.set_major_formatter(DecimalComma(formatter))
    fig.savefig(FIGURES / f"{name}.pdf", dpi=600)
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=600)
    with Image.open(buffer) as image:
        image.convert("RGB").save(FIGURES / f"{name}.tiff", dpi=(600, 600), compression="tiff_lzw")
    width, height = fig.get_size_inches() * 25.4
    print(f"{name}: {width:.0f} × {height:.0f} мм")


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))
'''
)

md(
    r"""
### Эталонная реализация из записки

Листинг A.1 приведён без изменений, кроме имени класса (`ListingA1` вместо
`AdaptiveKalmanMonitor`, чтобы не путать с классом библиотеки).
"""
)

code(
    r'''
class ListingA1:
    """Одномерный адаптивный фильтр Калмана с выбором интервала опроса."""

    def __init__(self, z0, r, eps, q0=0.006, q_min=0.0015, q_max=0.3,
                 beta=0.3, dt_min=1.0, dt_max=10.0):
        # этап 1: инициализация первым замером
        self.x, self.p, self.q = z0, r, q0
        self.r, self.eps, self.beta = r, eps, beta
        self.q_min, self.q_max = q_min, q_max
        self.dt_min, self.dt_max = dt_min, dt_max

    def update(self, z, dt):
        """Обработать замер z, полученный через dt секунд после предыдущего.

        Возвращает оценку, её стандартное отклонение и интервал
        до следующего опроса.
        """
        nu = z - self.x                                 # этап 3
        q_obs = (nu * nu - self.p - self.r) / dt        # этап 4, формула (6)
        q = (1 - self.beta) * self.q + self.beta * q_obs
        self.q = min(self.q_max, max(self.q_min, q))
        p_prior = self.p + self.q * dt                  # этап 5, формула (1)
        k = p_prior / (p_prior + self.r)                # формула (2)
        self.x += k * nu                                # формула (3)
        self.p = (1 - k) * p_prior                      # формула (4)
        dt_next = (self.eps ** 2 - self.p) / self.q     # этап 7, формула (5)
        dt_next = min(self.dt_max, max(self.dt_min, dt_next))
        return self.x, math.sqrt(self.p), dt_next
'''
)

# ---------------------------------------------------------------------------------------------
md(
    r"""
## Эксперимент 1. Согласие реализаций и оптимальность при известной модели

**Вопрос.** Дают ли все реализации одинаковый результат на одной и той же задаче
и насколько близок он к теоретическому пределу точности?

**Условия.** Потребление памяти — случайное блуждание с интенсивностью
$q = 0{,}001$ ГБ²/с, шум замеров $R = 0{,}08^2$ ГБ² (80 МБ), замер каждую секунду,
2000 секунд. Модель и её параметры известны точно. Все реализации стартуют
одинаково, как на этапе 1 записки: $\hat x = z_0$, $P = R$; далее для каждого
замера выполняются прогноз и коррекция. У pykalman, statsmodels и simdkalman
начальное состояние задаётся как априорное для первого замера, поэтому им
передаётся дисперсия $R + q$ — это тот же старт. Листинг A.1 запускается с
$q_{\min} = q_{\max} = q$: адаптация отключена, и он становится обычным фильтром.
"""
)

code(
    r'''
Q1, R1, N1 = 0.001, 0.08**2, 2000  # ГБ²/с, ГБ², секунд


def random_walk(seed, n=N1, q=Q1, r=R1, start=6.0):
    """Случайное блуждание и его замеры с шумом: (истинный уровень, замеры)."""
    rng = np.random.default_rng(seed)
    level = start + np.cumsum(rng.normal(0.0, math.sqrt(q), n))
    return level, level + rng.normal(0.0, math.sqrt(r), n)


def by_trosna(z, q, r):
    means, _ = RandomWalkKalmanFilter(z[0], r, q, r).filter(z[1:])
    return np.r_[z[0], means]


def by_listing(z, q, r):
    monitor = ListingA1(z[0], r, eps=1.0, q0=q, q_min=q, q_max=q)
    return np.r_[z[0], [monitor.update(value, 1.0)[0] for value in z[1:]]]


def by_filterpy(z, q, r):
    kf = FilterPyKalman(dim_x=1, dim_z=1)
    kf.x[:], kf.P[:], kf.F[:], kf.H[:], kf.Q[:], kf.R[:] = z[0], r, 1.0, 1.0, q, r
    out = [z[0]]
    for value in z[1:]:
        kf.predict()
        kf.update(value)
        out.append(kf.x[0, 0])
    return np.array(out)


def by_pykalman(z, q, r):
    kf = PyKalman(
        transition_matrices=[[1.0]],
        observation_matrices=[[1.0]],
        transition_covariance=[[q]],
        observation_covariance=[[r]],
        initial_state_mean=[z[0]],
        initial_state_covariance=[[r + q]],
    )
    means, _ = kf.filter(z[1:])
    return np.r_[z[0], means[:, 0]]


def by_statsmodels(z, q, r):
    model = sm.tsa.UnobservedComponents(z[1:], level="llevel")
    model.initialize_known(np.array([z[0]]), np.array([[r + q]]))
    model.ssm.tolerance = 0.0  # без перехода на установившийся коэффициент усиления
    result = model.filter(np.array([r, q]))  # sigma2.irregular, sigma2.level
    return np.r_[z[0], result.filtered_state[0]]


def by_simdkalman(z, q, r):
    kf = simdkalman.KalmanFilter(
        state_transition=np.eye(1), process_noise=np.eye(1) * q, observation_model=np.eye(1), observation_noise=r
    )
    result = kf.compute(
        z[1:][None, :], 0, initial_value=np.array([z[0]]), initial_covariance=np.eye(1) * (r + q), filtered=True, smoothed=False
    )
    return np.r_[z[0], result.filtered.states.mean[0, :, 0]]


IMPLEMENTATIONS = {
    "trosna-kalman": by_trosna,
    "листинг A.1": by_listing,
    "filterpy": by_filterpy,
    "pykalman": by_pykalman,
    "statsmodels": by_statsmodels,
    "simdkalman": by_simdkalman,
}

level1, z1 = random_walk(seed=1)
estimates1 = {name: run(z1, Q1, R1) for name, run in IMPLEMENTATIONS.items()}
reference1 = estimates1["trosna-kalman"]
agreement = pd.DataFrame(
    {
        "макс. |разность| с trosna-kalman, ГБ": {
            name: float(np.max(np.abs(estimate - reference1))) for name, estimate in estimates1.items()
        },
        "СКО ошибки оценки, МБ": {name: rmse(estimate - level1) * MB for name, estimate in estimates1.items()},
    }
)
agreement.to_csv(RESULTS / "experiment1_agreement.csv")
agreement
'''
)

md(
    r"""
Теперь точность. Для случайного блуждания установившаяся дисперсия ошибки
оптимального фильтра известна в явном виде:
$P^-_\infty = \bigl(q + \sqrt{q^2 + 4qR}\bigr)/2$, $P_\infty = P^-_\infty R/(P^-_\infty + R)$.
Сравним с ней среднеквадратичную ошибку по 200 независимым рядам (первые
100 секунд переходного процесса отброшены). Для сравнения взяты сырые замеры и
адаптивный монитор, который не знает $q$ и оценивает его по невязкам (формула (6);
опрос каждую секунду, $q_0 = q$, границы $q_0/4$ и $50 q_0$ в тех же отношениях,
что в записке, $\beta = 0{,}3$).
"""
)

code(
    r'''
p_prior = (Q1 + math.sqrt(Q1**2 + 4 * Q1 * R1)) / 2
rmse_theory = math.sqrt(p_prior * R1 / (p_prior + R1))

errors1 = {"сырые замеры": [], "стандартный фильтр (q, R известны)": [], "адаптивный монитор (q оценивается)": []}
burn = slice(100, None)
for seed in range(100, 300):
    level, z = random_walk(seed)
    monitor = AdaptiveKalmanMonitor(z[0], R1, eps=0.1, q0=Q1, q_min=Q1 / 4, q_max=50 * Q1, dt_max=1.0)
    adaptive = np.r_[z[0], [monitor.update(value, 1.0).estimate for value in z[1:]]]
    errors1["сырые замеры"].append(rmse(z[burn] - level[burn]))
    errors1["стандартный фильтр (q, R известны)"].append(rmse(by_trosna(z, Q1, R1)[burn] - level[burn]))
    errors1["адаптивный монитор (q оценивается)"].append(rmse(adaptive[burn] - level[burn]))

accuracy1 = pd.DataFrame(
    {
        "СКО ошибки, МБ": {name: np.mean(values) * MB for name, values in errors1.items()},
        "к теоретическому пределу": {name: np.mean(values) / rmse_theory for name, values in errors1.items()},
    }
)
accuracy1.loc["теоретический предел (уравнение Риккати)"] = [rmse_theory * MB, 1.0]
accuracy1.to_csv(RESULTS / "experiment1_accuracy.csv")
accuracy1.round(3)
'''
)

code(
    r'''
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
window = slice(1000, 1300)
t = np.arange(N1)[window]
ax_left.plot(t, z1[window], ".", color=COLORS["замеры"], markersize=3, label="замеры")
ax_left.plot(t, level1[window], color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.2, label="истинное потребление")
for name in ["trosna-kalman", "filterpy", "statsmodels"]:
    ax_left.plot(t, estimates1[name][window], color=COLORS[name], linestyle=LINESTYLES[name], linewidth=1.6, label=name)
ax_left.set_xlabel("Время, с")
ax_left.set_ylabel("Память ВМ, ГБ")
ax_left.set_title("а) Оценки разных реализаций совпадают")
ax_left.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)

names = list(errors1)
values = [np.mean(errors1[name]) * MB for name in names]
bars = ax_right.barh(range(len(names)), values, color=[COLORS["замеры"], "#6f6e69", COLORS["trosna-kalman"]], height=0.55)
ax_right.axvline(rmse_theory * MB, color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.2)
ax_right.set_ylim(-1.1, len(names) - 0.5)
ax_right.text(rmse_theory * MB + 1.5, -0.85, f"предел Риккати {comma(rmse_theory * MB)} МБ", fontsize=8)
ax_right.set_yticks(range(len(names)), ["сырые замеры", "стандартный\nфильтр", "адаптивный\nмонитор"])
for bar, value in zip(bars, values):
    ax_right.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{comma(value)} МБ", va="center", fontsize=8)
ax_right.set_xlabel("СКО ошибки оценки, МБ")
ax_right.set_title("б) Точность по 200 рядам")
ax_right.set_xlim(0, max(values) * 1.3)
save(fig, "fig1_agreement")
plt.show()
'''
)

md(
    r"""
**Вывод эксперимента 1.** Все пять реализаций дают одну и ту же оценку: у
filterpy, pykalman, simdkalman и листинга A.1 расхождение с trosna-kalman — на
уровне ошибок округления, statsmodels совпадает с той же точностью после
отключения перехода на установившийся коэффициент. При известных $q$ и $R$
стандартный фильтр Калмана достигает теоретического предела точности — его
нельзя превзойти никаким другим линейным способом. Это сильная сторона
классических реализаций: когда модель известна, они оптимальны и проверены
годами использования. Адаптивный монитор, который вынужден оценивать $q$ по
невязкам, здесь уступает оптимуму примерно на 20 %: при замере каждую секунду
мгновенные оценки формулы (6) шумные, и это цена незнания параметров.
"""
)

# ---------------------------------------------------------------------------------------------
md(
    r"""
## Эксперимент 2. Адаптивный опрос при скачках нагрузки и одинаковом числе опросов

**Вопрос.** Записка предлагает опрашивать ВМ тогда, когда неуверенность оценки
достигнет допустимой величины $\varepsilon$ (формула (5)), а интенсивность
изменений оценивать по невязкам (формула (6)). Даёт ли это меньшую ошибку, чем
обычный фильтр Калмана с опросом через постоянный интервал при **том же числе
опросов**?

**Условия** — модельный эксперимент раздела 4.5 записки без изменений:
сценарий длится 300 с, потребление держится около 2 ГБ, в случайный момент
между 75 и 105 с скачком растёт на 0,8–1,1 ГБ, затем растёт со скоростью
4 МБ/с, в случайный момент между 185 и 215 с падает до 2,3–2,5 ГБ; на всё
наложены медленные колебания амплитудой 30 МБ, шум замеров 80 МБ. Все методы
видят одни и те же замеры в одни и те же секунды.

* **Адаптивный монитор trosna-kalman** — параметры записки ($R = 0{,}08^2$,
  $q_0 = 0{,}006$, $q_{\min} = 0{,}0015$, $q_{\max} = 0{,}3$, $\beta = 0{,}3$,
  интервал 1–10 с с округлением до целых секунд) без подбора; меняется только
  $\varepsilon$ — так задаётся число опросов.
* **Листинг A.1** с теми же параметрами — для проверки, что библиотека
  воспроизводит эталон.
* **Стандартный фильтр Калмана (filterpy)** с опросом через постоянный интервал
  $\Delta t = 1, \dots, 10$ с. Для каждого интервала интенсивность $q$ подбирается
  перебором по сетке на **отдельных** 100 обучающих сценариях (наилучшая средняя
  ошибка), а сравнение идёт на других 100 проверочных сценариях. Это лучшая
  возможная настройка базового фильтра.
* **Исходная эвристика системы** (таблица 0 → 10 с … 5 → 1 с по числу
  перераспределений за пять циклов; сырые замеры, окно перераспределения 0,2 ГБ).
"""
)

code(
    r'''
train = make_scenarios(100, seed=2026)  # только для подбора q базового фильтра
test = make_scenarios(100, seed=1016)  # для всех методов


class FilterPyFixedInterval:
    """Стандартный фильтр Калмана filterpy с опросом через постоянный интервал."""

    def __init__(self, z0, q, interval, r=0.08**2):
        self.kf = FilterPyKalman(dim_x=1, dim_z=1)
        self.kf.x[:], self.kf.P[:], self.kf.F[:], self.kf.H[:], self.kf.R[:] = z0, r, 1.0, 1.0, r
        self.q, self.interval = q, float(interval)

    @property
    def estimate(self):
        return float(self.kf.x[0, 0])

    @property
    def std(self):
        return math.sqrt(self.kf.P[0, 0])

    def update(self, z, dt):
        self.kf.predict(Q=np.array([[self.q * dt]]))
        self.kf.update(z)
        return self.estimate, self.std, self.interval


class ListingAsMonitor:
    """Листинг A.1 в интерфейсе монитора (округление интервала выполняет simulate)."""

    def __init__(self, z0, eps):
        self.listing = ListingA1(z0, 0.08**2, eps)
        self.interval = 1.0

    @property
    def estimate(self):
        return self.listing.x

    @property
    def std(self):
        return math.sqrt(self.listing.p)

    def update(self, z, dt):
        x, sigma, self.interval = self.listing.update(z, dt)
        return x, sigma, self.interval


INTERVALS = list(range(1, 11))
Q_GRID = np.geomspace(2e-4, 5e-2, 25)
started = time.perf_counter()
tuned_q = {}
for interval in INTERVALS:
    scores = [evaluate(partial(FilterPyFixedInterval, q=q, interval=interval), train).mean_mae for q in Q_GRID]
    best = int(np.argmin(scores))
    assert 0 < best < len(Q_GRID) - 1, "оптимум на краю сетки"
    tuned_q[interval] = float(Q_GRID[best])
print(f"подбор q для 10 интервалов: {time.perf_counter() - started:.0f} с")
pd.Series(tuned_q, name="подобранное q, ГБ²/с").rename_axis("интервал, с").to_frame().T
'''
)

code(
    r'''
EPS_GRID = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.15, 0.20, 0.30]
adaptive2 = {eps: evaluate(adaptive_factory(paper_config(eps)), test) for eps in EPS_GRID}
fixed2 = {
    interval: evaluate(partial(FilterPyFixedInterval, q=tuned_q[interval], interval=interval), test)
    for interval in INTERVALS
}
heuristic2 = evaluate(partial(HeuristicMonitor, window=0.2), test)
every_second2 = evaluate(adaptive_factory(paper_config(0.1, dt_max=1.0)), test)

# Листинг A.1 с параметрами записки даёт в точности те же результаты, что библиотека.
for eps in (0.10, 0.15):
    listing = evaluate(partial(ListingAsMonitor, eps=eps), test)
    assert np.array_equal(listing.mae, adaptive2[eps].mae) and np.array_equal(listing.n_polls, adaptive2[eps].n_polls)
print("листинг A.1 и trosna-kalman: совпадение по всем 100 сценариям — да")

rows = []
for eps, result in adaptive2.items():
    rows.append({"метод": "адаптивный (trosna-kalman)", "параметр": f"ε = {eps * MB:.0f} МБ", **result.summary()})
for interval, result in fixed2.items():
    rows.append({"метод": "постоянный интервал (filterpy)", "параметр": f"Δt = {interval} с", **result.summary()})
rows.append({"метод": "исходная эвристика", "параметр": "окно 0,2 ГБ", **heuristic2.summary()})
rows.append({"метод": "опрос каждую секунду", "параметр": "ε = 100 МБ", **every_second2.summary()})
pareto2 = pd.DataFrame(rows)
pareto2["mae"] *= MB
pareto2 = pareto2.rename(columns={"polls": "опросов за 300 с", "mae": "ошибка, МБ", "detection_delay": "задержка обнаружения, с"})
pareto2.to_csv(RESULTS / "experiment2_pareto.csv", index=False)
pareto2.round(2)
'''
)

md(
    r"""
Сравним методы при одинаковом числе опросов. Для каждого постоянного
интервала ошибка адаптивного монитора при том же среднем числе опросов
получена линейной интерполяцией по его кривой (только внутри измеренного
диапазона).
"""
)

code(
    r'''
ad_polls = np.array([adaptive2[eps].mean_polls for eps in EPS_GRID])
ad_mae = np.array([adaptive2[eps].mean_mae for eps in EPS_GRID]) * MB
order = np.argsort(ad_polls)
budget_rows = []
for interval, result in fixed2.items():
    polls, mae_fixed = result.mean_polls, result.mean_mae * MB
    inside = ad_polls.min() <= polls <= ad_polls.max()
    mae_adaptive = float(np.interp(polls, ad_polls[order], ad_mae[order])) if inside else float("nan")
    budget_rows.append(
        {
            "интервал, с": interval,
            "опросов за 300 с": polls,
            "постоянный интервал, МБ": mae_fixed,
            "адаптивный при том же числе опросов, МБ": mae_adaptive,
            "разница, %": 100 * (mae_adaptive - mae_fixed) / mae_fixed,
        }
    )
budget2 = pd.DataFrame(budget_rows).set_index("интервал, с")
budget2.to_csv(RESULTS / "experiment2_budget.csv")
budget2.round(1)
'''
)

md(
    r"""
Проверим значимость на двух парах с почти равным числом опросов по 100
проверочным сценариям (парный t-критерий по ошибкам в каждом сценарии).
"""
)

code(
    r'''
significance = []
for eps, interval in [(0.10, 3), (0.15, 7)]:
    a, f = adaptive2[eps], fixed2[interval]
    difference = (a.mae - f.mae) * MB
    low, high = stats.t.interval(0.95, difference.size - 1, loc=difference.mean(), scale=stats.sem(difference))
    significance.append(
        {
            "пара": f"ε = {eps * MB:.0f} МБ против Δt = {interval} с",
            "опросов (адапт. / пост.)": f"{a.mean_polls:.1f} / {f.mean_polls:.0f}",
            "средняя разница ошибок, МБ": difference.mean(),
            "95 % дов. интервал, МБ": f"[{low:.1f}; {high:.1f}]",
            "p-значение": stats.ttest_rel(a.mae, f.mae).pvalue,
            "сценариев, где адаптивный точнее": int((difference < 0).sum()),
        }
    )
pd.DataFrame(significance).set_index("пара").round(4)
'''
)

code(
    r'''
fig, ax = plt.subplots(figsize=(WIDTH, 3.9), layout="constrained")
ax.plot(ad_polls[order], ad_mae[order], "-o", color=COLORS["trosna-kalman"], markersize=5, label="адаптивный монитор trosna-kalman")
fx = np.array([fixed2[i].mean_polls for i in INTERVALS])
fy = np.array([fixed2[i].mean_mae for i in INTERVALS]) * MB
ax.plot(fx, fy, marker="s", linestyle=LINESTYLES["filterpy"], color=COLORS["filterpy"], markersize=5, markerfacecolor="white", label="фильтр filterpy, постоянный интервал (q подобрано)")
ax.plot(heuristic2.mean_polls, heuristic2.mean_mae * MB, "D", color=COLORS["эвристика"], markersize=7, label="исходная эвристика")
ax.axhline(every_second2.mean_mae * MB, color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.0)
ax.text(31, every_second2.mean_mae * MB + 0.7, "опрос каждую секунду (300 опросов)", fontsize=8)
for eps, offset in {0.07: (-30, 6), 0.10: (-8, -13), 0.15: (8, -1), 0.30: (-30, -13)}.items():
    r = adaptive2[eps]
    ax.annotate(f"ε = {eps * MB:.0f} МБ", (r.mean_polls, r.mean_mae * MB), textcoords="offset points", xytext=offset, fontsize=8)
for interval, offset in {1: (-12, 6), 2: (4, 4), 3: (4, 4), 5: (4, 4), 7: (-18, -11), 10: (4, 4)}.items():
    r = fixed2[interval]
    ax.annotate(f"{interval} с", (r.mean_polls, r.mean_mae * MB), textcoords="offset points", xytext=offset, fontsize=8)
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ticks = [30, 40, 50, 60, 80, 100, 150, 200, 300]
ax.set_xticks(ticks, [str(tick) for tick in ticks])
ax.set_xlabel("Среднее число опросов за 300 с (логарифмическая шкала)")
ax.set_ylabel("Средняя абсолютная ошибка, МБ")
ax.set_title("Ошибка оценки и число опросов: 100 проверочных сценариев")
ax.legend(loc="upper right")
save(fig, "fig2_budget")
plt.show()
'''
)

md(
    r"""
Чем объясняется разница, видно на одном сценарии: при близком числе опросов
адаптивный монитор редко опрашивает ВМ в спокойные периоды и часто — сразу
после скачков, а постоянный интервал тратит опросы равномерно.
"""
)

code(
    r'''
scenario = test[3]
runs = {
    "адаптивный, ε = 100 МБ": simulate(scenario.measurements, adaptive_factory(paper_config(0.10))),
    "постоянный интервал 3 с": simulate(scenario.measurements, partial(FilterPyFixedInterval, q=tuned_q[3], interval=3)),
}
fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.6), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 1]})
t = scenario.times
axes[1].sharey(axes[0])
for ax, (name, run), color in zip(axes[:2], runs.items(), [COLORS["trosna-kalman"], COLORS["filterpy"]]):
    ax.plot(t, scenario.truth, color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.1, label="истинное потребление")
    ax.step(t, run.estimates, where="post", color=color, linewidth=1.5, label="оценка между опросами")
    ax.plot(run.poll_times, scenario.measurements[run.poll_times], ".", color=COLORS["замеры"], markersize=5, label="замер при опросе")
    mae = run.mean_absolute_error(scenario.truth) * MB
    ax.set_title(f"{name}: {run.n_polls} {plural(run.n_polls, 'опрос', 'опроса', 'опросов')}, ошибка {mae:.0f} МБ", loc="left")
    ax.set_ylabel("Память ВМ, ГБ")
    for change in scenario.change_times:
        ax.axvline(change, color=COLORS["истина"], linewidth=0.6, alpha=0.5)
axes[0].legend(loc="upper left", ncol=1)
for (name, run), key in zip(runs.items(), ["trosna-kalman", "filterpy"]):
    axes[2].step(run.poll_times, run.intervals, where="post", color=COLORS[key], linestyle=LINESTYLES[key], linewidth=1.4, label=name)
axes[2].set_ylabel("Интервал, с")
axes[2].set_xlabel("Время, с")
axes[2].set_yticks([1, 5, 10])
axes[2].set_ylim(0, 12.5)
axes[2].legend(loc="upper right", ncol=2, fontsize=7.5)
save(fig, "fig3_timeline")
plt.show()

window = 12
share = {}
for name, factory in {
    "адаптивный, ε = 100 МБ": adaptive_factory(paper_config(0.10)),
    "постоянный интервал 3 с": partial(FilterPyFixedInterval, q=tuned_q[3], interval=3),
}.items():
    counts = []
    for s in test:
        run = simulate(s.measurements, factory)
        after = np.zeros(s.truth.size, bool)
        for change in s.change_times:
            after[change : change + window] = True
        counts.append(after[run.poll_times].mean())
    share[name] = np.mean(counts)
print(f"Доля опросов в первые {window} с после скачков (это {2 * window / 300:.0%} времени):")
for name, value in share.items():
    print(f"  {name}: {value:.0%}")
'''
)

md(
    r"""
**Вывод эксперимента 2.** В задаче, для которой предложен метод, адаптивный
выбор интервала при среднем бюджете — от 60 до 150 опросов за 300 с — даёт
ошибку на 5–10 % меньше, чем фильтр Калмана с опросом через постоянный
интервал (при 100 опросах — около 52 против 58 МБ), хотя интенсивность $q$
базового фильтра подобрана наилучшим образом для каждого интервала, а
параметры адаптивного монитора взяты из записки без подбора; при 100 опросах
разница значима (p < 0,001, адаптивный монитор точнее в 76 сценариях из 100).
При малом
бюджете (50 опросов и меньше) преимущество исчезает: постоянный интервал с
подобранным $q$ здесь не хуже, а при 43 опросах точнее в среднем на 2,5 МБ
(около 3 %, разница незначима: p ≈ 0,11), потому что адаптивный монитор
упирается в ограничение $\Delta t_{\max} = 10$ с и дольше не замечает скачок. Механизм выигрыша — перераспределение опросов:
в спокойные периоды монитор опрашивает ВМ редко, а после скачка — каждую
секунду. Обратная сторона видна в таблице: средняя задержка обнаружения
скачка у адаптивного монитора больше, потому что скачок застаёт его на
длинном интервале. Листинг A.1 записки и библиотека дали одинаковые
результаты во всех сценариях; исходная эвристика при близком числе опросов
менее точна, так как работает по сырым замерам.
"""
)

# ---------------------------------------------------------------------------------------------
md(
    r"""
## Эксперимент 3. Офлайн-анализ журнала: оценка параметров и сглаживание

**Вопрос.** Как точно реализации оценивают параметры шума по журналу и
насколько точно восстанавливают историю потребления, если доступна вся запись?

**Условия.** (а) 20 журналов по 1000 замеров (около 17 минут с замером раз в
секунду) случайного блуждания с $q = 0{,}0004$ ГБ²/с и $R = 0{,}08^2$ ГБ².
Параметры оцениваются методом максимального правдоподобия в statsmodels
(диффузная инициализация, численная оптимизация, стандартные ошибки),
EM-алгоритмом в pykalman и функцией `estimate_noise_parameters` библиотеки
trosna-kalman (максимальное правдоподобие и метод моментов). EM сходится
медленно: ему дано 100 итераций, что на этих данных достаточно для
сходимости, а из-за времени работы (около 50 с на журнал) он запускается на
первых 5 журналах; траектория сходимости приведена ниже. (б) 30 сценариев раздела 4.5 с
замером каждую секунду: каузальный фильтр, который использует только прошлые
замеры, против сглаживателя Рауха — Тунга — Штрибеля (RTS), который
использует всю запись. У trosna-kalman сглаживателя нет: библиотека
предназначена для работы в реальном времени.
"""
)

code(
    r'''
Q3, R3, N3 = 0.0004, 0.08**2, 1000
logs = [random_walk(500 + seed, n=N3, q=Q3, r=R3)[1] for seed in range(20)]
EM = "pykalman (EM, 100 итераций)"
estimation = {name: {"q": [], "r": [], "time": []} for name in ["statsmodels (ММП)", "trosna-kalman (ММП)", "trosna-kalman (моменты)", EM]}


def record(name, q_hat, r_hat, started):
    estimation[name]["time"].append(time.perf_counter() - started)
    estimation[name]["q"].append(q_hat)
    estimation[name]["r"].append(r_hat)


standard_errors = None
not_converged = 0
for z in logs:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.tsa.UnobservedComponents(z, level="llevel").fit(disp=False)
    not_converged += any("converge" in str(w.message) for w in caught)
    record("statsmodels (ММП)", fit.params[1], fit.params[0], started)
    if standard_errors is None:
        standard_errors = pd.DataFrame(
            {"оценка": fit.params, "стандартная ошибка": fit.bse}, index=["R (sigma2.irregular)", "q (sigma2.level)"]
        )
    for name, method in [("trosna-kalman (ММП)", "mle"), ("trosna-kalman (моменты)", "moments")]:
        started = time.perf_counter()
        q_hat, r_hat = estimate_noise_parameters(z, method=method)
        record(name, q_hat, r_hat, started)

trajectory = []
for index, z in enumerate(logs[:5]):
    started = time.perf_counter()
    model = PyKalman(
        transition_matrices=[[1.0]],
        observation_matrices=[[1.0]],
        transition_covariance=[[0.01]],
        observation_covariance=[[0.01]],
        initial_state_mean=[z[0]],
        initial_state_covariance=[[0.01]],
        em_vars=["transition_covariance", "observation_covariance"],
    )
    for _ in range(10):  # 10 раз по 10 итераций: на первом журнале записывается траектория
        model = model.em(z, n_iter=10)
        if index == 0:
            trajectory.append((model.transition_covariance[0, 0] / Q3, model.observation_covariance[0, 0] / R3))
    record(EM, model.transition_covariance[0, 0], model.observation_covariance[0, 0], started)

estimation3 = pd.DataFrame(
    {
        name: {
            "журналов": len(values["q"]),
            "q̂/q, среднее": np.mean(values["q"]) / Q3,
            "q̂/q, ст. откл.": np.std(np.array(values["q"]) / Q3),
            "R̂/R, среднее": np.mean(values["r"]) / R3,
            "R̂/R, ст. откл.": np.std(np.array(values["r"]) / R3),
            "время на журнал, с": np.mean(values["time"]),
        }
        for name, values in estimation.items()
    }
).T
estimation3.to_csv(RESULTS / "experiment3_estimation.csv")
display(estimation3.round(3))
gap = np.abs(np.array(estimation["trosna-kalman (ММП)"]["q"]) / np.array(estimation["statsmodels (ММП)"]["q"]) - 1)
print(f"ММП trosna-kalman и statsmodels: наибольшее расхождение оценок q по 20 журналам — {gap.max():.2%}")
print(f"предупреждений statsmodels о несходимости оптимизатора: {not_converged} из {len(logs)}")

print("Сходимость EM на первом журнале (для сравнения ММП statsmodels: "
      f"q̂/q = {estimation['statsmodels (ММП)']['q'][0] / Q3:.3f}, R̂/R = {estimation['statsmodels (ММП)']['r'][0] / R3:.3f}):")
display(pd.DataFrame(trajectory, columns=["q̂/q", "R̂/R"], index=pd.Index(range(10, 101, 10), name="итераций EM")).T.round(3))
print("statsmodels сразу даёт и стандартные ошибки оценок (первый журнал):")
standard_errors
'''
)

code(
    r'''
q_s, r_s = tuned_q[1], 0.08**2  # q, лучшее для опроса каждую секунду (эксперимент 2), и R записки
smoothing = {name: [] for name in ["фильтр trosna-kalman (каузальный)", "адаптивный монитор (каузальный)", "RTS filterpy", "RTS pykalman", "RTS statsmodels", "RTS simdkalman"]}
for s in make_scenarios(30, seed=303):
    z, truth = s.measurements, s.truth
    smoothing["фильтр trosna-kalman (каузальный)"].append(np.mean(np.abs(by_trosna(z, q_s, r_s) - truth)))
    adaptive = simulate(z, adaptive_factory(paper_config(0.1, dt_max=1.0))).estimates
    smoothing["адаптивный монитор (каузальный)"].append(np.mean(np.abs(adaptive - truth)))

    kf = FilterPyKalman(dim_x=1, dim_z=1)
    kf.x[:], kf.P[:], kf.F[:], kf.H[:], kf.Q[:], kf.R[:] = z[0], r_s + q_s, 1.0, 1.0, q_s, r_s
    means, covariances, _, _ = kf.batch_filter(z, update_first=True)
    smoothed, _, _, _ = rts_smoother(means, covariances, np.array([kf.F] * z.size), np.array([kf.Q] * z.size))
    smoothing["RTS filterpy"].append(np.mean(np.abs(smoothed[:, 0, 0] - truth)))

    pk = PyKalman(
        transition_matrices=[[1.0]], observation_matrices=[[1.0]], transition_covariance=[[q_s]],
        observation_covariance=[[r_s]], initial_state_mean=[z[0]], initial_state_covariance=[[r_s + q_s]],
    )
    smoothing["RTS pykalman"].append(np.mean(np.abs(pk.smooth(z)[0][:, 0] - truth)))

    model = sm.tsa.UnobservedComponents(z, level="llevel")
    model.initialize_known(np.array([z[0]]), np.array([[r_s + q_s]]))
    smoothing["RTS statsmodels"].append(np.mean(np.abs(model.smooth(np.array([r_s, q_s])).smoothed_state[0] - truth)))

    sk = simdkalman.KalmanFilter(state_transition=np.eye(1), process_noise=np.eye(1) * q_s, observation_model=np.eye(1), observation_noise=r_s)
    result = sk.compute(z[None, :], 0, initial_value=np.array([z[0]]), initial_covariance=np.eye(1) * (r_s + q_s), smoothed=True)
    smoothing["RTS simdkalman"].append(np.mean(np.abs(result.smoothed.states.mean[0, :, 0] - truth)))

smoothing3 = pd.DataFrame({"средняя абсолютная ошибка, МБ": {name: np.mean(v) * MB for name, v in smoothing.items()}})
smoothing3.to_csv(RESULTS / "experiment3_smoothing.csv")
smoothing3.round(2)
'''
)

code(
    r'''
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
labels = list(estimation)
data = [np.array(estimation[name]["q"]) / Q3 for name in labels]
colors = [COLORS["statsmodels"], COLORS["trosna-kalman"], COLORS["trosna-kalman"], COLORS["pykalman"]]
box = ax_left.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, medianprops={"color": "black"})
for patch, color, hatch in zip(box["boxes"], colors, ["", "", "////", ""]):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
    patch.set_hatch(hatch)
ax_left.axvline(1.0, color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.0)
short = {
    "statsmodels (ММП)": "statsmodels\n(ММП, 20 журналов)",
    "trosna-kalman (ММП)": "trosna-kalman\n(ММП, 20 журналов)",
    "trosna-kalman (моменты)": "trosna-kalman\n(моменты, 20 журналов)",
    EM: "pykalman\n(EM, 5 журналов)",
}
ax_left.set_yticks(range(1, len(labels) + 1), [short[name] for name in labels], fontsize=7.5)
ax_left.set_xlabel("Оценка q / истинное q")
ax_left.set_title("а) Оценка интенсивности q")

s = make_scenarios(30, seed=303)[0]
z, truth = s.measurements, s.truth
window = slice(max(s.jump_time - 25, 0), s.jump_time + 35)
t = s.times[window]
pk = PyKalman(
    transition_matrices=[[1.0]], observation_matrices=[[1.0]], transition_covariance=[[q_s]],
    observation_covariance=[[r_s]], initial_state_mean=[z[0]], initial_state_covariance=[[r_s + q_s]],
)
ax_right.plot(t, z[window], ".", color=COLORS["замеры"], markersize=3, label="замеры")
ax_right.plot(t, truth[window], color=COLORS["истина"], linestyle=LINESTYLES["истина"], linewidth=1.1, label="истинное потребление")
ax_right.plot(t, by_trosna(z, q_s, r_s)[window], color=COLORS["trosna-kalman"], linewidth=1.5, label="фильтр (только прошлое)")
ax_right.plot(t, pk.smooth(z)[0][window, 0], color=COLORS["pykalman"], linestyle=LINESTYLES["pykalman"], linewidth=1.6, label="сглаживание RTS (вся запись)")
ax_right.set_xlabel("Время, с")
ax_right.set_ylabel("Память ВМ, ГБ")
ax_right.set_title("б) Фрагмент вокруг скачка нагрузки")
ax_right.legend(loc="upper left", fontsize=7.5)
save(fig, "fig4_offline")
plt.show()
'''
)

md(
    r"""
**Вывод эксперимента 3.** Для анализа журналов задним числом стандартные
библиотеки дают то, чего в trosna-kalman нет. Сглаживатель RTS использует и
прошлые, и будущие замеры и снижает ошибку восстановления истории потребления
с 40 до 33 МБ (на 17 %) по сравнению с каузальным фильтром; все четыре
реализации сглаживания дают одинаковый результат. statsmodels оценивает
параметры методом максимального правдоподобия и сразу сообщает их
стандартные ошибки, а pykalman предлагает EM-алгоритм, который без изменений
переносится на модели с многомерным состоянием, но сходится медленно: ему
нужно около 100 итераций, то есть примерно 50 с на журнал из 1000 замеров.
Оценка максимального правдоподобия `estimate_noise_parameters` в этой простой
модели практически совпала с оценкой statsmodels и получена быстрее, но
стандартных ошибок не даёт; метод моментов мгновенный, но шумный и смещённый
вверх на коротких журналах.
"""
)

# ---------------------------------------------------------------------------------------------
md(
    r"""
## Эксперимент 4. Производительность

**Вопрос.** Сколько стоит обработка одного замера в разных реализациях?

**Условия.** (а) Скалярная модель записки: один ряд из 50 000 замеров
(случайное блуждание, известные $q$ и $R$), время фильтрации всего ряда.
(б) Та же модель на пакете из множества рядов (как журналы многих ВМ).
(в) Модель с векторным состоянием «уровень плюс скорость роста» (раздел 5
записки): один ряд из 20 000 замеров; trosna-kalman здесь работает общим
матричным фильтром. Везде измеряется время на один замер; для каждого
способа берётся лучшее из трёх повторений. Время зависит от компьютера,
важны соотношения.
"""
)

code(
    r'''
def best_time(function, repeat=3):
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return min(times)


N4 = 50_000
level4, z4 = random_walk(seed=4, n=N4)


def adaptive_loop(z):
    monitor = AdaptiveKalmanMonitor(z[0], R1, eps=0.1, dt_max=1.0)
    update = monitor.update
    return [update(value, 1.0) for value in z[1:]]


def listing_loop(z):
    monitor = ListingA1(z[0], R1, eps=0.1, dt_max=1.0)
    update = monitor.update
    return [update(value, 1.0) for value in z[1:]]


def matrix_trosna(z):
    return KalmanFilter([z[0]], [[R1]], [[1.0]], [[1.0]], [[Q1]], [[R1]]).filter(z[1:])


single = {
    "statsmodels": lambda: by_statsmodels(z4, Q1, R1),
    "simdkalman": lambda: by_simdkalman(z4, Q1, R1),
    "trosna-kalman: RandomWalkKalmanFilter": lambda: by_trosna(z4, Q1, R1),
    "листинг A.1 (адаптивный)": lambda: listing_loop(z4),
    "trosna-kalman: AdaptiveKalmanMonitor": lambda: adaptive_loop(z4),
    "pykalman": lambda: by_pykalman(z4, Q1, R1),
    "filterpy": lambda: by_filterpy(z4, Q1, R1),
    "trosna-kalman: KalmanFilter (матричный)": lambda: matrix_trosna(z4),
}
speed_single = pd.Series({name: best_time(function) / N4 * 1e6 for name, function in single.items()}, name="мкс на замер")
speed_single.sort_values().round(3).to_frame()
'''
)

code(
    r'''
def batch(n_series, length=1000, seed=40):
    rng = np.random.default_rng(seed)
    level = 2.0 + np.cumsum(rng.normal(0.0, math.sqrt(Q1), (n_series, length)), axis=1)
    return level + rng.normal(0.0, math.sqrt(R1), (n_series, length))


def simd_batch(data):
    kf = simdkalman.KalmanFilter(state_transition=np.eye(1), process_noise=np.eye(1) * Q1, observation_model=np.eye(1), observation_noise=R1)
    initial = np.eye(1) * (R1 + Q1)
    return kf.compute(data[:, 1:], 0, initial_value=data[:, :1, None], initial_covariance=initial, filtered=True, smoothed=False)


def denoiser_batch(data):
    return KalmanDenoiser(q=Q1, r=R1).fit(data.T).transform(data.T)  # столбцы — ряды


def loop_batch(run, data):
    return [run(series, Q1, R1) for series in data]


big, small = batch(2000), batch(50)
batch_methods = {
    "simdkalman (векторизация по рядам)": (lambda: simd_batch(big), big.size),
    "trosna-kalman: KalmanDenoiser (векторизация по рядам)": (lambda: denoiser_batch(big), big.size),
    "trosna-kalman: цикл по рядам": (lambda: loop_batch(by_trosna, big), big.size),
    "statsmodels: цикл по рядам": (lambda: loop_batch(by_statsmodels, small), small.size),
    "pykalman: цикл по рядам": (lambda: loop_batch(by_pykalman, small), small.size),
    "filterpy: цикл по рядам": (lambda: loop_batch(by_filterpy, small), small.size),
}
speed_batch = pd.Series({name: best_time(function) / size * 1e6 for name, (function, size) in batch_methods.items()}, name="мкс на замер")
speed_batch.sort_values().round(3).to_frame()
'''
)

code(
    r'''
F2 = np.array([[1.0, 1.0], [0.0, 1.0]])  # уровень и скорость роста
H2 = np.array([[1.0, 0.0]])
Q2 = np.diag([1e-4, 1e-7])
P2 = np.diag([R1, 1e-4])
N2 = 20_000
rng = np.random.default_rng(42)
trend = 0.004 + np.cumsum(rng.normal(0.0, math.sqrt(Q2[1, 1]), N2))
z2 = 2.0 + np.cumsum(trend + rng.normal(0.0, math.sqrt(Q2[0, 0]), N2)) + rng.normal(0.0, math.sqrt(R1), N2)
x2 = np.array([z2[0], 0.0])
prior_mean, prior_cov = F2 @ x2, F2 @ P2 @ F2.T + Q2  # априорное состояние для первого замера


def lt_trosna():
    return KalmanFilter(x2, P2, F2, H2, Q2, [[R1]]).filter(z2[1:])[0][:, 0]


def lt_filterpy():
    kf = FilterPyKalman(dim_x=2, dim_z=1)
    kf.x, kf.P, kf.F, kf.H, kf.Q, kf.R = x2.reshape(2, 1).copy(), P2.copy(), F2, H2, Q2, np.array([[R1]])
    out = np.empty(N2 - 1)
    for i, value in enumerate(z2[1:]):
        kf.predict()
        kf.update(value)
        out[i] = kf.x[0, 0]
    return out


def lt_pykalman():
    kf = PyKalman(
        transition_matrices=F2, observation_matrices=H2, transition_covariance=Q2,
        observation_covariance=[[R1]], initial_state_mean=prior_mean, initial_state_covariance=prior_cov,
    )
    return kf.filter(z2[1:])[0][:, 0]


def lt_statsmodels():
    model = sm.tsa.UnobservedComponents(z2[1:], level="local linear trend")
    model.initialize_known(prior_mean, prior_cov)
    model.ssm.tolerance = 0.0
    return model.filter(np.array([R1, Q2[0, 0], Q2[1, 1]])).filtered_state[0]


def lt_simdkalman():
    kf = simdkalman.KalmanFilter(state_transition=F2, process_noise=Q2, observation_model=H2, observation_noise=R1)
    result = kf.compute(z2[1:][None, :], 0, initial_value=prior_mean, initial_covariance=prior_cov, filtered=True, smoothed=False)
    return result.filtered.states.mean[0, :, 0]


level_trend = {
    "trosna-kalman: KalmanFilter (матричный)": lt_trosna,
    "filterpy": lt_filterpy,
    "pykalman": lt_pykalman,
    "statsmodels": lt_statsmodels,
    "simdkalman": lt_simdkalman,
}
reference2 = lt_trosna()
speed_vector = pd.Series({name: best_time(run) / (N2 - 1) * 1e6 for name, run in level_trend.items()}, name="мкс на замер")
agreement2 = pd.Series({name: float(np.max(np.abs(run() - reference2))) for name, run in level_trend.items()}, name="макс. |разность|, ГБ")
pd.concat({"один ряд": speed_single, "пакет рядов": speed_batch, "уровень + скорость": speed_vector}).to_csv(RESULTS / "experiment4_speed.csv")
pd.concat([speed_vector, agreement2], axis=1).sort_values("мкс на замер").round({"мкс на замер": 3})
'''
)

code(
    r'''
def color_of(name):
    for key, color in COLORS.items():
        if name.startswith(key) or key in name:
            return color
    return COLORS["листинг A.1"]


panels = [
    (speed_single, "а) Скалярная модель, один ряд из 50 000 замеров"),
    (speed_batch, "б) Скалярная модель, пакет рядов по 1000 замеров"),
    (speed_vector, "в) Модель «уровень + скорость роста», один ряд из 20 000 замеров"),
]
fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.0), layout="constrained", gridspec_kw={"height_ratios": [len(s) for s, _ in panels]})
for ax, (series, title) in zip(axes, panels):
    series = series.sort_values(ascending=False)
    bars = ax.barh(range(len(series)), series.values, color=[color_of(name) for name in series.index], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(range(len(series)), list(series.index), fontsize=8)
    for bar, value in zip(bars, series.values):
        label = comma(value, 0 if value >= 10 else 1 if value >= 1 else 2)
        ax.text(value * 1.15, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Время на один замер, мкс (логарифмическая шкала)")
    ax.set_title(title)
    ax.set_xlim(series.min() / 2, series.max() * 8)
save(fig, "fig5_speed")
plt.show()
'''
)

md(
    r"""
**Вывод эксперимента 4.** Результат зависит от модели. Для скалярной модели
случайного блуждания — модели метода записки — быстрее всех оказались
специализированные классы trosna-kalman: несколько микросекунд на замер на
чистом Python, тогда как библиотеки общего назначения на каждом шаге тратят
время на операции NumPy с маленькими матрицами (filterpy — примерно в 20 раз
дольше, simdkalman и pykalman на одном ряде — ещё дольше). На пакете рядов
векторизация simdkalman ускоряет обработку почти на два порядка по сравнению
с циклом filterpy, а векторизованный `KalmanDenoiser` trosna-kalman для
скалярной модели ещё быстрее. Для модели с векторным состоянием картина
обратная: скомпилированный фильтр statsmodels примерно в 10 раз быстрее
общего матричного фильтра trosna-kalman и в 5 раз быстрее filterpy, при
одинаковом результате. Это сильная сторона statsmodels: общие модели
пространства состояний любой размерности с циклом фильтра на Cython. Общий
матричный фильтр trosna-kalman написан на NumPy и на каждом шаге проверяет
данные, вычисляет форму Джозефа и правдоподобие: для больших векторных
моделей и массовой обработки разумнее взять statsmodels.
"""
)

# ---------------------------------------------------------------------------------------------
md(
    r"""
## Итоги

| Эксперимент | Что проверялось | Результат |
|---|---|---|
| 1. Согласие и оптимальность | одинаковость оценок и близость к пределу точности при известной модели | все реализации совпадают; стандартный фильтр с известными параметрами оптимален, адаптивный уступает ему примерно на 20 % |
| 2. Адаптивный опрос | ошибка при одинаковом числе опросов в сценариях записки | при 60–150 опросах за 300 с адаптивный монитор trosna-kalman точнее фильтра с постоянным интервалом на 5–10 %; при 50 опросах и меньше преимущества нет |
| 3. Офлайн-анализ | оценка параметров и восстановление истории по всей записи | сглаживатели RTS (filterpy, pykalman, statsmodels, simdkalman) точнее любого каузального фильтра на 17 %; statsmodels даёт стандартные ошибки, pykalman — EM для многомерных моделей |
| 4. Производительность | время на замер | для скалярной модели быстрее специализированные классы trosna-kalman; для векторной модели statsmodels примерно в 10 раз быстрее общего фильтра trosna-kalman |

Классические реализации остаются лучшим выбором, когда модель известна, когда
нужно сглаживание и оценка параметров по готовым журналам и когда модель
многомерна. Метод записки, реализованный в trosna-kalman, решает другую
задачу — когда опрашивать ВМ — и в ней при среднем бюджете опросов даёт
меньшую ошибку при том же числе опросов.
"""
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
target = Path("comparison.ipynb")
if "--keep-outputs" in sys.argv and target.exists():
    executed = nbf.read(target, as_version=4)
    kept = 0
    for new, old in zip(nb["cells"], executed["cells"], strict=False):
        if new["cell_type"] == old["cell_type"] == "code" and new["source"] == old["source"]:
            new["outputs"], new["execution_count"] = old["outputs"], old["execution_count"]
            kept += 1
    print(f"outputs kept for {kept} code cells")
    nb["metadata"] = executed["metadata"]
nbf.write(nb, target)
print(f"{target}: {len(cells)} cells")
