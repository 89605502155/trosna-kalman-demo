# trosna-kalman-demo

Сравнение библиотеки [trosna-kalman](https://github.com/89605502155/trosna-kalman)
с эталонными реализациями фильтра Калмана: листингом A.1 записки
А. О. Ферубко и О. Д. Казакова «Фильтр Калмана и адаптивный выбор интервала
мониторинга памяти виртуальных машин» и библиотеками
[filterpy](https://pypi.org/project/filterpy/),
[pykalman](https://pypi.org/project/pykalman/),
[statsmodels](https://pypi.org/project/statsmodels/) и
[simdkalman](https://pypi.org/project/simdkalman/).

Comparison of the [trosna-kalman](https://github.com/89605502155/trosna-kalman)
library with reference Kalman filter implementations: listing A.1 of the paper
by A. O. Ferubko and O. D. Kazakov "Kalman filter and adaptive choice of the
monitoring interval of virtual machine memory" and the libraries filterpy,
pykalman, statsmodels and simdkalman.

## Эксперименты / Experiments

| № | Эксперимент | Experiment |
|---|---|---|
| 1 | Согласие реализаций и оптимальность при известной модели | Agreement of the implementations and optimality when the model is known |
| 2 | Адаптивный опрос при скачках нагрузки и одинаковом числе опросов (модель раздела 4.5 записки) | Adaptive polling under load jumps with the same number of polls (the model of section 4.5) |
| 3 | Офлайн-анализ журнала: оценка параметров шума и сглаживание RTS | Offline analysis of a log: noise parameters and RTS smoothing |
| 4 | Производительность на одном ряде и на пакете рядов | Performance on one series and on a batch of series |

Условия каждого эксперимента описаны в ноутбуке; все случайные данные
порождаются с фиксированными зёрнами.

The conditions of every experiment are described in the notebook; all random
data come from fixed seeds.

## Главные результаты

1. Все реализации дают одинаковые оценки (расхождение — на уровне ошибок
   округления, до 10⁻¹⁵ ГБ); при известных $q$ и $R$ стандартный фильтр
   достигает теоретического предела точности, адаптивный монитор, оценивающий
   $q$ по невязкам, уступает ему примерно на 20 %.
2. В сценариях раздела 4.5 записки при 60–150 опросах за 300 с адаптивный
   монитор trosna-kalman даёт ошибку на 5–10 % меньше, чем фильтр filterpy с
   постоянным интервалом и подобранным $q$ (при 100 опросах — 52 против 58 МБ,
   p < 0,001); при 50 опросах и меньше преимущества нет.
3. Сглаживание RTS (filterpy, pykalman, statsmodels, simdkalman) восстанавливает
   историю потребления на 17 % точнее каузального фильтра; statsmodels даёт
   стандартные ошибки оценок, EM pykalman сходится примерно за 100 итераций.
4. Для скалярной модели быстрее всех специализированные классы trosna-kalman;
   для модели с векторным состоянием statsmodels примерно в 10 раз быстрее
   общего матричного фильтра trosna-kalman.

## Main results

1. All implementations give the same estimates (differences at the rounding
   level, up to 10⁻¹⁵ GB); with known $q$ and $R$ the standard filter reaches
   the theoretical limit of accuracy, while the adaptive monitor, which
   estimates $q$ from the innovations, is about 20 % worse.
2. In the scenarios of section 4.5 of the paper, with 60–150 polls per 300 s
   the adaptive monitor of trosna-kalman has a 5–10 % smaller error than the
   filterpy filter polled at a constant interval with a tuned $q$ (52 against
   58 MB at 100 polls, p < 0.001); with 50 polls or fewer there is no advantage.
3. RTS smoothing (filterpy, pykalman, statsmodels, simdkalman) reconstructs the
   history 17 % more accurately than a causal filter; statsmodels reports
   standard errors, pykalman's EM converges in about 100 iterations.
4. For the scalar model the specialized classes of trosna-kalman are the
   fastest; for a vector-state model statsmodels is about 10 times faster than
   the general matrix filter of trosna-kalman.

## Содержимое / Contents

| Файл / File | Назначение / Purpose |
|---|---|
| `comparison.ipynb` | ноутбук с экспериментами и результатами выполнения / the notebook with outputs |
| `make_notebook.py` | исходный текст ячеек ноутбука / the source of the notebook cells |
| `results/*.csv` | таблицы результатов / result tables |
| `figures/` | рисунки шириной 165 мм: PDF, TIFF 600 dpi, PNG / 165 mm wide figures: PDF, TIFF 600 dpi, PNG |

## Запуск / Running

```sh
uv sync
uv run python make_notebook.py   # пересобрать ноутбук из исходника / rebuild the notebook
uv run jupyter nbconvert --to notebook --execute --inplace comparison.ipynb --ExecutePreprocessor.timeout=3600
# или интерактивно / or interactively:
uv run jupyter lab comparison.ipynb
```

Нужен Python 3.11 или новее. Библиотека trosna-kalman 0.0.1 устанавливается
с [PyPI](https://pypi.org/project/trosna-kalman/0.0.1/).

Python 3.11 or newer is required. The trosna-kalman 0.0.1 library is installed
from [PyPI](https://pypi.org/project/trosna-kalman/0.0.1/).

## Лицензия / License

MIT
