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

## Содержимое / Contents

| Файл / File | Назначение / Purpose |
|---|---|
| `comparison.ipynb` | ноутбук с экспериментами и результатами выполнения / the notebook with outputs |
| `make_notebook.py` | исходный текст ячеек ноутбука / the source of the notebook cells |
| `results/*.csv` | таблицы результатов / result tables |
| `figures/` | рисунки, PNG и TIFF 600 dpi / figures, PNG and TIFF 600 dpi |

## Запуск / Running

```sh
uv sync
uv run python make_notebook.py   # пересобрать ноутбук из исходника / rebuild the notebook
uv run jupyter nbconvert --to notebook --execute --inplace comparison.ipynb --ExecutePreprocessor.timeout=3600
# или интерактивно / or interactively:
uv run jupyter lab comparison.ipynb
```

Нужен Python 3.11 или новее. Пока trosna-kalman 0.0.1 не опубликована на PyPI,
`uv sync` берёт её из соседней папки `../trosna-kalman` (раздел
`[tool.uv.sources]` в `pyproject.toml`); после публикации этот раздел нужно
удалить и выполнить `uv lock`.

Python 3.11 or newer is required. Until trosna-kalman 0.0.1 is published on
PyPI, `uv sync` takes it from the neighbouring folder `../trosna-kalman` (the
`[tool.uv.sources]` section of `pyproject.toml`); after the release remove that
section and run `uv lock`.

## Лицензия / License

MIT
