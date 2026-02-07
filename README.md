# Datathon 2026 - GenAI & Financial Markets

Repo for the Women's Datathon 2026. Data loading, analysis (DiD, event study, GARCH, VAR), and visualization.

## Setup

```bash
pip install -r requirements.txt
```

```bash
pip install pandas numpy statsmodels linearmodels plotly ruptures tfcausalimpact pysyncon
```

## Repository structure

- `01_data_preprocessing.py` - Load, clean, validate datasets
- `02_statistical_analysis.py` - DiD, panel regression, event study, GARCH, VAR
- `03_visualization.py` - Plots for time series, event study, diagnostics
- `utils.py` - LaTeX table export
- `test_new_features.py` - Feature tests
- `test_outputs/` - Outputs from `test_new_features.py`
- `project1/data/raw/` - Raw inputs for project 1
- `project1/data/processed/` - Cleaned/derived datasets for project 1
- `project1/src/01_prepare_state_panel.py` - Build state panel
- `project1/src/02_panel_regression.py` - Panel regression models
- `project1/src/03_synth_control.py` - Synthetic control
- `project1/src/04_causal_impact.py` - Causal impact
- `project1/src/05_event_study.py` - Event study
- `project1/outputs/tables/` - Tables
- `project1/outputs/figures/` - Figures

## Example

```python
import importlib.util
import pandas as pd

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

stats = import_module_from_file("statistical_analysis", "02_statistical_analysis.py")
EventStudy = stats.EventStudy

# load data
data_module = import_module_from_file("data_preprocessing", "01_data_preprocessing.py")
loader = data_module.DataLoader(data_dir="./Datathon Materials")
data = loader.load_all(include_large_files=False)

# clean stocks
prices = data_module.DataCleaner.clean_stock_data(data['sp500_prices'])

# run event study
result = EventStudy.market_adjusted_abnormal_returns(
    stock_returns, benchmark_returns,
    event_date=pd.Timestamp('2023-03-14'),
    event_window=(-5, 10)
)
```

## Git

```bash
git add -A
git commit -m "Your message"
git push origin main
```

## Research notes

- AI-exposed firms vs US macro slowdowns: sector splits and adoption stage effects.
- State-level GenAI reform impacts: startup gains vs job losses.
- Equity market reactions by adoption stage (intent, pilot, production, launch).
