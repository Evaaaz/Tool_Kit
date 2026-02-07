# Datathon 2026 - GenAI & Financial Markets

Toolkit for the Women's Datathon 2026. Covers data loading, statistical analysis (DiD, event study, GARCH, VAR), and visualization.

## Setup

```bash
pip install -r requirements.txt
```

```bash
pip install pandas numpy statsmodels linearmodels plotly ruptures tfcausalimpact pysyncon
```

## Repository structure

- `01_data_preprocessing.py` - Load/clean/validate all datasets
- `02_statistical_analysis.py` - DiD, panel regression, event study, GARCH, VAR
- `03_visualization.py` - Time series plots, event study charts, regression diagnostics
- `utils.py` - LaTeX table export
- `test_new_features.py` - Verify everything works
- `test_outputs/` - Output artifacts from `test_new_features.py`
- `project1/data/raw/` - Raw inputs for project 1
- `project1/data/processed/` - Cleaned/derived datasets for project 1
- `project1/src/01_prepare_state_panel.py` - Build state-level panel for project 1
- `project1/src/02_panel_regression.py` - Panel regression models for project 1
- `project1/src/03_synth_control.py` - Synthetic control workflow for project 1
- `project1/src/04_causal_impact.py` - Causal impact analysis for project 1
- `project1/src/05_event_study.py` - Event study analysis for project 1
- `project1/outputs/tables/` - Tables for project 1 results
- `project1/outputs/figures/` - Figures for project 1 results

## Quick usage

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

## Current research question

* Are AI-exposed firms (this could mean AI research companies, or AI infra, we can use the AI ) less affected (how to quanitfy this) by US macroeconomic slowdowns (maybe analyze by different sectors that use AI)

  * hypothesis: in economic downturns, general companies may turn more to AI automation or reliance on AI due to efficiency or cost concerns, making AI companies
  * hypothesis: in economic downturns, public sentiments may about AI may become more negative or and investor anxieties about AI investments may increase → less capital investments. economic downturn could mean interest rates increase
  * track changes throughout the recent years as AI has become more widely accepted as inevitability?
* How are different US states treated under GenAI reform? (some get a positive effect through new startups etc., but are these the same states that also have a lot of workers lose their jobs due to AI)

  * hypothesis: states where economy more dependent on manufacturing etc. should not be as affected in neither positive or negative direction, whereas e.g. California gets positive effects in the form of new startups and investments etc., but also negative in the sense that a lot of knowledge-driven roles will be affected
* How do equity markets price GenAI adoption news by maturity stage (intent -> pilot -> production -> launch)?

  * hypotheses:
    * Late-stage announcements (production/launch) have higher CAR than early-stage
    * Early-stage announcements are noisier with more dispersed reactions
