# Data loading and cleaning

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore', category=FutureWarning)


class DataLoader:
    """Load all datathon datasets."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data = {}

    def load_all(self, include_large_files: bool = True) -> dict:
        """Load all datasets. Set include_large_files=False to skip patents/SEC."""
        self._load_genai_dimension()
        self._load_genai_research()
        self._load_cursor_dimension()
        self._load_ticker_dimension()
        self._load_enterprise_ai_adoption()
        self._load_economic_data()
        self._load_stock_prices()
        self._load_currency_pairs()

        if include_large_files:
            self._load_sec_edgar()
            self._load_patent_data()

        print(f"Loaded {len(self.data)} datasets")
        return self.data

    def _load_genai_dimension(self):
        df = pd.read_csv(self.data_dir / "genai_dimension.csv")
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        self.data['genai_models'] = df

    def _load_genai_research(self):
        df = pd.read_csv(self.data_dir / "genai_research_dim.csv")
        df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
        self.data['genai_research'] = df

    def _load_cursor_dimension(self):
        df = pd.read_csv(self.data_dir / "cursor_dim.csv")
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        self.data['cursor'] = df

    def _load_ticker_dimension(self):
        df = pd.read_csv(self.data_dir / "ticker_dimension.csv")
        self.data['tickers'] = df

    def _load_enterprise_ai_adoption(self):
        df = pd.read_csv(self.data_dir / "enterprise_ai_adoption_internet_events.csv")
        df['annoucement_date'] = pd.to_datetime(df['annoucement_date'], errors='coerce')
        self.data['ai_adoption'] = df

    def _load_economic_data(self):
        df = pd.read_csv(self.data_dir / "economic_data_fred_bls_series_2015_2025.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        self.data['economic'] = df

    def _load_stock_prices(self):
        stock_files = {
            'etf_prices': 'etfs_prices_all_since_2015.csv',
            'sp500_prices': 'sp500_prices_all_since_2015.csv',
            'sp400_prices': 'sp400_prices_all_since_2015.csv',
            'sp600_prices': 'sp600_prices_all_since_2015.csv'
        }
        for key, filename in stock_files.items():
            df = pd.read_csv(self.data_dir / filename)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            self.data[key] = df

    def _load_currency_pairs(self):
        df = pd.read_csv(self.data_dir / "currency_pairs_against_usd_since_2015.csv")
        df['last_updated_utc'] = pd.to_datetime(df['last_updated_utc'], errors='coerce')
        self.data['currency'] = df

    def _load_patent_data(self):
        # WARNING: large files ~1.2GB total
        df1 = pd.read_csv(self.data_dir / "ai_model_predictions_2015_2019_patents.csv")
        df1['pub_dt'] = pd.to_datetime(df1['pub_dt'], errors='coerce')
        self.data['patents_2015_2019'] = df1

        df2 = pd.read_csv(self.data_dir / "ai_model_predictions_2020_2023_patents.csv")
        df2['pub_dt'] = pd.to_datetime(df2['pub_dt'], errors='coerce')
        self.data['patents_2020_2023'] = df2

    def _load_sec_edgar(self):
        # WARNING: ~1GB file
        df = pd.read_csv(self.data_dir / "sp_edgar_fundamentals.csv")
        date_cols = ['start_date', 'end_date', 'filed']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        self.data['sec_fundamentals'] = df


class DataCleaner:
    """Cleaning utilities."""

    @staticmethod
    def clean_stock_data(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
        """Clean and sort stock prices, flag suspicious returns."""
        df = df.copy()

        critical_cols = ['Date', 'Close', 'Adj Close', ticker_col]
        df = df.dropna(subset=critical_cols)
        df = df.sort_values([ticker_col, 'Date'])
        df = df.drop_duplicates(subset=[ticker_col, 'Date'], keep='last')

        # returns + winsorize
        df['returns'] = df.groupby(ticker_col)['Adj Close'].pct_change()
        from scipy.stats.mstats import winsorize
        df['returns_winsorized'] = df.groupby(ticker_col)['returns'].transform(
            lambda x: winsorize(x, limits=[0.01, 0.01])
        )

        return df

    @staticmethod
    def normalize_tickers(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
        """Uppercase + strip whitespace for reliable merges."""
        df = df.copy()
        if ticker_col not in df.columns:
            raise KeyError(f"{ticker_col} not found in dataframe")
        df[ticker_col] = (
            df[ticker_col]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r'\\s+', '', regex=True)
        )
        return df

    @staticmethod
    def build_maturity_level(
        df: pd.DataFrame,
        use_case_col: str = 'use_case',
        agent_use_case_col: str = 'agent_use_case',
        output_col: str = 'maturity_level'
    ) -> pd.DataFrame:
        """Create maturity level (0=intent/hiring, 1=pilot, 2=production, 3=productization)."""
        df = df.copy()

        if use_case_col not in df.columns and agent_use_case_col not in df.columns:
            raise KeyError("Neither use_case nor agent_use_case columns were found")

        text = (
            df.get(use_case_col, pd.Series('', index=df.index)).fillna('') + ' ' +
            df.get(agent_use_case_col, pd.Series('', index=df.index)).fillna('')
        ).str.lower()

        keywords = {
            3: ['product', 'productization', 'commercial', 'ga', 'general availability',
                'launch', 'customer', 'revenue', 'subscription', 'pricing'],
            2: ['production', 'rollout', 'deploy', 'deployment', 'scale', 'scaled',
                'integration', 'go-live'],
            1: ['pilot', 'poc', 'proof of concept', 'trial', 'beta', 'experiment',
                'prototype', 'sandbox'],
            0: ['intent', 'plan', 'planning', 'evaluate', 'evaluation', 'explore',
                'research', 'hiring', 'job posting', 'recruiting'],
        }

        def assign_level(t):
            if not t:
                return None
            for level in [3, 2, 1, 0]:
                if any(k in t for k in keywords[level]):
                    return level
            return None

        df[output_col] = text.apply(assign_level)
        return df

    @staticmethod
    def align_to_next_trading_day(
        events_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        event_date_col: str = 'annoucement_date',
        ticker_col: str = 'ticker',
        price_date_col: str = 'Date',
        output_col: str = 'event_trade_date'
    ) -> pd.DataFrame:
        """Map event dates to the next available trading day per ticker."""
        events = events_df.copy()
        prices = prices_df.copy()

        events[event_date_col] = pd.to_datetime(events[event_date_col], errors='coerce')
        prices[price_date_col] = pd.to_datetime(prices[price_date_col], errors='coerce')

        prices = DataCleaner.normalize_tickers(prices, ticker_col=ticker_col)
        events = DataCleaner.normalize_tickers(events, ticker_col=ticker_col)

        trading_dates = (
            prices[[ticker_col, price_date_col]]
            .dropna()
            .drop_duplicates()
            .sort_values([ticker_col, price_date_col])
        )

        date_map = {
            t: trading_dates[trading_dates[ticker_col] == t][price_date_col].values
            for t in trading_dates[ticker_col].unique()
        }

        def next_trade_date(row):
            ticker = row[ticker_col]
            event_date = row[event_date_col]
            if pd.isna(event_date) or ticker not in date_map:
                return pd.NaT
            dates = date_map[ticker]
            idx = dates.searchsorted(event_date)
            if idx >= len(dates):
                return pd.NaT
            return pd.Timestamp(dates[idx])

        events[output_col] = events.apply(next_trade_date, axis=1)
        return events

    @staticmethod
    def build_text_features(
        df: pd.DataFrame,
        text_cols: list,
        keyword_map: dict = None,
        prefix: str = 'kw_'
    ) -> pd.DataFrame:
        """Build keyword count features from text columns."""
        df = df.copy()

        if keyword_map is None:
            keyword_map = {
                'automation': ['automation', 'automate', 'autonomous'],
                'cost': ['cost', 'saving', 'efficiency', 'productivity'],
                'customer': ['customer', 'client', 'user', 'support'],
                'product': ['product', 'feature', 'launch', 'release'],
                'agent': ['agent', 'assistant', 'copilot'],
            }

        combined = (
            df[text_cols]
            .fillna('')
            .astype(str)
            .agg(' '.join, axis=1)
            .str.lower()
        )

        for name, keywords in keyword_map.items():
            pattern = '|'.join([k.replace(' ', '\\s+') for k in keywords])
            df[f"{prefix}{name}"] = combined.str.count(pattern)

        return df

    @staticmethod
    def clean_economic_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean FRED/BLS data: ffill/bfill missing values, drop series with >20% missing."""
        df = df.copy()
        df = df.sort_values(['indicator', 'state', 'date'])

        df['value'] = df.groupby(['indicator', 'state'])['value'].transform(
            lambda x: x.fillna(method='ffill', limit=3).fillna(method='bfill', limit=3)
        )

        missing_pct = df.groupby(['indicator', 'state'])['value'].apply(
            lambda x: x.isna().sum() / len(x)
        )
        high_missing = missing_pct[missing_pct > 0.2].index
        df = df[~df.set_index(['indicator', 'state']).index.isin(high_missing)]

        return df

    @staticmethod
    def clean_patent_data(df: pd.DataFrame, confidence_threshold: str = 'predict86') -> pd.DataFrame:
        """Filter patents by AI classification confidence level."""
        df = df.copy()
        df = df[df['pub_dt'].notna()]

        ai_cols = [col for col in df.columns if col.startswith(confidence_threshold)]
        df['is_ai'] = df[ai_cols].max(axis=1)
        df = df[df['is_ai'] == 1]

        return df

    @staticmethod
    def handle_outliers(series: pd.Series, method: str = 'winsorize',
                       limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
        """Handle outliers: winsorize, clip, or remove via IQR."""
        if method == 'winsorize':
            from scipy.stats.mstats import winsorize
            return pd.Series(winsorize(series, limits=limits), index=series.index)
        elif method == 'clip':
            lower = series.quantile(limits[0])
            upper = series.quantile(1 - limits[1])
            return series.clip(lower, upper)
        elif method == 'remove':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return series[(series >= lower) & (series <= upper)]
        else:
            raise ValueError(f"Unknown method: {method}")


class DataValidator:
    """Data quality checks."""

    @staticmethod
    def check_missing(df: pd.DataFrame, threshold: float = 0.1) -> dict:
        """Check for missing values above threshold."""
        missing = df.isna().sum() / len(df)
        problematic = missing[missing > threshold]
        return {
            'total_missing_pct': missing.sum() / len(df.columns),
            'problematic_columns': problematic.to_dict(),
            'rows_with_any_missing': df.isna().any(axis=1).sum()
        }

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset=None) -> dict:
        duplicates = df.duplicated(subset=subset, keep=False).sum()
        return {
            'duplicate_count': duplicates,
            'duplicate_pct': duplicates / len(df)
        }

    @staticmethod
    def check_date_gaps(df: pd.DataFrame, date_col: str, expected_freq: str = 'D') -> dict:
        """Check for gaps in time series."""
        df = df.sort_values(date_col)
        date_range = pd.date_range(start=df[date_col].min(),
                                   end=df[date_col].max(),
                                   freq=expected_freq)
        actual_dates = set(df[date_col].dt.normalize())
        expected_dates = set(date_range.normalize())
        missing_dates = expected_dates - actual_dates

        return {
            'expected_count': len(expected_dates),
            'actual_count': len(actual_dates),
            'missing_dates_count': len(missing_dates),
            'coverage_pct': len(actual_dates) / len(expected_dates)
        }

    @staticmethod
    def generate_report(df: pd.DataFrame, name: str = "Dataset") -> str:
        """Generate data quality report."""
        report = f"\nDATA QUALITY REPORT: {name}\n"
        report += f"{'='*50}\n\n"

        report += f"Shape: {df.shape}\n"
        report += f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n\n"

        missing = DataValidator.check_missing(df)
        report += f"Missing: {missing['total_missing_pct']:.2%}\n"
        report += f"Rows with missing: {missing['rows_with_any_missing']:,}\n"
        if missing['problematic_columns']:
            report += f"Problematic columns (>10% missing):\n"
            for col, pct in missing['problematic_columns'].items():
                report += f"  - {col}: {pct:.2%}\n"

        dupes = DataValidator.check_duplicates(df)
        report += f"\nDuplicates: {dupes['duplicate_count']:,} ({dupes['duplicate_pct']:.2%})\n"

        report += f"\nData Types:\n"
        for dtype, count in df.dtypes.value_counts().items():
            report += f"  {dtype}: {count} columns\n"

        return report


if __name__ == "__main__":
    loader = DataLoader(data_dir="./Datathon Materials")
    data = loader.load_all(include_large_files=False)

    if 'sp500_prices' in data:
        clean_stocks = DataCleaner.clean_stock_data(data['sp500_prices'])
        print(f"Cleaned SP500: {clean_stocks.shape}")

    if 'economic' in data:
        print(DataValidator.generate_report(data['economic'], name="Economic Data"))
