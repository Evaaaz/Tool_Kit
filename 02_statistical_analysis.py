# Statistical analysis methods for datathon

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class HypothesisTests:
    """Statistical tests with corrections and diagnostics."""

    @staticmethod
    def difference_in_differences(df: pd.DataFrame,
                                 outcome_col: str,
                                 treatment_col: str,
                                 post_col: str,
                                 controls: list = None) -> dict:
        """DiD estimation. Y = b0 + b1*Treat + b2*Post + b3*(Treat x Post) + Controls + e"""
        df = df.copy()
        df['treatment_post'] = df[treatment_col] * df[post_col]

        X_vars = [treatment_col, post_col, 'treatment_post']
        if controls:
            X_vars.extend(controls)

        # check multicollinearity
        X_check = df[X_vars].dropna()
        if len(X_check.columns) > 1:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_check.columns
            vif_data["VIF"] = [variance_inflation_factor(X_check.values, i)
                              for i in range(len(X_check.columns))]
            high_vif = vif_data[vif_data['VIF'] > 10]
            if not high_vif.empty:
                warnings.warn(f"High multicollinearity detected:\n{high_vif}")

        X = sm.add_constant(df[X_vars].dropna())
        y = df.loc[X.index, outcome_col]

        model = sm.OLS(y, X).fit(cov_type='HC3')

        did_coef = model.params['treatment_post']
        did_se = model.bse['treatment_post']
        did_pval = model.pvalues['treatment_post']
        ci_lower, ci_upper = model.conf_int().loc['treatment_post']

        # parallel trends check
        pre_period = df[df[post_col] == 0]
        if len(pre_period) > 0:
            pre_model = sm.OLS(
                pre_period[outcome_col],
                sm.add_constant(pre_period[[treatment_col]])
            ).fit(cov_type='HC3')
            parallel_trends_pval = pre_model.pvalues[treatment_col]
        else:
            parallel_trends_pval = np.nan

        return {
            'did_estimate': did_coef,
            'std_error': did_se,
            'p_value': did_pval,
            'ci_95': (ci_lower, ci_upper),
            'significant': did_pval < 0.05,
            'parallel_trends_test_pval': parallel_trends_pval,
            'parallel_trends_violated': parallel_trends_pval < 0.05 if not np.isnan(parallel_trends_pval) else None,
            'r_squared': model.rsquared,
            'n_obs': len(y),
            'full_model': model
        }

    @staticmethod
    def panel_regression(df: pd.DataFrame,
                        outcome_col: str,
                        X_cols: List[str],
                        entity_col: str,
                        time_col: str,
                        effects: str = 'entity') -> Dict:
        """Panel regression with fixed effects and clustered SEs."""
        from linearmodels.panel import PanelOLS

        df = df.copy()
        df = df.set_index([entity_col, time_col])

        y = df[outcome_col]
        X = df[X_cols]

        if effects == 'entity':
            model = PanelOLS(y, X, entity_effects=True, time_effects=False)
        elif effects == 'time':
            model = PanelOLS(y, X, entity_effects=False, time_effects=True)
        elif effects == 'twoway':
            model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        else:
            raise ValueError("effects must be 'entity', 'time', or 'twoway'")

        results = model.fit(cov_type='clustered', cluster_entity=True)

        return {
            'coefficients': results.params.to_dict(),
            'std_errors': results.std_errors.to_dict(),
            'p_values': results.pvalues.to_dict(),
            'r_squared': results.rsquared,
            'r_squared_within': results.rsquared_within,
            'f_statistic': results.f_statistic.stat,
            'f_pvalue': results.f_statistic.pval,
            'n_obs': results.nobs,
            'n_entities': results.entity_info.total,
            'full_results': results
        }

    @staticmethod
    def granger_causality(df: pd.DataFrame,
                         cause_col: str,
                         effect_col: str,
                         max_lag: int = 4,
                         alpha: float = 0.05) -> Dict:
        """Test whether cause_col Granger-causes effect_col."""
        data = df[[cause_col, effect_col]].dropna()
        results = grangercausalitytests(data, max_lag, verbose=False)

        pvalues = {}
        for lag in range(1, max_lag + 1):
            pvalues[f'lag_{lag}'] = results[lag][0]['ssr_ftest'][1]

        min_pval = min(pvalues.values())
        is_causal = min_pval < alpha

        return {
            'pvalues_by_lag': pvalues,
            'min_pvalue': min_pval,
            'granger_causes': is_causal,
            'optimal_lag': min(pvalues, key=pvalues.get).split('_')[1]
        }

    @staticmethod
    def multiple_testing_correction(pvalues: list,
                                    method: str = 'fdr_bh',
                                    alpha: float = 0.05) -> dict:
        reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method=method)

        return {
            'raw_pvalues': pvalues,
            'corrected_pvalues': pvals_corrected.tolist(),
            'reject_null': reject.tolist(),
            'method': method,
            'alpha': alpha,
            'num_significant_raw': sum(p < alpha for p in pvalues),
            'num_significant_corrected': sum(reject)
        }


class TimeSeriesAnalysis:
    """Time series stationarity checks and decomposition."""

    @staticmethod
    def check_stationarity(series: pd.Series,
                          name: str = "Series",
                          alpha: float = 0.05) -> Dict:
        """ADF test for stationarity. Reject H0 (unit root) if p < alpha."""
        series = series.dropna()
        result = adfuller(series, autolag='AIC')
        is_stationary = result[1] < alpha

        return {
            'series_name': name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary,
        }

    @staticmethod
    def decompose_series(series: pd.Series,
                        period: int,
                        model: str = 'additive') -> Dict:
        """Seasonal decomposition into trend, seasonal, residual."""
        decomposition = seasonal_decompose(series, model=model, period=period)
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'original': series
        }

    @staticmethod
    def autocorrelation_analysis(series: pd.Series, nlags: int = 40) -> Dict:
        series = series.dropna()
        acf_values = acf(series, nlags=nlags)
        pacf_values = pacf(series, nlags=nlags)

        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(series, lags=min(20, len(series)//5))

        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[-1],
            'has_autocorrelation': lb_result['lb_pvalue'].iloc[-1] < 0.05
        }

    @staticmethod
    def rolling_correlation(series1, series2, window=12):
        return series1.rolling(window).corr(series2)


class EventStudy:
    """Event study methodology for stock returns."""

    @staticmethod
    def abnormal_returns(stock_returns: pd.Series,
                        market_returns: pd.Series,
                        event_date: pd.Timestamp,
                        estimation_window: int = 120,
                        event_window: Tuple[int, int] = (-5, 10)) -> Dict:
        """Market model abnormal returns: estimate normal returns in estimation window, then compute AR and CAR."""
        df = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        event_idx = df.index.get_loc(event_date)
        est_start = event_idx - estimation_window - event_window[0]
        est_end = event_idx - event_window[0] - 1

        est_data = df.iloc[est_start:est_end]

        # market model: R_stock = a + b*R_market
        X = sm.add_constant(est_data['market'])
        y = est_data['stock']
        model = sm.OLS(y, X).fit()

        alpha = model.params['const']
        beta = model.params['market']

        # event window
        event_start = event_idx + event_window[0]
        event_end = event_idx + event_window[1] + 1
        event_data = df.iloc[event_start:event_end]

        expected_returns = alpha + beta * event_data['market']
        abnormal_returns = event_data['stock'] - expected_returns
        car = abnormal_returns.cumsum()

        # t-test
        car_mean = car.iloc[-1]
        car_se = abnormal_returns.std() / np.sqrt(len(abnormal_returns))
        t_stat = car_mean / car_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(abnormal_returns) - 1))

        return {
            'alpha': alpha,
            'beta': beta,
            'abnormal_returns': abnormal_returns,
            'car': car,
            'car_total': car_mean,
            'car_se': car_se,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    @staticmethod
    def bootstrap_confidence_intervals(abnormal_returns: pd.Series,
                                      n_bootstrap: int = 1000,
                                      alpha: float = 0.05) -> Tuple[float, float]:
        """Bootstrap CI for CAR."""
        np.random.seed(42)
        bootstrap_cars = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(abnormal_returns, size=len(abnormal_returns), replace=True)
            bootstrap_cars.append(sample.sum())

        lower = np.percentile(bootstrap_cars, 100 * alpha / 2)
        upper = np.percentile(bootstrap_cars, 100 * (1 - alpha / 2))
        return (lower, upper)

    @staticmethod
    def market_adjusted_abnormal_returns(
        stock_returns: pd.Series,
        benchmark_returns: pd.Series,
        event_date: pd.Timestamp,
        event_window: Tuple[int, int] = (-5, 10)
    ) -> Dict:
        """Simple market-adjusted AR: AR = stock - benchmark."""
        df = pd.DataFrame({
            'stock': stock_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if df.empty:
            raise ValueError("No overlapping returns for stock and benchmark")

        if event_date not in df.index:
            idx = df.index.searchsorted(event_date)
            if idx >= len(df.index):
                raise ValueError("Event date is after last available trading date")
            event_date = df.index[idx]

        event_idx = df.index.get_loc(event_date)
        event_start = event_idx + event_window[0]
        event_end = event_idx + event_window[1] + 1
        if event_start < 0 or event_end > len(df.index):
            raise ValueError("Event window exceeds available data range")

        event_data = df.iloc[event_start:event_end]
        abnormal_returns = event_data['stock'] - event_data['benchmark']
        car = abnormal_returns.cumsum()

        car_total = car.iloc[-1]
        car_se = abnormal_returns.std() / np.sqrt(len(abnormal_returns))
        t_stat = car_total / car_se if car_se != 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(abnormal_returns) - 1)) if np.isfinite(t_stat) else np.nan

        return {
            'abnormal_returns': abnormal_returns,
            'car': car,
            'car_total': car_total,
            'car_se': car_se,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': bool(p_value < 0.05) if np.isfinite(p_value) else False,
            'event_date': event_date
        }

    @staticmethod
    def batch_event_study(
        events_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        event_date_col: str = 'event_trade_date',
        ticker_col: str = 'ticker',
        date_col: str = 'Date',
        returns_col: str = 'returns',
        event_window: Tuple[int, int] = (-5, 10),
        method: str = 'market_adjusted'
    ) -> Dict:
        """Run event study across many events and compute CAAR."""
        events = events_df.copy()
        data = returns_df.copy()

        events[event_date_col] = pd.to_datetime(events[event_date_col], errors='coerce')
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

        data[ticker_col] = data[ticker_col].astype(str).str.upper().str.strip()
        events[ticker_col] = events[ticker_col].astype(str).str.upper().str.strip()

        benchmark = benchmark_returns.copy()
        benchmark.index = pd.to_datetime(benchmark.index, errors='coerce')

        event_results = []
        abnormal_matrix = []
        rel_days = list(range(event_window[0], event_window[1] + 1))

        for idx, row in events.iterrows():
            ticker = row[ticker_col]
            event_date = row[event_date_col]
            if pd.isna(event_date):
                continue

            stock_series = (
                data[data[ticker_col] == ticker]
                .dropna(subset=[date_col, returns_col])
                .set_index(date_col)[returns_col]
                .sort_index()
            )
            if stock_series.empty:
                continue

            try:
                if method == 'market_adjusted':
                    result = EventStudy.market_adjusted_abnormal_returns(
                        stock_series, benchmark, event_date, event_window
                    )
                else:
                    result = EventStudy.abnormal_returns(
                        stock_series, benchmark, event_date,
                        estimation_window=120, event_window=event_window
                    )
            except Exception:
                continue

            ar = result['abnormal_returns'].values
            if len(ar) != len(rel_days):
                continue

            abnormal_matrix.append(pd.Series(ar, index=rel_days, name=idx))
            event_results.append({
                'event_id': idx,
                'ticker': ticker,
                'event_date': result.get('event_date', event_date),
                'car_total': result['car_total'],
                'p_value': result['p_value'],
                'method': method
            })

        if not abnormal_matrix:
            raise ValueError("No valid events found for batch event study")

        abnormal_df = pd.DataFrame(abnormal_matrix)
        caar = abnormal_df.mean(axis=0)
        caar_cum = caar.cumsum()

        return {
            'event_results': pd.DataFrame(event_results),
            'abnormal_matrix': abnormal_df,
            'caar': caar,
            'caar_cum': caar_cum
        }

    @staticmethod
    def placebo_randomization(
        events_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        event_date_col: str = 'event_trade_date',
        ticker_col: str = 'ticker',
        date_col: str = 'Date',
        returns_col: str = 'returns',
        event_window: Tuple[int, int] = (-5, 10),
        n_iter: int = 500,
        random_state: int = 42
    ) -> Dict:
        """Placebo test: randomize event dates and compare to observed CAAR."""
        rng = np.random.default_rng(random_state)
        data = returns_df.copy()
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data[ticker_col] = data[ticker_col].astype(str).str.upper().str.strip()

        trading_dates = {
            t: data[data[ticker_col] == t][date_col].dropna().sort_values().values
            for t in data[ticker_col].unique()
        }

        def sample_date(ticker):
            dates = trading_dates.get(ticker)
            if dates is None or len(dates) == 0:
                return pd.NaT
            start = max(0, -event_window[0])
            end = len(dates) - event_window[1] - 1
            if end <= start:
                return pd.NaT
            idx = rng.integers(start, end)
            return pd.Timestamp(dates[idx])

        observed = EventStudy.batch_event_study(
            events_df, returns_df, benchmark_returns,
            event_date_col=event_date_col, ticker_col=ticker_col,
            date_col=date_col, returns_col=returns_col,
            event_window=event_window, method='market_adjusted'
        )
        observed_caar_total = observed['caar_cum'].iloc[-1]

        placebo_totals = []
        for _ in range(n_iter):
            randomized = events_df.copy()
            randomized[event_date_col] = randomized[ticker_col].apply(sample_date)
            try:
                result = EventStudy.batch_event_study(
                    randomized, returns_df, benchmark_returns,
                    event_date_col=event_date_col, ticker_col=ticker_col,
                    date_col=date_col, returns_col=returns_col,
                    event_window=event_window, method='market_adjusted'
                )
                placebo_totals.append(result['caar_cum'].iloc[-1])
            except Exception:
                continue

        placebo_series = pd.Series(placebo_totals)
        p_value = (abs(placebo_series) >= abs(observed_caar_total)).mean() if not placebo_series.empty else np.nan

        return {
            'observed_caar_total': observed_caar_total,
            'placebo_distribution': placebo_series,
            'p_value': p_value
        }


class AdoptionIntensityModel:
    """Ridge model for outcome-calibrated intensity scores using ticker splits."""

    @staticmethod
    def build_feature_matrix(df, feature_cols, categorical_cols=None, drop_first=True):
        X = df[feature_cols].copy()
        if categorical_cols:
            dummies = pd.get_dummies(df[categorical_cols], drop_first=drop_first)
            X = pd.concat([X, dummies], axis=1)
        return X

    @staticmethod
    def fit_ridge_intensity_model(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        group_col: str = 'ticker',
        categorical_cols: list = None,
        alphas: list = None,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict:
        """Fit Ridge with ticker-split CV and return intensity scores."""
        if alphas is None:
            alphas = [0.1, 1.0, 10.0, 50.0, 100.0]

        required_cols = feature_cols + (categorical_cols or [])
        data = df.dropna(subset=[target_col] + required_cols).copy()

        X = AdoptionIntensityModel.build_feature_matrix(
            data, feature_cols, categorical_cols=categorical_cols
        )
        y = data[target_col].values
        groups = data[group_col].astype(str).values

        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for ticker-split validation")
        gkf = GroupKFold(n_splits=min(n_splits, n_groups))

        alpha_scores = {}
        for alpha in alphas:
            rmses = []
            for train_idx, val_idx in gkf.split(X, y, groups=groups):
                model = Ridge(alpha=alpha, random_state=random_state)
                model.fit(X.iloc[train_idx], y[train_idx])
                preds = model.predict(X.iloc[val_idx])
                rmse = mean_squared_error(y[val_idx], preds, squared=False)
                rmses.append(rmse)
            alpha_scores[alpha] = float(np.mean(rmses))

        best_alpha = min(alpha_scores, key=alpha_scores.get)
        final_model = Ridge(alpha=best_alpha, random_state=random_state)
        final_model.fit(X, y)

        preds = final_model.predict(X)
        metrics = {
            'rmse': float(mean_squared_error(y, preds, squared=False)),
            'r2': float(r2_score(y, preds))
        }

        result = data.copy()
        result['intensity_score'] = preds

        return {
            'model': final_model,
            'best_alpha': best_alpha,
            'alpha_scores': alpha_scores,
            'metrics': metrics,
            'predictions': result['intensity_score'],
            'feature_columns': list(X.columns),
            'scored_df': result
        }


class RobustRegressionUtils:
    """Outlier detection and heteroskedasticity tests."""

    @staticmethod
    def detect_outliers(df: pd.DataFrame,
                       outcome_col: str,
                       X_cols: list,
                       method: str = 'cooks_distance') -> pd.Series:
        """Detect influential outliers via Cook's distance or DFFITS."""
        X = sm.add_constant(df[X_cols])
        y = df[outcome_col]

        model = sm.OLS(y, X).fit()
        influence = model.get_influence()

        if method == 'cooks_distance':
            cooks_d = influence.cooks_distance[0]
            threshold = 4 / len(df)
            return cooks_d > threshold
        elif method == 'dffits':
            dffits = influence.dffits[0]
            threshold = 2 * np.sqrt(len(X_cols) / len(df))
            return np.abs(dffits) > threshold
        else:
            raise ValueError("method must be 'cooks_distance' or 'dffits'")

    @staticmethod
    def check_heteroskedasticity(df: pd.DataFrame,
                                outcome_col: str,
                                X_cols: list) -> Dict:
        from statsmodels.stats.diagnostic import het_breuschpagan

        X = sm.add_constant(df[X_cols])
        y = df[outcome_col]
        model = sm.OLS(y, X).fit()

        bp_test = het_breuschpagan(model.resid, X)

        return {
            'test_statistic': bp_test[0],
            'p_value': bp_test[1],
            'heteroskedastic': bp_test[1] < 0.05,
        }


class GARCHAnalysis:
    """GARCH volatility models."""

    @staticmethod
    def test_volatility_clustering(returns: pd.Series,
                                   lags: int = 20,
                                   alpha: float = 0.05) -> Dict:
        """ARCH-LM test for volatility clustering. If significant, GARCH is appropriate."""
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        returns = returns.dropna()

        if len(returns) < 100:
            warnings.warn(f"Sample size ({len(returns)}) is small for ARCH test.")

        try:
            arch_test = het_arch(returns, nlags=lags)
            test_stat = arch_test[0]
            p_value = arch_test[1]
        except Exception as e:
            warnings.warn(f"ARCH-LM test failed: {e}")
            test_stat = np.nan
            p_value = np.nan

        returns_squared = returns ** 2
        acf_values = acf(returns_squared, nlags=min(lags, len(returns)//5))

        lb_result = acorr_ljungbox(returns_squared, lags=min(lags, len(returns)//5), return_df=True)
        lb_pval = lb_result['lb_pvalue'].iloc[-1]

        has_arch = p_value < alpha if not np.isnan(p_value) else False

        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'has_arch_effects': has_arch,
            'acf_squared_returns': acf_values,
            'ljung_box_pvalue': lb_pval,
        }

    @staticmethod
    def fit_garch(returns: pd.Series,
                  p: int = 1,
                  q: int = 1,
                  mean_model: str = 'Constant',
                  vol_model: str = 'GARCH',
                  dist: str = 't') -> Dict:
        """Fit GARCH(p,q). Returns params, conditional vol, persistence, half-life, etc."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("pip install arch")

        from statsmodels.stats.diagnostic import acorr_ljungbox

        returns = returns.dropna()

        if len(returns) < 100:
            warnings.warn(f"Small sample ({len(returns)}) for GARCH.")

        if mean_model == 'AR':
            mean = 'ARX'
        elif mean_model == 'Zero':
            mean = 'Zero'
        else:
            mean = 'Constant'

        model = arch_model(returns, mean=mean, vol=vol_model, p=p, q=q, dist=dist)
        fitted_model = model.fit(disp='off', show_warning=False)

        params = fitted_model.params
        cond_vol = fitted_model.conditional_volatility
        std_resid = fitted_model.std_resid

        # persistence = sum(alpha) + sum(beta)
        alpha_params = [params[f'alpha[{i}]'] for i in range(1, q+1) if f'alpha[{i}]' in params]
        beta_params = [params[f'beta[{i}]'] for i in range(1, p+1) if f'beta[{i}]' in params]

        alpha_sum = sum(alpha_params) if alpha_params else 0
        beta_sum = sum(beta_params) if beta_params else 0
        persistence = alpha_sum + beta_sum

        half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf

        # check remaining ARCH effects
        lb_test = acorr_ljungbox(std_resid**2, lags=min(20, len(std_resid)//5), return_df=True)
        lb_pval = lb_test['lb_pvalue'].iloc[-1]

        return {
            'model': fitted_model,
            'params': params.to_dict(),
            'conditional_volatility': cond_vol,
            'standardized_residuals': std_resid,
            'persistence': persistence,
            'half_life': half_life,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'loglikelihood': fitted_model.loglikelihood,
            'ljung_box_pval': lb_pval,
            'converged': fitted_model.convergence_flag == 0
        }

    @staticmethod
    def fit_egarch(returns: pd.Series, p: int = 1, q: int = 1) -> Dict:
        """EGARCH for asymmetric volatility (leverage effect). gamma < 0 means negative shocks increase vol more."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("pip install arch")

        returns = returns.dropna()

        model = arch_model(returns, mean='Constant', vol='EGARCH', p=p, q=q, dist='t')
        fitted_model = model.fit(disp='off', show_warning=False)

        params = fitted_model.params
        pvalues = fitted_model.pvalues

        gamma_params = [params[f'gamma[{i}]'] for i in range(1, q+1) if f'gamma[{i}]' in params]
        gamma_pvalues = [pvalues[f'gamma[{i}]'] for i in range(1, q+1) if f'gamma[{i}]' in params]

        if gamma_params:
            gamma_1 = gamma_params[0]
            gamma_1_pval = gamma_pvalues[0]
            leverage_effect = (gamma_1 < 0) and (gamma_1_pval < 0.05)
        else:
            leverage_effect = False

        base_results = GARCHAnalysis.fit_garch(returns, p=p, q=q, mean_model='Constant', vol_model='EGARCH', dist='t')

        base_results.update({
            'gamma_params': gamma_params,
            'gamma_pvalues': gamma_pvalues,
            'leverage_effect': leverage_effect,
        })

        return base_results

    @staticmethod
    def volatility_forecast(fitted_model, horizon: int = 10,
                           method: str = 'analytic',
                           n_simulations: int = 1000,
                           confidence_level: float = 0.95) -> Dict:
        """Forecast conditional volatility h-steps ahead."""
        model_obj = fitted_model['model']
        params = fitted_model['params']
        persistence = fitted_model['persistence']

        if method == 'analytic':
            forecasts = model_obj.forecast(horizon=horizon, reindex=False)
            mean_forecast = np.sqrt(forecasts.variance.values[-1, :])
            variance_forecast = forecasts.variance.values[-1, :]

            se = np.sqrt(variance_forecast) * 0.1
            alpha = 1 - confidence_level
            z_crit = stats.norm.ppf(1 - alpha/2)
            ci_lower = mean_forecast - z_crit * se
            ci_upper = mean_forecast + z_crit * se
            forecast_paths = None

        elif method == 'simulation':
            simulations = model_obj.forecast(horizon=horizon, method='simulation',
                                            simulations=n_simulations, reindex=False)
            sim_values = simulations.simulations.values

            mean_forecast = np.sqrt(sim_values.var(axis=2).mean(axis=0))
            variance_forecast = sim_values.var(axis=2).mean(axis=0)

            alpha = 1 - confidence_level
            ci_lower = np.percentile(np.sqrt(sim_values.var(axis=2)), 100*alpha/2, axis=0)
            ci_upper = np.percentile(np.sqrt(sim_values.var(axis=2)), 100*(1-alpha/2), axis=0)
            forecast_paths = sim_values
        else:
            raise ValueError("method must be 'analytic' or 'simulation'")

        # unconditional variance
        if 'omega' in params and persistence < 1:
            uncond_var = params['omega'] / (1 - persistence)
        else:
            uncond_var = np.nan

        return {
            'mean_forecast': mean_forecast,
            'variance_forecast': variance_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'forecast_paths': forecast_paths,
            'unconditional_variance': uncond_var,
            'horizon': horizon,
            'method': method
        }

    @staticmethod
    def abnormal_volatility_event_study(stock_returns: pd.Series,
                                        event_date: pd.Timestamp,
                                        estimation_window: int = 120,
                                        event_window: Tuple[int, int] = (-5, 10)) -> Dict:
        """Event study for volatility: fit GARCH pre-event, forecast expected vol, compute abnormal vol."""
        if not isinstance(stock_returns.index, pd.DatetimeIndex):
            raise ValueError("stock_returns must have a DatetimeIndex")

        try:
            event_idx = stock_returns.index.get_loc(event_date)
        except KeyError:
            event_idx = stock_returns.index.get_indexer([event_date], method='nearest')[0]

        est_start = event_idx - estimation_window - abs(event_window[0])
        est_end = event_idx + event_window[0] - 1

        if est_start < 0:
            raise ValueError(f"Not enough data before event. Need {estimation_window + abs(event_window[0])} obs.")

        est_returns = stock_returns.iloc[est_start:est_end]

        garch_fit = GARCHAnalysis.fit_garch(est_returns, p=1, q=1, dist='t')

        event_start = event_idx + event_window[0]
        event_end = event_idx + event_window[1] + 1

        if event_end > len(stock_returns):
            raise ValueError(f"Not enough data after event.")

        event_returns = stock_returns.iloc[event_start:event_end]

        n_periods = len(event_returns)
        vol_forecast = GARCHAnalysis.volatility_forecast(garch_fit, horizon=n_periods, method='analytic')
        expected_vol = vol_forecast['mean_forecast']

        actual_vol = np.abs(event_returns.values)
        abnormal_vol = actual_vol - expected_vol
        cumulative_abnormal_vol = abnormal_vol.sum()

        vol_mean = abnormal_vol.mean()
        vol_se = abnormal_vol.std() / np.sqrt(len(abnormal_vol))
        t_stat = vol_mean / vol_se if vol_se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(abnormal_vol) - 1))

        return {
            'expected_volatility': expected_vol,
            'actual_volatility': actual_vol,
            'abnormal_volatility': abnormal_vol,
            'cumulative_abnormal_vol': cumulative_abnormal_vol,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'garch_model': garch_fit,
            'event_window_dates': event_returns.index
        }


class VARAnalysis:
    """Vector Autoregression for multivariate time series."""

    @staticmethod
    def fit_var(data: pd.DataFrame,
               var_cols: list,
               maxlags: int = 10,
               ic: str = 'bic',
               trend: str = 'c',
               check_stationarity: bool = True) -> Dict:
        """Fit VAR with automatic lag selection. Returns model, coefficients, stability check, Granger matrix."""
        from statsmodels.tsa.api import VAR

        var_data = data[var_cols].dropna()

        if len(var_data) < 50:
            warnings.warn(f"Small sample ({len(var_data)}) for VAR.")

        if check_stationarity:
            for col in var_cols:
                adf_result = adfuller(var_data[col], autolag='AIC')
                if adf_result[1] > 0.05:
                    warnings.warn(f"'{col}' appears non-stationary (p={adf_result[1]:.4f}). Consider differencing.")

        model = VAR(var_data)
        lag_order = model.select_order(maxlags=maxlags)
        selected_lag = getattr(lag_order, ic)
        fitted_model = model.fit(maxlags=selected_lag, ic=None, trend=trend)

        # coefficients
        coefficients = {}
        for i in range(1, selected_lag + 1):
            coefficients[f'A_{i}'] = fitted_model.params.iloc[(i-1)*len(var_cols):i*len(var_cols)].values.T

        # stability
        try:
            eigenvalues = np.abs(np.linalg.eigvals(fitted_model.get_eq_index('companion')))
            is_stable = np.all(eigenvalues < 1.0)
            max_eigenvalue = np.max(eigenvalues)
        except:
            eigenvalues = []
            is_stable = None
            max_eigenvalue = np.nan

        residuals = fitted_model.resid
        resid_corr = residuals.corr()

        try:
            portmanteau = fitted_model.test_whiteness(nlags=min(20, len(residuals)//5))
            portmanteau_pval = portmanteau.pvalue
        except:
            portmanteau_pval = np.nan

        # granger causality matrix
        granger_matrix = pd.DataFrame(index=var_cols, columns=var_cols, dtype=float)
        for cause in var_cols:
            for effect in var_cols:
                if cause != effect:
                    try:
                        gc_result = fitted_model.test_causality(effect, cause, kind='f')
                        granger_matrix.loc[cause, effect] = gc_result.pvalue
                    except:
                        granger_matrix.loc[cause, effect] = np.nan
                else:
                    granger_matrix.loc[cause, effect] = np.nan

        return {
            'model': fitted_model,
            'selected_lag': selected_lag,
            'coefficients': coefficients,
            'residual_correlation': resid_corr,
            'stability_check': {
                'is_stable': is_stable,
                'max_eigenvalue': max_eigenvalue,
                'all_eigenvalues': eigenvalues
            },
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'hqic': fitted_model.hqic,
            'fpe': fitted_model.fpe,
            'n_obs': fitted_model.nobs,
            'df_model': fitted_model.df_model,
            'portmanteau_pvalue': portmanteau_pval,
            'granger_causality_matrix': granger_matrix
        }

    @staticmethod
    def impulse_response(var_result,
                        periods: int = 20,
                        impulse: Optional[str] = None,
                        response: Optional[str] = None,
                        orthogonalized: bool = True,
                        bootstrap: bool = False,
                        n_bootstrap: int = 1000,
                        alpha: float = 0.05) -> Dict:
        """Impulse response functions. Shock one variable, see how others respond over time."""
        model = var_result['model']

        if orthogonalized:
            irf_obj = model.irf(periods)
        else:
            irf_obj = model.irf(periods, orth=False)

        if impulse is not None and response is not None:
            impulse_idx = list(model.names).index(impulse)
            response_idx = list(model.names).index(response)
            irf_values = irf_obj.irfs[:, response_idx, impulse_idx]
            irf_df = pd.DataFrame({'IRF': irf_values}, index=range(periods+1))
        else:
            irf_values = irf_obj.irfs
            irf_df = None

        if bootstrap:
            try:
                if orthogonalized:
                    irf_ci = irf_obj.err_bands_mc(orth=True, sims=n_bootstrap, signif=alpha)
                else:
                    irf_ci = irf_obj.err_bands_mc(orth=False, sims=n_bootstrap, signif=alpha)

                if impulse is not None and response is not None:
                    ci_lower = irf_ci[0][:, response_idx, impulse_idx]
                    ci_upper = irf_ci[1][:, response_idx, impulse_idx]
                else:
                    ci_lower = irf_ci[0]
                    ci_upper = irf_ci[1]
            except Exception:
                ci_lower = None
                ci_upper = None
        else:
            ci_lower = None
            ci_upper = None

        if impulse is not None and response is not None:
            cumulative_irf = np.cumsum(irf_values)
        else:
            cumulative_irf = np.cumsum(irf_values, axis=0)

        return {
            'irf': irf_df if irf_df is not None else irf_values,
            'irf_ci_lower': ci_lower,
            'irf_ci_upper': ci_upper,
            'cumulative_irf': cumulative_irf,
            'periods': periods,
            'impulse': impulse,
            'response': response
        }

    @staticmethod
    def forecast_error_variance_decomposition(var_result,
                                             periods: int = 20,
                                             orthogonalized: bool = True) -> Dict:
        """FEVD: what % of forecast variance in Y is due to shocks in X?"""
        model = var_result['model']

        if orthogonalized:
            fevd_obj = model.fevd(periods)
        else:
            fevd_obj = model.fevd(periods, orth=False)

        var_names = model.names
        fevd_dict = {}
        for i, var in enumerate(var_names):
            fevd_dict[var] = fevd_obj.decomp[:, i, :]

        key_horizons = [h for h in [1, 5, 10, min(20, periods)] if h <= periods]
        key_horizons = list(dict.fromkeys(key_horizons))
        available_horizons = fevd_dict[var_names[0]].shape[0]
        key_horizons = [h for h in key_horizons if h <= available_horizons]
        key_indices = [h - 1 for h in key_horizons]

        fevd_summary = {}
        for var in var_names:
            fevd_summary[var] = pd.DataFrame(
                fevd_dict[var][key_indices],
                index=key_horizons,
                columns=var_names
            )

        return {
            'fevd': fevd_dict,
            'fevd_summary': fevd_summary,
            'periods': periods,
            'var_names': var_names
        }

    @staticmethod
    def var_granger_causality(var_result,
                             cause_var: str,
                             response_var: str,
                             signif: float = 0.05) -> Dict:
        """Granger causality within VAR framework."""
        model = var_result['model']
        selected_lag = var_result['selected_lag']

        gc_result = model.test_causality(response_var, cause_var, kind='f')

        test_stat = gc_result.test_statistic
        p_value = gc_result.pvalue
        df = gc_result.df
        granger_causes = p_value < signif
        if granger_causes:
            interpretation = f"{cause_var} Granger-causes {response_var} (p={p_value:.4g})"
        else:
            interpretation = f"No evidence that {cause_var} Granger-causes {response_var} (p={p_value:.4g})"

        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'df': df,
            'granger_causes': granger_causes,
            'lags_tested': selected_lag,
            'interpretation': interpretation,
        }

    @staticmethod
    def var_forecast(var_result, steps: int = 10, alpha: float = 0.05) -> Dict:
        """Forecast all variables h-steps ahead with CIs."""
        model = var_result['model']

        history = getattr(model, 'y', None)
        if history is None:
            history = model.endog
        forecast_result = model.forecast(history[-model.k_ar:], steps=steps)
        forecast_df = pd.DataFrame(forecast_result, columns=model.names)

        residuals = model.resid
        resid_cov = np.cov(residuals.T)

        forecast_se = []
        for h in range(1, steps + 1):
            se = np.sqrt(np.diag(resid_cov) * h)
            forecast_se.append(se)
        forecast_se = np.array(forecast_se)

        z_crit = stats.norm.ppf(1 - alpha/2)
        ci_lower = forecast_result - z_crit * forecast_se
        ci_upper = forecast_result + z_crit * forecast_se

        return {
            'forecast': forecast_df,
            'forecast_ci_lower': pd.DataFrame(ci_lower, columns=model.names),
            'forecast_ci_upper': pd.DataFrame(ci_upper, columns=model.names),
            'forecast_se': pd.DataFrame(forecast_se, columns=model.names),
            'steps': steps
        }


if __name__ == "__main__":
    print("Import and use in your analysis.")
