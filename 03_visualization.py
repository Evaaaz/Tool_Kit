# Visualization helpers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

# plot defaults
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ExploratoryVisuals:
    """EDA visualizations."""

    @staticmethod
    def time_series_plot(df, date_col, value_cols, title="Time Series",
                        ylabel="Value", figsize=(12, 6), save_path=None):
        """Multi-series time series plot."""
        fig, ax = plt.subplots(figsize=figsize)

        for col in value_cols:
            ax.plot(df[date_col], df[col], label=col, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def correlation_heatmap(df, cols=None, title="Correlation Matrix",
                           figsize=(10, 8), save_path=None):
        """Correlation heatmap with upper triangle masked."""
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontweight='bold', pad=20)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def distribution_comparison(df, value_col, group_col,
                               title="Distribution Comparison",
                               figsize=(12, 5), save_path=None):
        """Side-by-side violin + box plots for comparing groups."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.violinplot(data=df, x=group_col, y=value_col, ax=axes[0])
        axes[0].set_title('Violin Plot', fontweight='bold')
        axes[0].set_xlabel(group_col, fontweight='bold')
        axes[0].set_ylabel(value_col, fontweight='bold')

        sns.boxplot(data=df, x=group_col, y=value_col, ax=axes[1])
        axes[1].set_title('Box Plot', fontweight='bold')
        axes[1].set_xlabel(group_col, fontweight='bold')
        axes[1].set_ylabel(value_col, fontweight='bold')

        fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


class AnalysisVisuals:
    """Visualizations for statistical analyses."""

    @staticmethod
    def event_study_plot(abnormal_returns, car, event_date_label="Event",
                        title="Event Study: Cumulative Abnormal Returns",
                        figsize=(12, 6), save_path=None):
        """Event study plot: daily ARs + cumulative ARs."""
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        dates = abnormal_returns.index
        axes[0].bar(dates, abnormal_returns, color='steelblue', alpha=0.6)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0].set_ylabel('Abnormal Return', fontweight='bold')
        axes[0].set_title('Daily Abnormal Returns', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(dates, car, color='darkred', linewidth=2)
        axes[1].fill_between(dates, car, 0, alpha=0.3, color='darkred')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Event Time (Days)', fontweight='bold')
        axes[1].set_ylabel('CAR', fontweight='bold')
        axes[1].set_title('Cumulative Abnormal Returns (CAR)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        if 0 in dates:
            for ax in axes:
                ax.axvline(x=0, color='red', linestyle='-', linewidth=2, label=event_date_label)
                ax.legend()

        fig.suptitle(title, fontweight='bold', fontsize=14, y=1.00)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def did_visualization(df, outcome_col, treatment_col, time_col,
                         intervention_time, title="Difference-in-Differences",
                         figsize=(10, 6), save_path=None):
        """DiD parallel trends visualization."""
        fig, ax = plt.subplots(figsize=figsize)

        means = df.groupby([time_col, treatment_col])[outcome_col].mean().unstack()

        ax.plot(means.index, means[0], 'o-', label='Control Group',
               linewidth=2, markersize=6, color='steelblue')
        ax.plot(means.index, means[1], 's-', label='Treatment Group',
               linewidth=2, markersize=6, color='darkred')

        ax.axvline(x=intervention_time, color='black', linestyle='--',
                  linewidth=2, label='Intervention')

        ax.set_xlabel(time_col.capitalize(), fontweight='bold')
        ax.set_ylabel(outcome_col, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def sector_comparison_plot(df, sector_col, value_col,
                              title="Sector Comparison",
                              figsize=(12, 6), save_path=None):
        """Horizontal bar plot comparing sectors, color-coded by sign."""
        sector_means = df.groupby(sector_col)[value_col].mean().sort_values()

        fig, ax = plt.subplots(figsize=figsize)
        colors = ['green' if x > 0 else 'red' for x in sector_means.values]

        ax.barh(sector_means.index, sector_means.values, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel(value_col, fontweight='bold')
        ax.set_ylabel('Sector', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def regression_diagnostics(model_results, figsize=(12, 10), save_path=None):
        """Four-panel regression diagnostics: residuals vs fitted, QQ, scale-location, leverage."""
        from scipy import stats

        residuals = model_results.resid
        fitted = model_results.fittedvalues

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # residuals vs fitted
        axes[0, 0].scatter(fitted, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # QQ plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # scale-location
        standardized_resid = np.sqrt(np.abs(residuals / residuals.std()))
        axes[1, 0].scatter(fitted, standardized_resid, alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values', fontweight='bold')
        axes[1, 0].set_ylabel('sqrt(|Standardized Residuals|)', fontweight='bold')
        axes[1, 0].set_title('Scale-Location', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # leverage
        try:
            influence = model_results.get_influence()
            leverage = influence.hat_matrix_diag
            axes[1, 1].scatter(leverage, residuals, alpha=0.5)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_xlabel('Leverage', fontweight='bold')
            axes[1, 1].set_ylabel('Residuals', fontweight='bold')
            axes[1, 1].set_title('Residuals vs Leverage', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Leverage not available',
                          ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def coefficient_plot(results, exclude_vars=None, alpha=0.05,
                        sort_by_magnitude=False, figsize=(10, 8),
                        colors=('steelblue', 'lightgray'), save_path=None):
        """Coefficient plot with confidence intervals. Works with statsmodels, linearmodels, or dict."""
        def _as_series(values, names=None):
            if isinstance(values, pd.Series):
                return values
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
                return values
            arr = np.asarray(values).reshape(-1)
            if names is None:
                names = [f"x{i}" for i in range(len(arr))]
            return pd.Series(arr, index=names)

        # extract params
        if hasattr(results, 'params'):
            names = None
            if hasattr(results, 'model') and hasattr(results.model, 'exog_names'):
                names = list(results.model.exog_names)
            params = _as_series(results.params, names)
            try:
                ci = results.conf_int(alpha=alpha)
            except:
                ci = results.conf_int()
            if not isinstance(ci, pd.DataFrame):
                ci = pd.DataFrame(ci, index=params.index)
            pvalues = _as_series(results.pvalues, params.index)
        elif isinstance(results, dict):
            params = pd.Series(results['params']) if isinstance(results['params'], dict) else results['params']
            params = _as_series(params)
            if 'conf_int' in results:
                ci = results['conf_int']
            elif 'ci_lower' in results and 'ci_upper' in results:
                ci = pd.DataFrame({0: results['ci_lower'], 1: results['ci_upper']}, index=params.index)
            else:
                raise ValueError("Dict must have 'conf_int' or 'ci_lower'/'ci_upper'")
            if not isinstance(ci, pd.DataFrame):
                ci = pd.DataFrame(ci, index=params.index)
            pvalues = _as_series(results.get('pvalues', [1]*len(params)), params.index)
        else:
            raise TypeError("results must be a regression result or dict")

        if exclude_vars:
            params = params[~params.index.isin(exclude_vars)]
            ci = ci.loc[params.index]
            pvalues = pvalues.loc[params.index]

        is_significant = pvalues < alpha

        if sort_by_magnitude:
            sort_idx = params.abs().sort_values(ascending=True).index
            params = params[sort_idx]
            ci = ci.loc[sort_idx]
            is_significant = is_significant[sort_idx]

        fig, ax = plt.subplots(figsize=figsize)
        y_positions = np.arange(len(params))

        for i, var in enumerate(params.index):
            color = colors[0] if is_significant[var] else colors[1]
            ax.plot([ci.iloc[i, 0], ci.iloc[i, 1]], [i, i],
                   color=color, linewidth=2.5, alpha=0.8)
            ax.scatter(params[var], i, color=color, s=120, zorder=3,
                      alpha=0.9, edgecolors='white', linewidth=1.5)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(params.index, fontweight='bold')
        ax.set_xlabel('Coefficient Estimate', fontweight='bold', fontsize=12)
        ax.set_title(f'Regression Coefficients with {int((1-alpha)*100)}% CI',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], edgecolor='white', label=f'Significant (p < {alpha})'),
            Patch(facecolor=colors[1], edgecolor='white', label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, shadow=True, fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


class InteractiveVisuals:
    """Interactive plots using plotly (optional)."""

    @staticmethod
    def interactive_time_series(df, date_col, value_cols, title="Interactive Time Series"):
        """Interactive time series with zoom/pan. Requires plotly."""
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for col in value_cols:
                fig.add_trace(go.Scatter(x=df[date_col], y=df[col], mode='lines', name=col))
            fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value',
                            hovermode='x unified', template='plotly_white')
            fig.show()
        except ImportError:
            print("plotly not installed, run: pip install plotly")

    @staticmethod
    def interactive_scatter(df, x_col, y_col, color_col=None, size_col=None,
                           title="Interactive Scatter"):
        try:
            import plotly.express as px
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           size=size_col, title=title, hover_data=df.columns)
            fig.update_layout(template='plotly_white')
            fig.show()
        except ImportError:
            print("plotly not installed, run: pip install plotly")


if __name__ == "__main__":
    print("Import: from visualization import ExploratoryVisuals, AnalysisVisuals")
