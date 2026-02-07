# LaTeX export utilities

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class LaTeXExporter:
    """Export regression results to LaTeX tables."""

    @staticmethod
    def regression_to_latex(results,
                           model_name: str = "Model",
                           include_se: bool = True,
                           stars: bool = True,
                           include_stats: list = ['rsquared', 'nobs', 'fstat'],
                           decimal_places: int = 3,
                           se_format: str = 'parentheses') -> str:
        """Single regression result to LaTeX table with significance stars."""
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

        if hasattr(results, 'params'):
            names = None
            if hasattr(results, 'model') and hasattr(results.model, 'exog_names'):
                names = list(results.model.exog_names)
            params = _as_series(results.params, names)
            se = _as_series(results.bse, params.index)
            pvalues = _as_series(results.pvalues, params.index)
        elif isinstance(results, dict):
            params = pd.Series(results['params']) if isinstance(results['params'], dict) else results['params']
            params = _as_series(params)
            se = pd.Series(results['std_errors']) if isinstance(results.get('std_errors', {}), dict) else results.get('bse', params * 0)
            se = _as_series(se, params.index)
            pvalues = _as_series(results.get('pvalues', [1]*len(params)), params.index)
        else:
            raise TypeError("results must be statsmodels/linearmodels result or dict")

        def add_stars(pval):
            if stars:
                if pval < 0.01: return '***'
                elif pval < 0.05: return '**'
                elif pval < 0.10: return '*'
            return ''

        coef_str = []
        for var in params.index:
            coef = f"{params[var]:.{decimal_places}f}"
            coef += add_stars(pvalues[var])

            if se_format == 'brackets':
                se_str = f"[{se[var]:.{decimal_places}f}]"
            else:
                se_str = f"({se[var]:.{decimal_places}f})"

            coef_str.append((var, coef, se_str))

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Regression Results}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            f"& {model_name} \\\\",
            "\\midrule"
        ]

        for var, coef, se_str in coef_str:
            var_escaped = var.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
            lines.append(f"{var_escaped} & {coef} \\\\")
            if include_se:
                lines.append(f"& {se_str} \\\\")

        lines.append("\\midrule")

        for stat in include_stats:
            if stat == 'rsquared' and hasattr(results, 'rsquared'):
                lines.append(f"R² & {results.rsquared:.3f} \\\\")
            elif stat == 'rsquared_adj' and hasattr(results, 'rsquared_adj'):
                lines.append(f"Adjusted R² & {results.rsquared_adj:.3f} \\\\")
            elif stat == 'rsquared_within' and hasattr(results, 'rsquared_within'):
                lines.append(f"Within R² & {results.rsquared_within:.3f} \\\\")
            elif stat == 'nobs' and hasattr(results, 'nobs'):
                lines.append(f"N & {int(results.nobs):,} \\\\")
            elif stat == 'fstat':
                if hasattr(results, 'fvalue'):
                    lines.append(f"F-statistic & {results.fvalue:.2f} \\\\")
                elif hasattr(results, 'f_statistic'):
                    f_stat = results.f_statistic.stat if hasattr(results.f_statistic, 'stat') else results.f_statistic
                    lines.append(f"F-statistic & {f_stat:.2f} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\multicolumn{2}{l}{\\footnotesize *** p<0.01, ** p<0.05, * p<0.1}",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(lines)

    @staticmethod
    def multiple_models_to_latex(results_dict: dict,
                                table_title: str = "Regression Results",
                                label: str = "tab:results",
                                caption: str = None,
                                include_stats: list = ['rsquared', 'nobs'],
                                decimal_places: int = 3) -> str:
        caption_text = caption if caption else table_title
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

        # get all unique variables
        all_vars = []
        for model_name, result in results_dict.items():
            if hasattr(result, 'params'):
                names = None
                if hasattr(result, 'model') and hasattr(result.model, 'exog_names'):
                    names = list(result.model.exog_names)
                params = _as_series(result.params, names)
                all_vars.extend(params.index.tolist())
            elif isinstance(result, dict) and 'params' in result:
                params = result['params']
                if isinstance(params, dict):
                    all_vars.extend(params.keys())
                else:
                    params = _as_series(params)
                    all_vars.extend(params.index.tolist())
        all_vars = list(dict.fromkeys(all_vars))

        def add_stars(pval):
            if pval < 0.01: return '***'
            elif pval < 0.05: return '**'
            elif pval < 0.10: return '*'
            return ''

        model_data = {}
        for model_name, result in results_dict.items():
            if hasattr(result, 'params'):
                names = None
                if hasattr(result, 'model') and hasattr(result.model, 'exog_names'):
                    names = list(result.model.exog_names)
                params = _as_series(result.params, names)
                se = _as_series(result.bse, params.index)
                pvalues = _as_series(result.pvalues, params.index)
            elif isinstance(result, dict):
                params = pd.Series(result['params']) if isinstance(result['params'], dict) else result['params']
                params = _as_series(params)
                se = pd.Series(result.get('std_errors', {})) if isinstance(result.get('std_errors', {}), dict) else result.get('bse', params * 0)
                se = _as_series(se, params.index)
                pvalues = _as_series(result.get('pvalues', [1]*len(params)), params.index)
            else:
                continue
            model_data[model_name] = {'params': params, 'se': se, 'pvalues': pvalues, 'result': result}

        n_models = len(results_dict)
        col_spec = 'l' + 'c' * n_models

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption_text}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule"
        ]

        header = " & ".join([""] + list(results_dict.keys())) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for var in all_vars:
            var_escaped = var.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
            coef_row = [var_escaped]
            se_row = [""]

            for model_name in results_dict.keys():
                if model_name in model_data and var in model_data[model_name]['params'].index:
                    p = model_data[model_name]['params']
                    s = model_data[model_name]['se']
                    pv = model_data[model_name]['pvalues']
                    coef_row.append(f"{p[var]:.{decimal_places}f}" + add_stars(pv[var]))
                    se_row.append(f"({s[var]:.{decimal_places}f})")
                else:
                    coef_row.append("—")
                    se_row.append("")

            lines.append(" & ".join(coef_row) + " \\\\")
            lines.append(" & ".join(se_row) + " \\\\")

        lines.append("\\midrule")

        for stat in include_stats:
            stat_row = []
            if stat == 'rsquared':
                stat_row.append("R²")
                for mn in results_dict.keys():
                    r = model_data[mn]['result']
                    stat_row.append(f"{r.rsquared:.3f}" if hasattr(r, 'rsquared') else "—")
            elif stat == 'nobs':
                stat_row.append("N")
                for mn in results_dict.keys():
                    r = model_data[mn]['result']
                    stat_row.append(f"{int(r.nobs):,}" if hasattr(r, 'nobs') else "—")
            if stat_row:
                lines.append(" & ".join(stat_row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            f"\\multicolumn{{{n_models+1}}}{{l}}{{\\footnotesize *** p<0.01, ** p<0.05, * p<0.1}}",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(lines)

    @staticmethod
    def summary_statistics_to_latex(df: pd.DataFrame,
                                   vars_to_include: list = None,
                                   stats: list = ['mean', 'std', 'min', 'max', 'count'],
                                   var_labels: dict = None,
                                   decimal_places: int = 2,
                                   caption: str = "Summary Statistics",
                                   label: str = "tab:summary") -> str:
        """Summary statistics table (typical Table 1 in papers)."""
        if vars_to_include is None:
            vars_to_include = df.select_dtypes(include=[np.number]).columns.tolist()
        if var_labels is None:
            var_labels = {var: var for var in vars_to_include}

        stat_map = {
            'mean': ('Mean', lambda s: s.mean()),
            'std': ('Std. Dev.', lambda s: s.std()),
            'median': ('Median', lambda s: s.median()),
            'min': ('Min', lambda s: s.min()),
            'max': ('Max', lambda s: s.max()),
            'count': ('N', lambda s: s.count()),
            'p25': ('25th %', lambda s: s.quantile(0.25)),
            'p75': ('75th %', lambda s: s.quantile(0.75)),
        }

        col_headers = [stat_map[s][0] for s in stats if s in stat_map]
        n_cols = len(col_headers)
        col_spec = 'l' + 'c' * n_cols

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            "Variable & " + " & ".join(col_headers) + " \\\\",
            "\\midrule"
        ]

        for var in vars_to_include:
            row = [var_labels.get(var, var)]
            for stat in stats:
                if stat in stat_map:
                    val = stat_map[stat][1](df[var])
                    if stat == 'count':
                        row.append(f"{int(val):,}")
                    else:
                        row.append(f"{val:.{decimal_places}f}")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    print("Import: from utils import LaTeXExporter")
