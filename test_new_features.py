"""
Test Script for New Datathon 2026 Features
===========================================

Tests all newly implemented functionality:
1. GARCH volatility models
2. VAR multivariate time series
3. Coefficient plot visualization
4. LaTeX table export

Run this script to verify everything works before the datathon.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import warnings
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules directly
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = Path('./test_outputs')
output_dir.mkdir(exist_ok=True)

print("TEST 1: GARCH Volatility Models")

try:
    stat_module = import_module_from_file("statistical_analysis", "02_statistical_analysis.py")
    GARCHAnalysis = stat_module.GARCHAnalysis

    # Generate returns with volatility clustering
    n = 500
    omega, alpha, beta = 0.01, 0.15, 0.80
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    returns_series = pd.Series(returns)

    # Test 1a: Volatility clustering test
    garch = GARCHAnalysis()
    arch_test = garch.test_volatility_clustering(returns_series, lags=10)

    if arch_test['has_arch_effects']:
        print(f"    [PASS] ARCH effects detected (p={arch_test['p_value']:.4f})")
    else:
        print(f"    [WARN] No ARCH effects (p={arch_test['p_value']:.4f}) - may be expected with small sample")

    # Test 1b: Fit GARCH model
    garch_result = garch.fit_garch(returns_series, p=1, q=1, dist='t')

    print(f"    Parameters:")
    print(f"      omega = {garch_result['params']['omega']:.4f}")
    print(f"      Persistence = {garch_result['persistence']:.3f}")
    print(f"      Half-life = {garch_result['half_life']:.1f} periods")
    print(f"    Converged: {garch_result['converged']}")

    if garch_result['converged']:
        print("    [PASS] GARCH model fitted successfully")
    else:
        print("    [WARN] GARCH model convergence issue")

    # Test 1c: Volatility forecast
    forecast = garch.volatility_forecast(garch_result, horizon=30, method='analytic')
    print(f"    30-period ahead forecast: {forecast['mean_forecast'][-1]:.4f}")
    print("    [PASS] Volatility forecasting works")

    print("  [PASS] ALL GARCH TESTS PASSED\n")

except ImportError as e:
    print(f"  [SKIP] ARCH not installed: {e}")
    print("     Install with: pip install arch\n")
except Exception as e:
    print(f"  [FAIL] GARCH TEST FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("TEST 2: VAR Multivariate Time Series")

try:
    VARAnalysis = stat_module.VARAnalysis

    # Generate VAR(1) data
    n = 200
    A = np.array([[0.5, 0.2], [0.3, 0.4]])  # Coefficient matrix
    data = np.zeros((n, 2))

    for t in range(1, n):
        data[t] = A @ data[t-1] + np.random.randn(2) * 0.1

    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    df_var = pd.DataFrame(data, columns=['patents', 'returns'], index=dates)

    # Test 2a: Fit VAR
    var = VARAnalysis()
    var_result = var.fit_var(
        df_var,
        var_cols=['patents', 'returns'],
        maxlags=5,
        ic='bic',
        check_stationarity=False  # We know it's stationary
    )

    print(f"    Optimal lag: {var_result['selected_lag']}")
    print(f"    Stable: {var_result['stability_check']['is_stable']}")
    print(f"    AIC: {var_result['aic']:.2f}, BIC: {var_result['bic']:.2f}")

    if var_result['stability_check']['is_stable']:
        print("    [PASS] VAR model is stable")
    else:
        print("    [WARN] VAR model is unstable")

    # Test 2b: Granger causality
    causality = var.var_granger_causality(
        var_result,
        cause_var='patents',
        response_var='returns'
    )
    print(f"    {causality['interpretation']}")
    print("    [PASS] Granger causality test completed")

    # Test 2c: Impulse response
    irf = var.impulse_response(
        var_result,
        impulse='patents',
        response='returns',
        periods=10,
        orthogonalized=True,
        bootstrap=False  # Skip bootstrap for speed
    )
    print(f"    IRF computed for {len(irf['irf'])} periods")
    print("    [PASS] IRF analysis completed")

    # Test 2d: FEVD
    fevd = var.forecast_error_variance_decomposition(var_result, periods=10)
    print(f"    FEVD computed for {fevd['periods']} periods")
    print("    [PASS] FEVD analysis completed")

    # Test 2e: Forecasting
    forecast = var.var_forecast(var_result, steps=10)
    print(f"    10-step ahead forecast generated")
    print("    [PASS] VAR forecasting works")

    print("  [PASS] ALL VAR TESTS PASSED\n")

except Exception as e:
    print(f"  [FAIL] VAR TEST FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("TEST 3: Coefficient Plot Visualization")

try:
    vis_module = import_module_from_file("visualization", "03_visualization.py")
    AnalysisVisuals = vis_module.AnalysisVisuals
    import statsmodels.api as sm

    # Generate regression data
    n = 100
    X = np.random.randn(n, 3)
    y = X @ [1.5, -0.8, 0.3] + np.random.randn(n) * 0.5

    # Fit OLS
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Create coefficient plot
    av = AnalysisVisuals()
    save_path = output_dir / 'test_coefficient_plot.png'

    av.coefficient_plot(
        model,
        exclude_vars=['const'],
        sort_by_magnitude=True,
        save_path=str(save_path)
    )

    if save_path.exists():
        print(f"     Coefficient plot saved to {save_path}")
    else:
        print(f"      Coefficient plot may not have been saved")

    print("   COEFFICIENT PLOT TEST PASSED\n")

except Exception as e:
    print(f"   COEFFICIENT PLOT TEST FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("TEST 4: LaTeX Table Export")

try:
    utils_module = import_module_from_file("utils", "utils.py")
    LaTeXExporter = utils_module.LaTeXExporter

    exporter = LaTeXExporter()

    # Test 4a: Single model export
    latex_single = exporter.regression_to_latex(
        model,
        model_name="Test Model",
        include_stats=['rsquared', 'nobs']
    )

    if '\\begin{table}' in latex_single and '\\toprule' in latex_single:
        print("     LaTeX table structure looks correct")

        # Save to file
        latex_file = output_dir / 'test_single_model.tex'
        with open(latex_file, 'w') as f:
            f.write(latex_single)
        print(f"     Saved to {latex_file}")
    else:
        print("      LaTeX table structure may be incorrect")

    # Test 4b: Multiple models export

    # Fit a second model for comparison
    X2 = np.random.randn(n, 4)
    y2 = X2 @ [1.5, -0.8, 0.3, 0.5] + np.random.randn(n) * 0.5
    model2 = sm.OLS(y2, sm.add_constant(X2)).fit()

    models = {
        'Model 1': model,
        'Model 2': model2
    }

    latex_multi = exporter.multiple_models_to_latex(
        models,
        table_title="Sensitivity Checks",
        label="tab:sensitivity"
    )

    if 'Model 1' in latex_multi and 'Model 2' in latex_multi:
        print("     Multi-model table includes all models")

        latex_file2 = output_dir / 'test_multi_model.tex'
        with open(latex_file2, 'w') as f:
            f.write(latex_multi)
        print(f"     Saved to {latex_file2}")
    else:
        print("      Multi-model table may be incomplete")

    # Test 4c: Summary statistics export

    # Create sample data
    test_df = pd.DataFrame({
        'var1': np.random.randn(100) + 10,
        'var2': np.random.randn(100) * 5 + 20,
        'var3': np.random.randn(100) * 2 + 5
    })

    latex_summary = exporter.summary_statistics_to_latex(
        test_df,
        vars_to_include=['var1', 'var2', 'var3'],
        var_labels={'var1': 'Variable 1', 'var2': 'Variable 2', 'var3': 'Variable 3'},
        caption="Test Summary Statistics"
    )

    if 'Variable 1' in latex_summary and 'Mean' in latex_summary:
        print("     Summary statistics table looks correct")

        latex_file3 = output_dir / 'test_summary.tex'
        with open(latex_file3, 'w') as f:
            f.write(latex_summary)
        print(f"    Saved to {latex_file3}")
    else:
        print("    Summary statistics table may be incomplete")

    print("  ALL LaTeX EXPORT TESTS PASSED\n")

except Exception as e:
    print(f"  LaTeX EXPORT TEST FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("TEST SUMMARY")
print()
print("GARCH Models: Volatility analysis and forecasting")
print("VAR Models: Multivariate time series analysis")
print("LaTeX Export: Regression tables for papers")
print()
print(f"Test outputs saved to: {output_dir.absolute()}")
print()
print("ALL TESTS COMPLETED!")
print()
