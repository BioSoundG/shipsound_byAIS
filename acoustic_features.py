import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.stats import kurtosis, t as t_dist
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


def perform_regression_analysis(speed, size, sel, feature_name):
    """
    Perform multiple regression analysis to quantify effects of speed and size on SEL.
    
    Model: SEL = β0 + β1*speed + β2*size + β3*(speed*size) + ε
    
    Args:
        speed (np.ndarray): Vessel speed (knots)
        size (np.ndarray): Vessel size (m²)
        sel (np.ndarray): SEL values (dB)
        feature_name (str): Name of the acoustic feature
    
    Returns:
        dict: Analysis results including coefficients, p-values, R², residuals
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(speed) | np.isnan(size) | np.isnan(sel))
    speed_clean = speed[valid_mask]
    size_clean = size[valid_mask]
    sel_clean = sel[valid_mask]
    
    if len(speed_clean) < 4:  # Need at least 4 points for 3 predictors
        return None
    
    # Standardize predictors for better numerical stability
    scaler_speed = StandardScaler()
    scaler_size = StandardScaler()
    
    speed_std = scaler_speed.fit_transform(speed_clean.reshape(-1, 1)).flatten()
    size_std = scaler_size.fit_transform(size_clean.reshape(-1, 1)).flatten()
    
    # Create design matrix: [speed, size, speed*size]
    X = np.column_stack([
        speed_std,
        size_std,
        speed_std * size_std  # Interaction term
    ])
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Fit regression model
    model = LinearRegression(fit_intercept=False)  # We already added intercept
    model.fit(X_with_intercept, sel_clean)
    
    # Predictions and residuals
    y_pred = model.predict(X_with_intercept)
    residuals = sel_clean - y_pred
    
    # Calculate statistics
    n = len(sel_clean)
    k = X_with_intercept.shape[1] - 1  # Number of predictors (excluding intercept)
    df_residual = n - k - 1
    
    # Sum of squares
    ss_total = np.sum((sel_clean - np.mean(sel_clean)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    ss_regression = ss_total - ss_residual
    
    # R-squared and adjusted R-squared
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    adj_r_squared = 1 - (ss_residual / df_residual) / (ss_total / (n - 1)) if n > 1 else 0
    
    # Mean squared error
    mse = ss_residual / df_residual if df_residual > 0 else np.inf
    rmse = np.sqrt(mse)
    
    # Standard errors of coefficients
    try:
        # Covariance matrix of coefficients
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_coef = np.sqrt(np.diag(cov_matrix))
        
        # t-statistics and p-values
        t_stats = model.coef_ / se_coef
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df_residual))
    except:
        se_coef = np.full(len(model.coef_), np.nan)
        t_stats = np.full(len(model.coef_), np.nan)
        p_values = np.full(len(model.coef_), np.nan)
    
    # De-standardize coefficients for interpretation
    # Original scale: SEL = β0 + β1_orig*speed + β2_orig*size + β3_orig*(speed*size)
    speed_mean = scaler_speed.mean_[0]
    speed_std_val = scaler_speed.scale_[0]
    size_mean = scaler_size.mean_[0]
    size_std_val = scaler_size.scale_[0]
    
    beta_speed_orig = model.coef_[1] / speed_std_val
    beta_size_orig = model.coef_[2] / size_std_val
    beta_interaction_orig = model.coef_[3] / (speed_std_val * size_std_val)
    
    # AIC and BIC
    aic = n * np.log(ss_residual / n) + 2 * (k + 1)
    bic = n * np.log(ss_residual / n) + np.log(n) * (k + 1)
    
    return {
        'feature_name': feature_name,
        'n_samples': n,
        'coefficients': {
            'intercept': model.coef_[0],
            'speed_std': model.coef_[1],
            'size_std': model.coef_[2],
            'interaction_std': model.coef_[3],
            'speed_orig': beta_speed_orig,
            'size_orig': beta_size_orig,
            'interaction_orig': beta_interaction_orig
        },
        'std_errors': {
            'intercept': se_coef[0],
            'speed': se_coef[1],
            'size': se_coef[2],
            'interaction': se_coef[3]
        },
        't_statistics': {
            'intercept': t_stats[0],
            'speed': t_stats[1],
            'size': t_stats[2],
            'interaction': t_stats[3]
        },
        'p_values': {
            'intercept': p_values[0],
            'speed': p_values[1],
            'size': p_values[2],
            'interaction': p_values[3]
        },
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
        'fitted_values': y_pred,
        'speed_range': (speed_clean.min(), speed_clean.max()),
        'size_range': (size_clean.min(), size_clean.max()),
        'sel_range': (sel_clean.min(), sel_clean.max())
    }


def plot_residuals(result, output_dir, feature_key):
    """
    Plot residual diagnostics for regression analysis.
    
    Args:
        result (dict): Regression analysis results
        output_dir (str): Directory to save plots
        feature_key (str): Feature identifier for filename
    """
    residuals = result['residuals']
    fitted = result['fitted_values']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Residual Diagnostics: {result["feature_name"]}', fontsize=14, fontweight='bold')
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values (dB)', fontsize=11)
    axes[0, 0].set_ylabel('Residuals (dB)', fontsize=11)
    axes[0, 0].set_title('Residuals vs Fitted', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add LOESS smooth to residuals
    try:
        x_smooth, y_smooth, _, _ = loess_smooth(fitted, residuals, frac=0.5, degree=1)
        axes[0, 0].plot(x_smooth, y_smooth, 'b-', linewidth=2, label='LOESS smooth')
        axes[0, 0].legend()
    except:
        pass
    
    # 2. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale-Location Plot (sqrt of standardized residuals)
    standardized_residuals = residuals / np.std(residuals)
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
    axes[1, 0].scatter(fitted, sqrt_abs_std_residuals, alpha=0.6, s=50, 
                       edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Fitted Values (dB)', fontsize=11)
    axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
    axes[1, 0].set_title('Scale-Location Plot', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    try:
        x_smooth, y_smooth, _, _ = loess_smooth(fitted, sqrt_abs_std_residuals, frac=0.5, degree=1)
        axes[1, 0].plot(x_smooth, y_smooth, 'r-', linewidth=2)
    except:
        pass
    
    # 4. Residuals Histogram
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals (dB)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Residual Distribution', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add normal curve overlay
    mu, sigma = 0, np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    # Scale to match histogram
    bin_width = (residuals.max() - residuals.min()) / 20
    scale_factor = len(residuals) * bin_width
    axes[1, 1].plot(x, stats.norm.pdf(x, mu, sigma) * scale_factor, 
                    'r-', linewidth=2, label='Normal distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = f"residuals_{feature_key}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()


def generate_regression_report(results, output_path):
    """
    Generate a comprehensive text report for regression analyses.
    
    Args:
        results (dict): Dictionary of regression results (key: feature_key, value: result dict)
        output_path (str): Path to save the report file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("統計解析レポート: SELに対する速度とサイズの影響\n")
        f.write("Multiple Regression Analysis: Effects of Speed and Size on SEL\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("回帰モデル / Regression Model:\n")
        f.write("  SEL = β0 + β1 × Speed + β2 × Size + β3 × (Speed × Size) + ε\n\n")
        f.write("  β0: 切片 (Intercept)\n")
        f.write("  β1: 速度の効果 (Effect of speed, dB/knot)\n")
        f.write("  β2: サイズの効果 (Effect of size, dB/1000m²)\n")
        f.write("  β3: 交互作用 (Interaction term)\n")
        f.write("  ε: 誤差項 (Error term)\n\n")
        
        f.write("有意水準 / Significance levels:\n")
        f.write("  ***  p < 0.001  (highly significant)\n")
        f.write("  **   p < 0.01   (significant)\n")
        f.write("  *    p < 0.05   (marginally significant)\n")
        f.write("  n.s. p ≥ 0.05   (not significant)\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("【サマリー表 / Summary Table】\n\n")
        f.write(f"{'Feature':<20} {'N':<6} {'R²':<8} {'Adj R²':<8} {'RMSE':<8} "
                f"{'Speed':<12} {'Size':<12} {'Interaction':<12}\n")
        f.write("-" * 116 + "\n")
        
        for feature_key, result in results.items():
            feature_name = result['feature_name']
            n = result['n_samples']
            r2 = result['r_squared']
            adj_r2 = result['adj_r_squared']
            rmse = result['rmse']
            
            # Format p-values with significance stars
            def format_pval(p):
                if p < 0.001:
                    return "***"
                elif p < 0.01:
                    return "**"
                elif p < 0.05:
                    return "*"
                else:
                    return "n.s."
            
            speed_sig = format_pval(result['p_values']['speed'])
            size_sig = format_pval(result['p_values']['size'])
            inter_sig = format_pval(result['p_values']['interaction'])
            
            f.write(f"{feature_name:<20} {n:<6} {r2:<8.3f} {adj_r2:<8.3f} {rmse:<8.2f} "
                    f"{speed_sig:<12} {size_sig:<12} {inter_sig:<12}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed results for each feature
        for feature_key, result in results.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"【{result['feature_name']}】\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"サンプル数 / Sample size: {result['n_samples']}\n\n")
            
            f.write("データ範囲 / Data ranges:\n")
            f.write(f"  速度 (Speed):     {result['speed_range'][0]:.1f} - {result['speed_range'][1]:.1f} knots\n")
            f.write(f"  サイズ (Size):    {result['size_range'][0]:.0f} - {result['size_range'][1]:.0f} m²\n")
            f.write(f"  SEL:              {result['sel_range'][0]:.1f} - {result['sel_range'][1]:.1f} dB\n\n")
            
            f.write("回帰係数 / Regression Coefficients (Original Scale):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Parameter':<20} {'Estimate':<15} {'Std Error':<15} {'t-value':<12} {'p-value':<12} {'Sig.':<6}\n")
            f.write("-" * 80 + "\n")
            
            coef = result['coefficients']
            se = result['std_errors']
            t_stats = result['t_statistics']
            p_vals = result['p_values']
            
            def format_sig(p):
                if p < 0.001:
                    return "***"
                elif p < 0.01:
                    return "**"
                elif p < 0.05:
                    return "*"
                else:
                    return "n.s."
            
            # Intercept
            f.write(f"{'Intercept':<20} {coef['intercept']:<15.3f} {se['intercept']:<15.3f} "
                    f"{t_stats['intercept']:<12.3f} {p_vals['intercept']:<12.4f} "
                    f"{format_sig(p_vals['intercept']):<6}\n")
            
            # Speed (original scale)
            f.write(f"{'Speed (dB/knot)':<20} {coef['speed_orig']:<15.4f} {se['speed']:<15.3f} "
                    f"{t_stats['speed']:<12.3f} {p_vals['speed']:<12.4f} "
                    f"{format_sig(p_vals['speed']):<6}\n")
            
            # Size (original scale, per 1000 m²)
            size_per_1000 = coef['size_orig'] * 1000
            f.write(f"{'Size (dB/1000m²)':<20} {size_per_1000:<15.4f} {se['size']:<15.3f} "
                    f"{t_stats['size']:<12.3f} {p_vals['size']:<12.4f} "
                    f"{format_sig(p_vals['size']):<6}\n")
            
            # Interaction
            inter_per_1000 = coef['interaction_orig'] * 1000
            f.write(f"{'Interaction':<20} {inter_per_1000:<15.6f} {se['interaction']:<15.3f} "
                    f"{t_stats['interaction']:<12.3f} {p_vals['interaction']:<12.4f} "
                    f"{format_sig(p_vals['interaction']):<6}\n")
            
            f.write("\n")
            
            f.write("モデル適合度 / Model Fit:\n")
            f.write(f"  R² (決定係数):             {result['r_squared']:.4f}\n")
            f.write(f"  Adjusted R²:               {result['adj_r_squared']:.4f}\n")
            f.write(f"  RMSE (残差標準誤差):       {result['rmse']:.3f} dB\n")
            f.write(f"  AIC (赤池情報量基準):      {result['aic']:.2f}\n")
            f.write(f"  BIC (ベイズ情報量基準):    {result['bic']:.2f}\n\n")
            
            f.write("解釈 / Interpretation:\n")
            
            # Speed effect
            speed_coef = coef['speed_orig']
            speed_p = p_vals['speed']
            if speed_p < 0.05:
                if speed_coef > 0:
                    f.write(f"  • 速度が1ノット増加すると、SELは約{abs(speed_coef):.3f} dB 増加 (有意)\n")
                    f.write(f"    Speed: +1 knot → SEL: +{abs(speed_coef):.3f} dB (significant)\n")
                else:
                    f.write(f"  • 速度が1ノット増加すると、SELは約{abs(speed_coef):.3f} dB 減少 (有意)\n")
                    f.write(f"    Speed: +1 knot → SEL: -{abs(speed_coef):.3f} dB (significant)\n")
            else:
                f.write(f"  • 速度の効果は統計的に有意ではない (p={speed_p:.3f})\n")
                f.write(f"    Speed effect is not statistically significant (p={speed_p:.3f})\n")
            
            # Size effect
            size_coef = coef['size_orig'] * 1000
            size_p = p_vals['size']
            if size_p < 0.05:
                if size_coef > 0:
                    f.write(f"  • サイズが1000 m²増加すると、SELは約{abs(size_coef):.3f} dB 増加 (有意)\n")
                    f.write(f"    Size: +1000 m² → SEL: +{abs(size_coef):.3f} dB (significant)\n")
                else:
                    f.write(f"  • サイズが1000 m²増加すると、SELは約{abs(size_coef):.3f} dB 減少 (有意)\n")
                    f.write(f"    Size: +1000 m² → SEL: -{abs(size_coef):.3f} dB (significant)\n")
            else:
                f.write(f"  • サイズの効果は統計的に有意ではない (p={size_p:.3f})\n")
                f.write(f"    Size effect is not statistically significant (p={size_p:.3f})\n")
            
            # Interaction effect
            inter_p = p_vals['interaction']
            if inter_p < 0.05:
                f.write(f"  • 速度とサイズの交互作用が有意 (p={inter_p:.4f})\n")
                f.write(f"    → 速度の効果はサイズによって変化する、またはその逆\n")
                f.write(f"    Significant interaction (p={inter_p:.4f})\n")
                f.write(f"    → Effect of speed depends on size, or vice versa\n")
            else:
                f.write(f"  • 速度とサイズの交互作用は有意ではない (p={inter_p:.3f})\n")
                f.write(f"    → 速度とサイズは独立に効果を持つ\n")
                f.write(f"    No significant interaction (p={inter_p:.3f})\n")
                f.write(f"    → Speed and size have independent effects\n")
            
            # Model quality
            r2 = result['r_squared']
            if r2 >= 0.7:
                f.write(f"  • モデルの説明力は高い (R²={r2:.3f})\n")
                f.write(f"    Model has high explanatory power (R²={r2:.3f})\n")
            elif r2 >= 0.4:
                f.write(f"  • モデルの説明力は中程度 (R²={r2:.3f})\n")
                f.write(f"    Model has moderate explanatory power (R²={r2:.3f})\n")
            else:
                f.write(f"  • モデルの説明力は低い (R²={r2:.3f})\n")
                f.write(f"    → 他の要因も重要である可能性\n")
                f.write(f"    Model has low explanatory power (R²={r2:.3f})\n")
                f.write(f"    → Other factors may be important\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("レポート終了 / End of Report\n")
        f.write("=" * 80 + "\n")


def loess_smooth(x, y, frac=0.3, degree=2):
    """
    LOESS (Locally Weighted Scatterplot Smoothing) regression.
    
    Args:
        x (np.ndarray): Independent variable
        y (np.ndarray): Dependent variable
        frac (float): Fraction of data used for local regression (0-1)
        degree (int): Degree of polynomial fit
    
    Returns:
        tuple: (x_smooth, y_smooth, y_lower, y_upper) for plotting
    """
    # Sort by x
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # Number of points for smoothing
    n = len(x_sorted)
    window_size = max(int(frac * n), 3)  # At least 3 points
    
    # Create smooth prediction points
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
    y_smooth = np.zeros_like(x_smooth)
    y_std = np.zeros_like(x_smooth)
    
    # Perform local weighted regression for each point
    for i, xi in enumerate(x_smooth):
        # Find nearest neighbors
        distances = np.abs(x_sorted - xi)
        nearest_idx = np.argsort(distances)[:window_size]
        
        # Local data
        x_local = x_sorted[nearest_idx]
        y_local = y_sorted[nearest_idx]
        
        # Weights (tricube kernel)
        max_dist = np.max(np.abs(x_local - xi))
        if max_dist > 0:
            weights = (1 - (np.abs(x_local - xi) / max_dist) ** 3) ** 3
        else:
            weights = np.ones_like(x_local)
        
        # Weighted polynomial regression
        try:
            poly_model = make_pipeline(
                PolynomialFeatures(degree=degree),
                LinearRegression()
            )
            
            # Weighted fit
            sample_weight = weights
            poly_model.fit(x_local.reshape(-1, 1), y_local, 
                          linearregression__sample_weight=sample_weight)
            
            # Predict
            y_smooth[i] = poly_model.predict([[xi]])[0]
            
            # Estimate local variance for confidence interval
            y_pred_local = poly_model.predict(x_local.reshape(-1, 1))
            residuals = y_local - y_pred_local
            y_std[i] = np.sqrt(np.average(residuals**2, weights=weights))
        except:
            # Fallback to simple weighted mean
            y_smooth[i] = np.average(y_local, weights=weights)
            y_std[i] = np.sqrt(np.average((y_local - y_smooth[i])**2, weights=weights))
    
    # 95% confidence interval
    y_lower = y_smooth - 1.96 * y_std
    y_upper = y_smooth + 1.96 * y_std
    
    return x_smooth, y_smooth, y_lower, y_upper


def calculate_third_octave_sel(audio_data, sr, center_freqs=[63, 125, 200, 500, 1000, 2000]):
    """
    Calculate Sound Exposure Level (SEL) for 1/3 octave bands.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        center_freqs (list): Center frequencies for 1/3 octave bands (Hz)
    
    Returns:
        dict: SEL values for each center frequency (dB re 1 μPa²·s)
    """
    sel_values = {}
    
    for fc in center_freqs:
        # 1/3オクターブバンドの上限・下限周波数
        # f_lower = fc / 2^(1/6), f_upper = fc * 2^(1/6)
        f_lower = fc / (2 ** (1/6))
        f_upper = fc * (2 ** (1/6))
        
        # バンドパスフィルタの設計
        sos = signal.butter(4, [f_lower, f_upper], btype='band', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio_data)
        
        # SEL計算: SEL = 10 * log10(sum(p^2) / p_ref^2)
        # p_ref = 1 μPa (水中音響の基準)
        p_ref = 1.0  # 正規化された信号を仮定
        energy = np.sum(filtered ** 2)
        
        if energy > 0:
            sel_db = 10 * np.log10(energy / (p_ref ** 2))
        else:
            sel_db = -np.inf
        
        sel_values[f"{fc}Hz"] = sel_db
    
    return sel_values


def calculate_third_octave_sel_with_background(audio_data, sr, background_data=None, 
                                                 center_freqs=[63, 125, 200, 500, 1000, 2000]):
    """
    Calculate SEL with background subtraction.
    
    Args:
        audio_data (np.ndarray): Audio signal with vessel
        sr (int): Sample rate
        background_data (np.ndarray): Background noise signal (optional)
        center_freqs (list): Center frequencies for 1/3 octave bands (Hz)
    
    Returns:
        tuple: (sel_values, sel_background_subtracted)
    """
    sel_values = calculate_third_octave_sel(audio_data, sr, center_freqs)
    
    if background_data is not None and len(background_data) > 0:
        sel_bg = calculate_third_octave_sel(background_data, sr, center_freqs)
        sel_subtracted = {}
        
        for key in sel_values.keys():
            # 背景差分: SEL_vessel - SEL_background (dB)
            sel_subtracted[f"{key}_bg_sub"] = sel_values[key] - sel_bg[key]
        
        return sel_values, sel_subtracted
    else:
        return sel_values, {}


def detect_tonals_and_calculate_level(audio_data, sr, n_harmonics=5, freq_range=(50, 5000)):
    """
    Detect tonal components (1-5th harmonics) and calculate total tonal level.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        n_harmonics (int): Number of harmonics to detect (default: 5)
        freq_range (tuple): Frequency range to search (Hz)
    
    Returns:
        dict: {
            'tonal_level_db': Total level of detected tonals (dB),
            'num_tonals': Number of detected tonal components,
            'fundamental_freq': Estimated fundamental frequency (Hz)
        }
    """
    # FFTでスペクトル計算
    nperseg = min(4096, len(audio_data))
    f, psd = signal.welch(audio_data, sr, nperseg=nperseg)
    
    # 指定周波数範囲内のインデックス
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_range = f[freq_mask]
    psd_range = psd[freq_mask]
    
    if len(psd_range) == 0:
        return {'tonal_level_db': -np.inf, 'num_tonals': 0, 'fundamental_freq': 0}
    
    # ピーク検出（トーナル成分）
    # 周囲より20dB以上高いピークを検出
    psd_db = 10 * np.log10(psd_range + 1e-12)
    peaks, properties = signal.find_peaks(psd_db, prominence=10, distance=10)
    
    if len(peaks) == 0:
        return {'tonal_level_db': -np.inf, 'num_tonals': 0, 'fundamental_freq': 0}
    
    # 最も強いピークを基音と仮定
    fundamental_idx = peaks[np.argmax(psd_range[peaks])]
    fundamental_freq = f_range[fundamental_idx]
    
    # 高調波の検出（基音の整数倍）
    detected_tonals = [fundamental_freq]
    tonal_powers = [psd_range[fundamental_idx]]
    
    for n in range(2, n_harmonics + 1):
        harmonic_freq = fundamental_freq * n
        if harmonic_freq > freq_range[1]:
            break
        
        # 高調波周波数の近傍でピークを探す
        tolerance = fundamental_freq * 0.1  # ±10%の許容範囲
        harmonic_mask = np.abs(f_range - harmonic_freq) < tolerance
        
        if np.any(harmonic_mask):
            harmonic_psd = psd_range[harmonic_mask]
            if len(harmonic_psd) > 0:
                max_harmonic_power = np.max(harmonic_psd)
                # 閾値以上のパワーがあれば高調波として認識
                if max_harmonic_power > np.median(psd_range) * 2:
                    detected_tonals.append(harmonic_freq)
                    tonal_powers.append(max_harmonic_power)
    
    # トーナル合計レベル（線形パワーの合計をdBに変換）
    total_tonal_power = np.sum(tonal_powers)
    tonal_level_db = 10 * np.log10(total_tonal_power + 1e-12)
    
    return {
        'tonal_level_db': tonal_level_db,
        'num_tonals': len(detected_tonals),
        'fundamental_freq': fundamental_freq
    }


def calculate_spectral_centroid(audio_data, sr, freq_range=(50, 5000)):
    """
    Calculate spectral centroid within specified frequency range.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        freq_range (tuple): Frequency range (Hz)
    
    Returns:
        float: Spectral centroid (Hz)
    """
    # FFTでスペクトル計算
    nperseg = min(4096, len(audio_data))
    f, psd = signal.welch(audio_data, sr, nperseg=nperseg)
    
    # 指定周波数範囲内のインデックス
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_range = f[freq_mask]
    psd_range = psd[freq_mask]
    
    if len(psd_range) == 0 or np.sum(psd_range) == 0:
        return 0.0
    
    # スペクトル重心: Σ(f * P(f)) / Σ(P(f))
    centroid = np.sum(f_range * psd_range) / np.sum(psd_range)
    
    return centroid


def calculate_kurtosis_and_crest_factor(audio_data):
    """
    Calculate kurtosis and crest factor of audio signal.
    
    Args:
        audio_data (np.ndarray): Audio signal
    
    Returns:
        dict: {
            'kurtosis': Kurtosis value,
            'crest_factor_db': Crest factor in dB
        }
    """
    # クルトシス（尖度）
    kurt = kurtosis(audio_data, fisher=True)  # Excess kurtosis (正規分布=0)
    
    # クレストファクタ: peak / RMS
    peak = np.max(np.abs(audio_data))
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    if rms > 0:
        crest_factor = peak / rms
        crest_factor_db = 20 * np.log10(crest_factor)
    else:
        crest_factor_db = np.inf
    
    return {
        'kurtosis': kurt,
        'crest_factor_db': crest_factor_db
    }


def extract_audio_segment(wav_path, start_time, duration, sr_target=None):
    """
    Extract audio segment from WAV file.
    
    Args:
        wav_path (str): Path to WAV file
        start_time (float): Start time in seconds from beginning of file
        duration (float): Duration in seconds
        sr_target (int): Target sample rate (if None, use original)
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        start_frame = int(start_time * sr)
        num_frames = int(duration * sr)
        
        # ファイル範囲内に収める
        start_frame = max(0, start_frame)
        start_frame = min(start_frame, len(f) - 1)
        num_frames = min(num_frames, len(f) - start_frame)
        
        if num_frames <= 0:
            return np.array([]), sr
        
        f.seek(start_frame)
        audio_data = f.read(num_frames)
        
        # ステレオの場合はモノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
    
    # リサンプリング（必要な場合）
    if sr_target is not None and sr != sr_target:
        num_samples = int(len(audio_data) * sr_target / sr)
        audio_data = signal.resample(audio_data, num_samples)
        sr = sr_target
    
    return audio_data, sr


def calculate_all_features(audio_data, sr, background_data=None):
    """
    Calculate all acoustic features.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        background_data (np.ndarray): Background noise (optional)
    
    Returns:
        dict: All acoustic features
    """
    features = {}
    
    # 1. 1/3オクターブバンドSEL
    sel_values, sel_bg_sub = calculate_third_octave_sel_with_background(
        audio_data, sr, background_data
    )
    features.update(sel_values)
    features.update(sel_bg_sub)
    
    # 2. トーナル合計レベルとトーナル数
    tonal_info = detect_tonals_and_calculate_level(audio_data, sr)
    features['tonal_level_db'] = tonal_info['tonal_level_db']
    features['num_tonals'] = tonal_info['num_tonals']
    features['fundamental_freq'] = tonal_info['fundamental_freq']
    
    # 3. スペクトル重心
    features['spectral_centroid_hz'] = calculate_spectral_centroid(audio_data, sr)
    
    # 4. クルトシスとクレストファクタ
    kurt_crest = calculate_kurtosis_and_crest_factor(audio_data)
    features['kurtosis'] = kurt_crest['kurtosis']
    features['crest_factor_db'] = kurt_crest['crest_factor_db']
    
    return features


def analyze_vessel_acoustic_features(wav_list, wav_index, distances_df, comp_df, 
                                       analysis_window=10, output_dir=None):
    """
    Analyze acoustic features for all vessels at their minimum distance time.
    
    Args:
        wav_list (list): List of WAV file paths
        wav_index: WAV index object
        distances_df (pd.DataFrame): Distance information with min_distance_time
        comp_df (pd.DataFrame): Complemented AIS data with vessel info
        analysis_window (float): Time window around min distance (seconds)
        output_dir (str): Output directory for results
    
    Returns:
        pd.DataFrame: Acoustic features for each vessel
    """
    results = []
    
    print(f"\n音響特徴量分析開始: {len(distances_df)} 隻")
    
    for idx, row in distances_df.iterrows():
        mmsi = row['mmsi']
        min_dist_time = pd.to_datetime(row['min_distance_time'])
        min_distance = row['min_distance [m]']
        
        # 船舶情報を取得
        vessel_info = comp_df[comp_df['mmsi'] == mmsi].iloc[0]
        vessel_length = vessel_info.get('length', np.nan)
        vessel_width = vessel_info.get('width', np.nan)
        vessel_speed = vessel_info.get('speed', np.nan)
        vessel_size = vessel_length * vessel_width if not np.isnan(vessel_length) and not np.isnan(vessel_width) else np.nan
        
        # 最短距離時刻に対応するWAVファイルを探す
        wav_idx = wav_index.find_wav_index(min_dist_time, 0)
        
        if wav_idx is None or wav_idx >= len(wav_list):
            print(f"  船舶 {mmsi}: WAVファイルが見つかりません")
            continue
        
        wav_path = wav_list[wav_idx]
        wav_start_time = wav_index.get_start_time(wav_idx)
        
        # WAVファイル内での相対時刻
        time_offset = (min_dist_time - wav_start_time).total_seconds()
        
        # 音声セグメントを抽出（最短距離時刻 ± analysis_window秒）
        start_time = max(0, time_offset - analysis_window)
        duration = analysis_window * 2
        
        try:
            audio_data, sr = extract_audio_segment(wav_path, start_time, duration)
            
            if len(audio_data) == 0:
                print(f"  船舶 {mmsi}: 音声データが空です")
                continue
            
            # 音響特徴量を計算
            features = calculate_all_features(audio_data, sr)
            
            # 結果を記録
            result = {
                'mmsi': mmsi,
                'min_distance_m': min_distance,
                'min_distance_time': min_dist_time,
                'vessel_length_m': vessel_length,
                'vessel_width_m': vessel_width,
                'vessel_size_m2': vessel_size,
                'vessel_speed_knots': vessel_speed,
            }
            result.update(features)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"  処理済み: {idx + 1}/{len(distances_df)} 隻")
        
        except Exception as e:
            print(f"  船舶 {mmsi}: エラー - {e}")
            continue
    
    print(f"音響特徴量分析完了: {len(results)} 隻")
    
    # DataFrameに変換
    results_df = pd.DataFrame(results)
    
    # CSVとして保存
    if output_dir and len(results_df) > 0:
        csv_path = os.path.join(output_dir, "acoustic_features.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"音響特徴量をCSVに保存: {csv_path}")
    
    return results_df


def plot_feature_vs_vessel_params(features_df, output_dir):
    """
    Plot acoustic features vs vessel parameters (speed and size).
    
    Args:
        features_df (pd.DataFrame): DataFrame with acoustic features and vessel params
        output_dir (str): Output directory for plots
    """
    if len(features_df) == 0:
        print("プロット用のデータがありません")
        return
    
    # 有効なデータのみ抽出
    valid_speed = features_df.dropna(subset=['vessel_speed_knots'])
    valid_size = features_df.dropna(subset=['vessel_size_m2'])
    
    # 特徴量リスト
    feature_keys = [
        ('63Hz', '63 Hz SEL (dB)'),
        ('125Hz', '125 Hz SEL (dB)'),
        ('200Hz', '200 Hz SEL (dB)'),
        ('500Hz', '500 Hz SEL (dB)'),
        ('1000Hz', '1 kHz SEL (dB)'),
        ('2000Hz', '2 kHz SEL (dB)'),
        ('tonal_level_db', 'Tonal Level (dB)'),
        ('num_tonals', 'Number of Tonals'),
        ('spectral_centroid_hz', 'Spectral Centroid (Hz)'),
        ('kurtosis', 'Kurtosis'),
        ('crest_factor_db', 'Crest Factor (dB)'),
    ]
    
    # 背景差分版のSELも追加
    bg_sub_keys = [
        ('63Hz_bg_sub', '63 Hz SEL (bg sub, dB)'),
        ('125Hz_bg_sub', '125 Hz SEL (bg sub, dB)'),
        ('200Hz_bg_sub', '200 Hz SEL (bg sub, dB)'),
        ('500Hz_bg_sub', '500 Hz SEL (bg sub, dB)'),
        ('1000Hz_bg_sub', '1 kHz SEL (bg sub, dB)'),
        ('2000Hz_bg_sub', '2 kHz SEL (bg sub, dB)'),
    ]
    
    # 背景差分データがあれば追加
    if '63Hz_bg_sub' in features_df.columns:
        feature_keys.extend(bg_sub_keys)
    
    print(f"\n2次元グラフ生成中...")
    
    # 1. 音響特徴量 vs 船舶速度
    if len(valid_speed) > 0:
        # SEL周波数帯のキー（特別な処理が必要）
        sel_keys = ['63Hz', '125Hz', '200Hz', '500Hz', '1000Hz', '2000Hz',
                    '63Hz_bg_sub', '125Hz_bg_sub', '200Hz_bg_sub', 
                    '500Hz_bg_sub', '1000Hz_bg_sub', '2000Hz_bg_sub']
        
        for feature_key, feature_label in feature_keys:
            if feature_key not in features_df.columns:
                continue
            
            valid_data = valid_speed.dropna(subset=[feature_key, 'vessel_size_m2'])
            if len(valid_data) < 2:
                continue
            
            plt.figure(figsize=(11, 6))
            
            # SEL系の特徴量の場合: 船舶サイズを色に反映、LOESS回帰を追加
            if feature_key in sel_keys:
                # 船舶サイズを色に変換（小型=青、大型=赤）
                size_colors = valid_data['vessel_size_m2'].values
                
                # LOESS回帰（±95%CI）を追加（散布図の前に描画）
                if len(valid_data) >= 5:  # 最低5点必要
                    try:
                        x_smooth, y_smooth, y_lower, y_upper = loess_smooth(
                            valid_data['vessel_speed_knots'].values,
                            valid_data[feature_key].values,
                            frac=0.4,  # 40%のデータで局所回帰
                            degree=2
                        )
                        
                        # 95% 信頼区間（最背面、灰色、高透明度）
                        plt.fill_between(x_smooth, y_lower, y_upper, 
                                        alpha=0.15, color='gray', label='95% CI', zorder=1)
                        
                        # LOESS曲線（中間層）
                        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='LOESS fit', zorder=2)
                        
                    except Exception as e:
                        print(f"    Warning: LOESS fitting failed for {feature_key}: {e}")
                
                # 散布図（最前面、全て同じサイズ）
                scatter = plt.scatter(valid_data['vessel_speed_knots'], valid_data[feature_key], 
                           c=size_colors, cmap='coolwarm', alpha=0.7, s=80, 
                           edgecolors='black', linewidth=0.5,
                           label='Data points', vmin=size_colors.min(), vmax=size_colors.max(),
                           zorder=3)
                
                # カラーバーを追加
                cbar = plt.colorbar(scatter, ax=plt.gca())
                cbar.set_label('Vessel Size (m²)', fontsize=11)
                
                # 凡例を追加
                plt.legend(loc='best', fontsize=10)
            else:
                # SEL以外の特徴量は従来通り
                plt.scatter(valid_data['vessel_speed_knots'], valid_data[feature_key], 
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            plt.xlabel('Vessel Speed (knots)', fontsize=12)
            plt.ylabel(feature_label, fontsize=12)
            plt.title(f'{feature_label} vs Vessel Speed', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"feature_vs_speed_{feature_key}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150)
            plt.close()
        
        print(f"  速度との関係グラフ: {len([k for k, _ in feature_keys if k in features_df.columns])} 個")
    
    # 2. 音響特徴量 vs 船舶サイズ
    if len(valid_size) > 0:
        # SEL周波数帯のキー（特別な処理が必要）
        sel_keys = ['63Hz', '125Hz', '200Hz', '500Hz', '1000Hz', '2000Hz',
                    '63Hz_bg_sub', '125Hz_bg_sub', '200Hz_bg_sub', 
                    '500Hz_bg_sub', '1000Hz_bg_sub', '2000Hz_bg_sub']
        
        for feature_key, feature_label in feature_keys:
            if feature_key not in features_df.columns:
                continue
            
            valid_data = valid_size.dropna(subset=[feature_key, 'min_distance_m', 'vessel_speed_knots'])
            if len(valid_data) < 2:
                continue
            
            plt.figure(figsize=(11, 6))
            
            # SEL系の特徴量の場合: 速度を色に反映、サイズは統一
            if feature_key in sel_keys:
                # 速度を色に変換（低速=青、高速=赤）
                speed_colors = valid_data['vessel_speed_knots'].values
                
                # LOESS回帰（±95%CI）を追加（散布図の前に描画）
                if len(valid_data) >= 5:  # 最低5点必要
                    try:
                        x_smooth, y_smooth, y_lower, y_upper = loess_smooth(
                            valid_data['vessel_size_m2'].values,
                            valid_data[feature_key].values,
                            frac=0.4,  # 40%のデータで局所回帰
                            degree=2
                        )
                        
                        # 95% 信頼区間（最背面、灰色、高透明度）
                        plt.fill_between(x_smooth, y_lower, y_upper, 
                                        alpha=0.15, color='gray', label='95% CI', zorder=1)
                        
                        # LOESS曲線（中間層）
                        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='LOESS fit', zorder=2)
                        
                    except Exception as e:
                        print(f"    Warning: LOESS fitting failed for {feature_key}: {e}")
                
                # 散布図（最前面、全て同じサイズ）
                scatter = plt.scatter(valid_data['vessel_size_m2'], valid_data[feature_key], 
                           c=speed_colors, cmap='coolwarm', alpha=0.7, s=80, 
                           edgecolors='black', linewidth=0.5,
                           label='Data points', vmin=speed_colors.min(), vmax=speed_colors.max(),
                           zorder=3)
                
                # カラーバーを追加
                cbar = plt.colorbar(scatter, ax=plt.gca())
                cbar.set_label('Vessel Speed (knots)', fontsize=11)
                
                # 凡例を追加
                plt.legend(loc='best', fontsize=10)
            else:
                # SEL以外の特徴量は従来通り
                plt.scatter(valid_data['vessel_size_m2'], valid_data[feature_key], 
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            plt.xlabel('Vessel Size (m²)', fontsize=12)
            plt.ylabel(feature_label, fontsize=12)
            plt.title(f'{feature_label} vs Vessel Size', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"feature_vs_size_{feature_key}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150)
            plt.close()
        
        print(f"  サイズとの関係グラフ: {len([k for k, _ in feature_keys if k in features_df.columns])} 個")
    
    # 3. 船舶サイズ vs 船舶速度
    valid_both = features_df.dropna(subset=['vessel_size_m2', 'vessel_speed_knots'])
    if len(valid_both) > 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_both['vessel_speed_knots'], valid_both['vessel_size_m2'], 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.xlabel('Vessel Speed (knots)', fontsize=12)
        plt.ylabel('Vessel Size (m²)', fontsize=12)
        plt.title('Vessel Size vs Vessel Speed', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = "vessel_size_vs_speed.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        print(f"  船舶サイズ vs 速度グラフ: 1 個")
    
    # 4. 統計解析レポートの生成
    print(f"\n統計解析レポート生成中...")
    regression_results = {}
    
    # SEL周波数帯の特徴量について重回帰分析を実施
    sel_keys = ['63Hz', '125Hz', '200Hz', '500Hz', '1000Hz', '2000Hz',
                '63Hz_bg_sub', '125Hz_bg_sub', '200Hz_bg_sub', 
                '500Hz_bg_sub', '1000Hz_bg_sub', '2000Hz_bg_sub']
    
    for feature_key, feature_label in feature_keys:
        if feature_key not in features_df.columns:
            continue
        if feature_key not in sel_keys:
            continue  # SEL特徴量のみ解析
        
        valid_data = features_df.dropna(subset=['vessel_speed_knots', 'vessel_size_m2', feature_key])
        if len(valid_data) < 4:
            continue
        
        speed = valid_data['vessel_speed_knots'].values
        size = valid_data['vessel_size_m2'].values
        sel = valid_data[feature_key].values
        
        result = perform_regression_analysis(speed, size, sel, feature_label)
        if result is not None:
            regression_results[feature_key] = result
            
            # 残差プロットの生成
            plot_residuals(result, output_dir, feature_key)
    
    # レポートファイルの生成
    if regression_results:
        report_path = os.path.join(output_dir, "regression_analysis_report.txt")
        generate_regression_report(regression_results, report_path)
        print(f"  統計解析レポート: 1 個")
        print(f"  残差プロット: {len(regression_results)} 個")
    
    print(f"グラフ保存完了: {output_dir}")

