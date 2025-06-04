"""
Temporal Fire Pattern Analysis

This module provides advanced temporal analysis of fire patterns including:
1. Fire regime change detection
2. Long-term trend analysis  
3. Fire cycle analysis
4. Seasonal pattern characterization
5. Fire clustering in time

Complements the main fire metrics calculator with specialized temporal analyses.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalFireAnalyzer:
    """
    Advanced temporal analysis of fire patterns for watershed fire regime characterization.
    
    Provides methods to analyze temporal patterns, trends, and changes in fire regimes
    that complement the basic fire metrics calculated in fire_metrics.py.
    """
    
    def __init__(self, min_years_for_trend: int = 10):
        """
        Initialize temporal fire analyzer.
        
        Args:
            min_years_for_trend: Minimum years of data required for trend analysis
        """
        self.min_years_for_trend = min_years_for_trend
        logger.info(f"Temporal Fire Analyzer initialized (min years for trends: {min_years_for_trend})")
    
    def analyze_fire_trends(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze long-term trends in fire activity for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data with dates
            
        Returns:
            pd.DataFrame: Trend analysis results per watershed
        """
        logger.info("Analyzing fire trends")
        
        trend_results = []
        
        for huc12_id in fire_watershed_df['huc12'].unique():
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            # Create annual fire counts
            annual_counts = watershed_fires.groupby('year').size()
            years = annual_counts.index
            counts = annual_counts.values
            
            result = {'huc12': huc12_id}
            
            if len(years) >= self.min_years_for_trend:
                # Linear trend analysis
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
                
                result.update({
                    'trend_slope': slope,  # Fires per year change
                    'trend_r_squared': r_value**2,
                    'trend_p_value': p_value,
                    'trend_significant': p_value < 0.05,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'years_of_data': len(years)
                })
                
                # Mann-Kendall test for non-parametric trend detection
                mk_result = self._mann_kendall_test(counts)
                result.update({
                    'mk_trend': mk_result['trend'],
                    'mk_p_value': mk_result['p'],
                    'mk_significant': mk_result['p'] < 0.05
                })
                
                # Detect change points
                change_points = self._detect_change_points(years, counts)
                result.update({
                    'n_change_points': len(change_points),
                    'change_point_years': change_points if change_points else []
                })
                
                # Calculate trend magnitude
                if len(years) > 1:
                    trend_magnitude = abs(slope) / np.mean(counts) if np.mean(counts) > 0 else 0
                    result['trend_magnitude'] = trend_magnitude
                else:
                    result['trend_magnitude'] = 0
                
            else:
                # Insufficient data for trend analysis
                result.update({
                    'trend_slope': np.nan,
                    'trend_r_squared': np.nan,
                    'trend_p_value': np.nan,
                    'trend_significant': False,
                    'trend_direction': 'insufficient_data',
                    'years_of_data': len(years),
                    'mk_trend': 'insufficient_data',
                    'mk_p_value': np.nan,
                    'mk_significant': False,
                    'n_change_points': 0,
                    'change_point_years': [],
                    'trend_magnitude': np.nan
                })
            
            trend_results.append(result)
        
        trend_df = pd.DataFrame(trend_results)
        
        logger.info(f"Trend analysis completed:")
        logger.info(f"  - Watersheds with sufficient data: {(trend_df['years_of_data'] >= self.min_years_for_trend).sum()}")
        logger.info(f"  - Significant increasing trends: {((trend_df['trend_direction'] == 'increasing') & trend_df['trend_significant']).sum()}")
        logger.info(f"  - Significant decreasing trends: {((trend_df['trend_direction'] == 'decreasing') & trend_df['trend_significant']).sum()}")
        
        return trend_df
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform Mann-Kendall test for trend detection.
        
        Args:
            data: Time series data
            
        Returns:
            Dict: Test results
        """
        n = len(data)
        
        if n < 3:
            return {'trend': 'insufficient_data', 'p': np.nan, 'tau': np.nan}
        
        # Calculate S statistic
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        # Determine trend
        if p_value < 0.05:
            if S > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no_trend'
        
        # Calculate Kendall's tau
        tau = S / (n * (n - 1) / 2)
        
        return {
            'trend': trend,
            'p': p_value,
            'tau': tau,
            'S': S,
            'Z': Z
        }
    
    def _detect_change_points(self, years: np.ndarray, counts: np.ndarray, 
                            min_segment_length: int = 5) -> List[int]:
        """
        Detect change points in fire activity time series.
        
        Args:
            years: Year values
            counts: Fire count values
            min_segment_length: Minimum length of segments
            
        Returns:
            List[int]: Years where change points occur
        """
        if len(years) < min_segment_length * 2:
            return []
        
        change_points = []
        
        # Simple change point detection using moving variance
        window_size = min(5, len(years) // 3)
        
        if window_size < 2:
            return []
        
        variances = []
        for i in range(window_size, len(counts) - window_size):
            left_var = np.var(counts[i-window_size:i])
            right_var = np.var(counts[i:i+window_size])
            total_var = np.var(counts[i-window_size:i+window_size])
            
            # Change point score
            score = total_var - (left_var + right_var) / 2
            variances.append((i, score))
        
        if variances:
            # Find peaks in change point scores
            scores = [v[1] for v in variances]
            indices = [v[0] for v in variances]
            
            # Find significant peaks
            if len(scores) > 3:
                threshold = np.mean(scores) + 1.5 * np.std(scores)
                peaks, _ = find_peaks(scores, height=threshold, distance=min_segment_length)
                
                change_points = [years[indices[p]] for p in peaks]
        
        return sorted(change_points)
    
    def analyze_fire_cycles(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze fire cycles and periodicity for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Fire cycle analysis results per watershed
        """
        logger.info("Analyzing fire cycles and periodicity")
        
        cycle_results = []
        
        for huc12_id in fire_watershed_df['huc12'].unique():
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            result = {'huc12': huc12_id}
            
            if len(watershed_fires) >= 3:
                # Create annual fire presence time series
                years = range(watershed_fires['year'].min(), watershed_fires['year'].max() + 1)
                fire_years = set(watershed_fires['year'])
                presence_series = [1 if year in fire_years else 0 for year in years]
                
                if len(presence_series) >= 10:  # Need sufficient data for cycle analysis
                    # Autocorrelation analysis
                    autocorr = self._calculate_autocorrelation(presence_series)
                    
                    # Find dominant periods
                    dominant_periods = self._find_dominant_periods(autocorr)
                    
                    result.update({
                        'has_cycles': len(dominant_periods) > 0,
                        'dominant_cycle_length': dominant_periods[0] if dominant_periods else np.nan,
                        'secondary_cycle_length': dominant_periods[1] if len(dominant_periods) > 1 else np.nan,
                        'cycle_strength': max(autocorr[1:]) if len(autocorr) > 1 else 0,
                        'n_potential_cycles': len(dominant_periods)
                    })
                    
                    # Fire clustering analysis
                    cluster_metrics = self._analyze_fire_clustering(years, fire_years)
                    result.update(cluster_metrics)
                    
                else:
                    result.update({
                        'has_cycles': False,
                        'dominant_cycle_length': np.nan,
                        'secondary_cycle_length': np.nan,
                        'cycle_strength': 0,
                        'n_potential_cycles': 0,
                        'fire_clustering_index': np.nan,
                        'max_cluster_length': np.nan,
                        'fire_burst_probability': np.nan
                    })
            else:
                result.update({
                    'has_cycles': False,
                    'dominant_cycle_length': np.nan,
                    'secondary_cycle_length': np.nan,
                    'cycle_strength': 0,
                    'n_potential_cycles': 0,
                    'fire_clustering_index': np.nan,
                    'max_cluster_length': np.nan,
                    'fire_burst_probability': np.nan
                })
            
            cycle_results.append(result)
        
        cycle_df = pd.DataFrame(cycle_results)
        
        logger.info(f"Fire cycle analysis completed:")
        logger.info(f"  - Watersheds with potential cycles: {cycle_df['has_cycles'].sum()}")
        if cycle_df['dominant_cycle_length'].notna().any():
            logger.info(f"  - Mean dominant cycle length: {cycle_df['dominant_cycle_length'].mean():.1f} years")
        
        return cycle_df
    
    def _calculate_autocorrelation(self, series: List[int], max_lag: Optional[int] = None) -> np.ndarray:
        """Calculate autocorrelation function for time series."""
        series = np.array(series)
        n = len(series)
        
        if max_lag is None:
            max_lag = min(n // 3, 20)  # Limit to reasonable lag
        
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr[:max_lag+1] / autocorr[0]  # Normalize
        
        return autocorr
    
    def _find_dominant_periods(self, autocorr: np.ndarray, min_period: int = 2, 
                             threshold: float = 0.3) -> List[int]:
        """Find dominant periods in autocorrelation function."""
        if len(autocorr) < min_period + 1:
            return []
        
        # Find peaks in autocorrelation (excluding lag 0)
        peaks, properties = find_peaks(autocorr[1:], height=threshold, distance=min_period)
        
        # Adjust for the offset (we excluded lag 0)
        peaks = peaks + 1
        
        # Sort by peak height (correlation strength)
        if len(peaks) > 0:
            peak_heights = autocorr[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]  # Descending order
            dominant_periods = peaks[sorted_indices].tolist()
        else:
            dominant_periods = []
        
        return dominant_periods
    
    def _analyze_fire_clustering(self, years: range, fire_years: set) -> Dict[str, float]:
        """Analyze temporal clustering of fire years."""
        fire_year_list = sorted(fire_years)
        
        if len(fire_year_list) < 2:
            return {
                'fire_clustering_index': np.nan,
                'max_cluster_length': np.nan,
                'fire_burst_probability': np.nan
            }
        
        # Calculate gaps between fire years
        gaps = [fire_year_list[i+1] - fire_year_list[i] for i in range(len(fire_year_list)-1)]
        
        # Clustering index: proportion of short gaps (≤2 years)
        short_gaps = sum(1 for gap in gaps if gap <= 2)
        clustering_index = short_gaps / len(gaps) if gaps else 0
        
        # Find maximum cluster length (consecutive or near-consecutive fire years)
        max_cluster = 1
        current_cluster = 1
        
        for gap in gaps:
            if gap <= 2:  # Consider gap of ≤2 years as same cluster
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 1
        
        # Fire burst probability (probability of fire given recent fire)
        fire_after_fire = 0
        total_opportunities = 0
        
        for year in years:
            if year > min(fire_years) and year < max(fire_years):
                # Check if there was a fire in previous 1-2 years
                recent_fire = any(fy in range(year-2, year) for fy in fire_years)
                if recent_fire:
                    total_opportunities += 1
                    if year in fire_years:
                        fire_after_fire += 1
        
        burst_probability = fire_after_fire / total_opportunities if total_opportunities > 0 else np.nan
        
        return {
            'fire_clustering_index': clustering_index,
            'max_cluster_length': max_cluster,
            'fire_burst_probability': burst_probability
        }
    
    def analyze_seasonal_patterns(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced seasonal pattern analysis for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Advanced seasonal analysis results per watershed
        """
        logger.info("Analyzing advanced seasonal patterns")
        
        seasonal_results = []
        
        for huc12_id in fire_watershed_df['huc12'].unique():
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            result = {'huc12': huc12_id}
            
            if len(watershed_fires) >= 3:
                # Day of year analysis
                doy_values = watershed_fires['day_of_year'].values
                
                # Circular statistics for day of year
                circular_stats = self._circular_statistics(doy_values)
                result.update(circular_stats)
                
                # Multi-modal seasonality detection
                seasonality_modes = self._detect_seasonality_modes(doy_values)
                result.update(seasonality_modes)
                
                # Seasonal consistency analysis
                consistency_metrics = self._analyze_seasonal_consistency(watershed_fires)
                result.update(consistency_metrics)
                
                # Fire season definition
                fire_season = self._define_fire_season(doy_values)
                result.update(fire_season)
                
            else:
                # Insufficient data
                result.update({
                    'seasonal_mean_doy': np.nan,
                    'seasonal_concentration': np.nan,
                    'seasonal_variance': np.nan,
                    'bimodal_seasonality': False,
                    'n_seasonal_modes': 0,
                    'seasonal_consistency_index': np.nan,
                    'fire_season_start_doy': np.nan,
                    'fire_season_end_doy': np.nan,
                    'fire_season_length_days': np.nan
                })
            
            seasonal_results.append(result)
        
        seasonal_df = pd.DataFrame(seasonal_results)
        
        logger.info(f"Advanced seasonal analysis completed:")
        logger.info(f"  - Watersheds with bimodal seasonality: {seasonal_df['bimodal_seasonality'].sum()}")
        
        return seasonal_df
    
    def _circular_statistics(self, doy_values: np.ndarray) -> Dict[str, float]:
        """Calculate circular statistics for day of year data."""
        # Convert day of year to radians
        angles = 2 * np.pi * doy_values / 365.25
        
        # Circular mean
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        mean_doy = (mean_angle * 365.25 / (2 * np.pi)) % 365.25
        
        # Circular variance and concentration
        R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
        circular_variance = 1 - R
        concentration = R  # Higher values = more concentrated
        
        return {
            'seasonal_mean_doy': mean_doy,
            'seasonal_concentration': concentration,
            'seasonal_variance': circular_variance
        }
    
    def _detect_seasonality_modes(self, doy_values: np.ndarray) -> Dict[str, Any]:
        """Detect multiple seasonal modes (e.g., spring and fall fire seasons)."""
        if len(doy_values) < 6:
            return {
                'bimodal_seasonality': False,
                'n_seasonal_modes': 0
            }
        
        # Use K-means clustering to detect seasonal modes
        doy_reshaped = doy_values.reshape(-1, 1)
        
        # Test for 1-3 modes
        best_n_modes = 1
        best_score = float('inf')
        
        for n_modes in range(1, min(4, len(doy_values) // 2 + 1)):
            try:
                kmeans = KMeans(n_clusters=n_modes, random_state=42, n_init=10)
                labels = kmeans.fit_predict(doy_reshaped)
                
                # Calculate silhouette-like score
                inertia = kmeans.inertia_
                score = inertia / len(doy_values)  # Normalized inertia
                
                if score < best_score and n_modes > 1:
                    # Check if clusters are well-separated
                    cluster_centers = kmeans.cluster_centers_.flatten()
                    if len(cluster_centers) > 1:
                        min_separation = np.min(np.diff(np.sort(cluster_centers)))
                        if min_separation > 30:  # At least 30 days apart
                            best_n_modes = n_modes
                            best_score = score
                        
            except:
                continue
        
        return {
            'bimodal_seasonality': best_n_modes == 2,
            'n_seasonal_modes': best_n_modes
        }
    
    def _analyze_seasonal_consistency(self, watershed_fires: pd.DataFrame) -> Dict[str, float]:
        """Analyze how consistent seasonal patterns are across years."""
        if len(watershed_fires) < 3:
            return {'seasonal_consistency_index': np.nan}
        
        # Group by year and calculate mean day of year
        yearly_mean_doy = watershed_fires.groupby('year')['day_of_year'].mean()
        
        if len(yearly_mean_doy) < 2:
            return {'seasonal_consistency_index': np.nan}
        
        # Calculate coefficient of variation of yearly means
        cv = yearly_mean_doy.std() / yearly_mean_doy.mean() if yearly_mean_doy.mean() > 0 else np.nan
        
        # Consistency index: higher values = more consistent
        consistency_index = 1 / (1 + cv) if not np.isnan(cv) else np.nan
        
        return {'seasonal_consistency_index': consistency_index}
    
    def _define_fire_season(self, doy_values: np.ndarray, percentile_range: Tuple[float, float] = (10, 90)) -> Dict[str, float]:
        """Define fire season boundaries using percentile-based approach."""
        if len(doy_values) < 3:
            return {
                'fire_season_start_doy': np.nan,
                'fire_season_end_doy': np.nan,
                'fire_season_length_days': np.nan
            }
        
        start_doy = np.percentile(doy_values, percentile_range[0])
        end_doy = np.percentile(doy_values, percentile_range[1])
        
        # Handle year wrap-around if necessary
        if end_doy < start_doy:
            # Fire season wraps around year end
            season_length = (365 - start_doy) + end_doy
        else:
            season_length = end_doy - start_doy
        
        return {
            'fire_season_start_doy': start_doy,
            'fire_season_end_doy': end_doy,
            'fire_season_length_days': season_length
        }
    
    def calculate_regime_change_indicators(self, fire_watershed_df: pd.DataFrame,
                                         split_year: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate indicators of fire regime change by comparing time periods.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            split_year: Year to split analysis (default: middle of time series)
            
        Returns:
            pd.DataFrame: Regime change indicators per watershed
        """
        logger.info("Calculating fire regime change indicators")
        
        change_results = []
        
        # Determine split year if not provided
        if split_year is None:
            all_years = fire_watershed_df['year'].unique()
            split_year = int(np.median(all_years))
            logger.info(f"Using median split year: {split_year}")
        
        for huc12_id in fire_watershed_df['huc12'].unique():
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            result = {'huc12': huc12_id, 'split_year': split_year}
            
            # Split data into two periods
            early_fires = watershed_fires[watershed_fires['year'] < split_year]
            late_fires = watershed_fires[watershed_fires['year'] >= split_year]
            
            early_years = range(watershed_fires['year'].min(), split_year) if len(early_fires) > 0 else []
            late_years = range(split_year, watershed_fires['year'].max() + 1) if len(late_fires) > 0 else []
            
            # Calculate metrics for each period
            if len(early_years) > 0 and len(late_years) > 0:
                early_freq = len(early_fires) / len(early_years)
                late_freq = len(late_fires) / len(late_years)
                
                # Frequency change
                freq_change = late_freq - early_freq
                freq_change_ratio = late_freq / early_freq if early_freq > 0 else np.inf if late_freq > 0 else np.nan
                
                result.update({
                    'early_period_frequency': early_freq,
                    'late_period_frequency': late_freq,
                    'frequency_change': freq_change,
                    'frequency_change_ratio': freq_change_ratio,
                    'frequency_change_significant': abs(freq_change) > 0.1  # Arbitrary threshold
                })
                
                # Seasonal change
                if len(early_fires) > 0 and len(late_fires) > 0:
                    early_mean_doy = early_fires['day_of_year'].mean()
                    late_mean_doy = late_fires['day_of_year'].mean()
                    seasonal_shift = late_mean_doy - early_mean_doy
                    
                    result.update({
                        'early_period_mean_doy': early_mean_doy,
                        'late_period_mean_doy': late_mean_doy,
                        'seasonal_shift_days': seasonal_shift,
                        'seasonal_shift_significant': abs(seasonal_shift) > 30  # >1 month shift
                    })
                else:
                    result.update({
                        'early_period_mean_doy': np.nan,
                        'late_period_mean_doy': np.nan,
                        'seasonal_shift_days': np.nan,
                        'seasonal_shift_significant': False
                    })
                
                # Overall regime change index
                freq_component = min(abs(freq_change) / 0.5, 1.0)  # Normalize by 0.5 fires/year
                seasonal_component = min(abs(seasonal_shift) / 60, 1.0) if not np.isnan(seasonal_shift) else 0  # Normalize by 60 days
                
                regime_change_index = (freq_component + seasonal_component) / 2
                
                result['regime_change_index'] = regime_change_index
                result['regime_change_detected'] = regime_change_index > 0.5
                
            else:
                # Insufficient data for comparison
                result.update({
                    'early_period_frequency': np.nan,
                    'late_period_frequency': np.nan,
                    'frequency_change': np.nan,
                    'frequency_change_ratio': np.nan,
                    'frequency_change_significant': False,
                    'early_period_mean_doy': np.nan,
                    'late_period_mean_doy': np.nan,
                    'seasonal_shift_days': np.nan,
                    'seasonal_shift_significant': False,
                    'regime_change_index': np.nan,
                    'regime_change_detected': False
                })
            
            change_results.append(result)
        
        change_df = pd.DataFrame(change_results)
        
        logger.info(f"Regime change analysis completed:")
        logger.info(f"  - Watersheds with regime changes detected: {change_df['regime_change_detected'].sum()}")
        logger.info(f"  - Watersheds with frequency increases: {(change_df['frequency_change'] > 0.1).sum()}")
        logger.info(f"  - Watersheds with frequency decreases: {(change_df['frequency_change'] < -0.1).sum()}")
        
        return change_df
    
    def create_temporal_summary(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal analysis summary.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Complete temporal analysis results
        """
        logger.info("Creating comprehensive temporal analysis summary")
        
        # Run all temporal analyses
        trend_df = self.analyze_fire_trends(fire_watershed_df)
        cycle_df = self.analyze_fire_cycles(fire_watershed_df)
        seasonal_df = self.analyze_seasonal_patterns(fire_watershed_df)
        change_df = self.calculate_regime_change_indicators(fire_watershed_df)
        
        # Merge all results
        temporal_summary = trend_df.copy()
        
        for df in [cycle_df, seasonal_df, change_df]:
            temporal_summary = temporal_summary.merge(df, on='huc12', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            temporal_summary = temporal_summary.loc[:, ~temporal_summary.columns.str.endswith('_dup')]
        
        logger.info(f"Temporal analysis summary created for {len(temporal_summary)} watersheds")
        
        return temporal_summary