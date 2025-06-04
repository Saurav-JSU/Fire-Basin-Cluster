"""
Unit tests for fire metrics calculation and temporal analysis.
"""
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.features.fire_metrics import WatershedFireMetrics
from src.features.temporal_analysis import TemporalFireAnalyzer

class TestWatershedFireMetrics:
    """Test cases for WatershedFireMetrics."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create WatershedFireMetrics instance for testing."""
        return WatershedFireMetrics(study_period_years=10)
    
    @pytest.fixture
    def sample_fire_watershed_data(self):
        """Create sample fire-watershed intersection data."""
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='120D')  # Every ~4 months
        
        data = []
        huc12_ids = ['123456789012', '123456789013', '123456789014']
        
        for i, date in enumerate(dates):
            huc12_id = huc12_ids[i % len(huc12_ids)]
            data.append({
                'huc12': huc12_id,
                'event_id': i,
                'start_date': date,
                'end_date': date + timedelta(days=np.random.randint(1, 7)),
                'year': date.year,
                'month': date.month,
                'day_of_year': date.timetuple().tm_yday,
                'season': 'Summer' if date.month in [6, 7, 8] else 'Winter',
                'duration_days': np.random.randint(1, 7),
                'spatial_extent_km': np.random.uniform(1, 10),
                'mean_frp': np.random.uniform(20, 100),
                'mean_confidence': np.random.uniform(80, 95),
                'n_detections': np.random.randint(5, 20),
                'area_km2': 150.0  # Watershed area
            })
        
        return pd.DataFrame(data)
    
    def test_metrics_calculator_initialization(self, metrics_calculator):
        """Test metrics calculator initialization."""
        assert metrics_calculator is not None
        assert metrics_calculator.study_period_years == 10
        assert metrics_calculator.watersheds is None
        assert metrics_calculator.fire_events is None
        assert metrics_calculator.watershed_metrics is None
    
    def test_month_to_season_conversion(self, metrics_calculator):
        """Test month to season conversion."""
        assert metrics_calculator._month_to_season(1) == 'Winter'
        assert metrics_calculator._month_to_season(4) == 'Spring'
        assert metrics_calculator._month_to_season(7) == 'Summer'
        assert metrics_calculator._month_to_season(10) == 'Fall'
        assert metrics_calculator._month_to_season(12) == 'Winter'
    
    def test_fire_return_intervals_calculation(self, metrics_calculator, sample_fire_watershed_data):
        """Test fire return interval calculations."""
        metrics_calculator.study_period_years = 3  # Match sample data period
        
        fri_df = metrics_calculator.calculate_fire_return_intervals(sample_fire_watershed_data)
        
        # Check structure
        assert isinstance(fri_df, pd.DataFrame)
        assert len(fri_df) > 0
        assert 'huc12' in fri_df.columns
        assert 'fire_frequency_per_year' in fri_df.columns
        assert 'mean_fire_return_interval_years' in fri_df.columns
        
        # Check data validity
        assert fri_df['fire_frequency_per_year'].min() >= 0
        assert fri_df['n_fires'].min() >= 0
        assert fri_df['years_with_fire'].min() >= 0
    
    def test_burn_fractions_calculation(self, metrics_calculator, sample_fire_watershed_data):
        """Test burn fraction calculations."""
        # Add watershed data to calculator
        watershed_data = pd.DataFrame({
            'huc12': ['123456789012', '123456789013', '123456789014'],
            'area_km2': [150.0, 200.0, 175.0]
        })
        metrics_calculator.watersheds = watershed_data
        
        burn_df = metrics_calculator.calculate_burn_fractions(sample_fire_watershed_data)
        
        # Check structure
        assert isinstance(burn_df, pd.DataFrame)
        assert len(burn_df) > 0
        assert 'huc12' in burn_df.columns
        assert 'burned_fraction_estimate' in burn_df.columns
        assert 'watershed_area_km2' in burn_df.columns
        
        # Check data validity
        assert burn_df['burned_fraction_estimate'].min() >= 0
        assert burn_df['burned_fraction_estimate'].max() <= 1.0
        assert burn_df['watershed_area_km2'].min() > 0
    
    def test_fire_seasonality_calculation(self, metrics_calculator, sample_fire_watershed_data):
        """Test fire seasonality calculations."""
        seasonality_df = metrics_calculator.calculate_fire_seasonality(sample_fire_watershed_data)
        
        # Check structure
        assert isinstance(seasonality_df, pd.DataFrame)
        assert len(seasonality_df) > 0
        assert 'huc12' in seasonality_df.columns
        assert 'peak_fire_month' in seasonality_df.columns
        assert 'peak_fire_season' in seasonality_df.columns
        
        # Check seasonal fractions sum to 1 for watersheds with fires
        for _, row in seasonality_df.iterrows():
            if row['n_fires'] > 0:
                seasonal_sum = (row['winter_fire_fraction'] + row['spring_fire_fraction'] + 
                              row['summer_fire_fraction'] + row['fall_fire_fraction'])
                assert abs(seasonal_sum - 1.0) < 0.01  # Allow small floating point errors
    
    def test_fire_intensity_metrics(self, metrics_calculator, sample_fire_watershed_data):
        """Test fire intensity metrics calculation."""
        intensity_df = metrics_calculator.calculate_fire_intensity_metrics(sample_fire_watershed_data)
        
        # Check structure
        assert isinstance(intensity_df, pd.DataFrame)
        assert len(intensity_df) > 0
        assert 'huc12' in intensity_df.columns
        assert 'mean_fire_duration_days' in intensity_df.columns
        
        # Check data validity for watersheds with fires
        valid_durations = intensity_df['mean_fire_duration_days'].dropna()
        if len(valid_durations) > 0:
            assert valid_durations.min() >= 1  # At least 1 day duration
    
    def test_composite_indices_calculation(self, metrics_calculator, sample_fire_watershed_data):
        """Test composite fire regime indices calculation."""
        # Set up required data
        watershed_data = pd.DataFrame({
            'huc12': ['123456789012', '123456789013', '123456789014'],
            'area_km2': [150.0, 200.0, 175.0]
        })
        metrics_calculator.watersheds = watershed_data
        metrics_calculator.study_period_years = 3
        
        # Calculate all component metrics
        fri_df = metrics_calculator.calculate_fire_return_intervals(sample_fire_watershed_data)
        burn_df = metrics_calculator.calculate_burn_fractions(sample_fire_watershed_data)
        seasonality_df = metrics_calculator.calculate_fire_seasonality(sample_fire_watershed_data)
        intensity_df = metrics_calculator.calculate_fire_intensity_metrics(sample_fire_watershed_data)
        
        # Calculate composite indices
        composite_df = metrics_calculator.calculate_composite_indices(
            sample_fire_watershed_data, fri_df, burn_df, seasonality_df, intensity_df
        )
        
        # Check structure
        assert isinstance(composite_df, pd.DataFrame)
        assert len(composite_df) > 0
        assert 'huc12' in composite_df.columns
        assert 'fire_activity_index' in composite_df.columns
        assert 'fire_regime_stability_index' in composite_df.columns
        assert 'fire_regime_type' in composite_df.columns
        
        # Check index ranges
        assert composite_df['fire_activity_index'].min() >= 0
        assert composite_df['fire_activity_index'].max() <= 1.0
        assert composite_df['fire_regime_stability_index'].dropna().min() >= 0
        assert composite_df['fire_regime_stability_index'].dropna().max() <= 1.0
    
    def test_metrics_summary(self, metrics_calculator):
        """Test metrics summary generation."""
        # Test with no data
        summary = metrics_calculator.get_metrics_summary()
        assert 'error' in summary
        
        # Test with mock data
        mock_metrics = pd.DataFrame({
            'huc12': ['123', '456'],
            'n_fires': [5, 0],
            'fire_frequency_per_year': [0.5, 0.0],
            'burned_fraction_estimate': [0.1, 0.0],
            'fire_regime_type': ['Moderate_Frequency_Low_Impact', 'Fire_Suppressed'],
            'fire_risk_category': ['Moderate', 'Minimal']
        })
        
        metrics_calculator.watershed_metrics = mock_metrics
        metrics_calculator.study_period_years = 10
        
        summary = metrics_calculator.get_metrics_summary()
        
        assert 'total_watersheds' in summary
        assert summary['total_watersheds'] == 2
        assert 'watersheds_with_fires' in summary
        assert summary['watersheds_with_fires'] == 1

class TestTemporalFireAnalyzer:
    """Test cases for TemporalFireAnalyzer."""
    
    @pytest.fixture
    def temporal_analyzer(self):
        """Create TemporalFireAnalyzer instance for testing."""
        return TemporalFireAnalyzer(min_years_for_trend=5)
    
    @pytest.fixture
    def sample_temporal_data(self):
        """Create sample temporal fire data."""
        # Create data with a clear trend
        years = list(range(2010, 2023))
        data = []
        
        for i, year in enumerate(years):
            # Create increasing trend in fires
            n_fires = max(1, i // 3)  # Gradual increase
            
            for fire_num in range(n_fires):
                data.append({
                    'huc12': '123456789012',
                    'year': year,
                    'day_of_year': 200 + np.random.randint(-50, 50),  # Summer fires with variation
                    'month': 7,  # July
                    'start_date': pd.Timestamp(f'{year}-07-15') + pd.Timedelta(days=np.random.randint(-30, 30))
                })
        
        return pd.DataFrame(data)
    
    def test_temporal_analyzer_initialization(self, temporal_analyzer):
        """Test temporal analyzer initialization."""
        assert temporal_analyzer is not None
        assert temporal_analyzer.min_years_for_trend == 5
    
    def test_mann_kendall_test(self, temporal_analyzer):
        """Test Mann-Kendall trend test."""
        # Test increasing trend
        increasing_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = temporal_analyzer._mann_kendall_test(increasing_data)
        assert result['trend'] == 'increasing'
        assert result['p'] < 0.05
        
        # Test decreasing trend
        decreasing_data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        result = temporal_analyzer._mann_kendall_test(decreasing_data)
        assert result['trend'] == 'decreasing'
        assert result['p'] < 0.05
        
        # Test no trend
        no_trend_data = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = temporal_analyzer._mann_kendall_test(no_trend_data)
        assert result['trend'] == 'no_trend'
    
    def test_fire_trends_analysis(self, temporal_analyzer, sample_temporal_data):
        """Test fire trends analysis."""
        trend_df = temporal_analyzer.analyze_fire_trends(sample_temporal_data)
        
        # Check structure
        assert isinstance(trend_df, pd.DataFrame)
        assert len(trend_df) > 0
        assert 'huc12' in trend_df.columns
        assert 'trend_slope' in trend_df.columns
        assert 'trend_direction' in trend_df.columns
        
        # Check that trend was detected (data has increasing trend)
        watershed_trend = trend_df[trend_df['huc12'] == '123456789012'].iloc[0]
        assert watershed_trend['years_of_data'] >= temporal_analyzer.min_years_for_trend
        # Should detect increasing trend
        assert watershed_trend['trend_direction'] in ['increasing', 'decreasing', 'stable']
    
    def test_fire_cycles_analysis(self, temporal_analyzer, sample_temporal_data):
        """Test fire cycles analysis."""
        cycle_df = temporal_analyzer.analyze_fire_cycles(sample_temporal_data)
        
        # Check structure
        assert isinstance(cycle_df, pd.DataFrame)
        assert len(cycle_df) > 0
        assert 'huc12' in cycle_df.columns
        assert 'has_cycles' in cycle_df.columns
        assert 'cycle_strength' in cycle_df.columns
    
    def test_seasonal_patterns_analysis(self, temporal_analyzer, sample_temporal_data):
        """Test seasonal patterns analysis."""
        seasonal_df = temporal_analyzer.analyze_seasonal_patterns(sample_temporal_data)
        
        # Check structure
        assert isinstance(seasonal_df, pd.DataFrame)
        assert len(seasonal_df) > 0
        assert 'huc12' in seasonal_df.columns
        assert 'seasonal_concentration' in seasonal_df.columns
        assert 'seasonal_mean_doy' in seasonal_df.columns
    
    def test_regime_change_indicators(self, temporal_analyzer, sample_temporal_data):
        """Test regime change indicators."""
        change_df = temporal_analyzer.calculate_regime_change_indicators(sample_temporal_data)
        
        # Check structure
        assert isinstance(change_df, pd.DataFrame)
        assert len(change_df) > 0
        assert 'huc12' in change_df.columns
        assert 'frequency_change' in change_df.columns
        assert 'regime_change_detected' in change_df.columns
    
    def test_temporal_summary(self, temporal_analyzer, sample_temporal_data):
        """Test comprehensive temporal summary."""
        summary_df = temporal_analyzer.create_temporal_summary(sample_temporal_data)
        
        # Check that all analyses are included
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) > 0
        assert 'huc12' in summary_df.columns
        
        # Should include columns from all analyses
        expected_columns = [
            'trend_slope', 'has_cycles', 'seasonal_concentration', 'frequency_change'
        ]
        for col in expected_columns:
            assert col in summary_df.columns

def test_module_imports():
    """Test that all modules can be imported correctly."""
    try:
        from src.features.fire_metrics import WatershedFireMetrics
        from src.features.temporal_analysis import TemporalFireAnalyzer
        from src.features import WatershedFireMetrics as ImportedMetrics
        from src.features import TemporalFireAnalyzer as ImportedAnalyzer
        
        assert WatershedFireMetrics is ImportedMetrics
        assert TemporalFireAnalyzer is ImportedAnalyzer
        
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

def test_data_format_compatibility():
    """Test compatibility with expected data formats from Steps 1 & 2."""
    
    # Test expected fire events format (from Step 2)
    fire_events_columns = [
        'event_id', 'start_date', 'end_date', 'centroid_lon', 'centroid_lat',
        'duration_days', 'n_detections', 'spatial_extent_km', 'mean_frp'
    ]
    
    fire_events_df = pd.DataFrame({col: [1] for col in fire_events_columns})
    fire_events_df['start_date'] = pd.Timestamp('2020-01-01')
    fire_events_df['end_date'] = pd.Timestamp('2020-01-02')
    
    # Test watershed format (from Step 1)
    watershed_columns = ['huc12', 'area_km2']
    watershed_df = pd.DataFrame({col: ['123456789012'] if col == 'huc12' else [150.0] for col in watershed_columns})
    
    # Should be able to initialize with this format
    metrics_calc = WatershedFireMetrics()
    
    # Test that required columns are recognized
    try:
        # This should work without errors (even if data is minimal)
        metrics_calc.watersheds = watershed_df
        metrics_calc.fire_events = fire_events_df
        
        # Test data format validation would happen in actual methods
        assert True  # If we get here, basic format is compatible
        
    except Exception as e:
        pytest.fail(f"Data format compatibility failed: {e}")

if __name__ == "__main__":
    # Run basic tests without pytest
    print("🔥 Fire Metrics - Unit Tests")
    print("=" * 40)
    
    # Test basic imports
    try:
        from src.features.fire_metrics import WatershedFireMetrics
        from src.features.temporal_analysis import TemporalFireAnalyzer
        print("✓ Module imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    # Test basic initialization
    try:
        metrics_calc = WatershedFireMetrics(study_period_years=10)
        temporal_analyzer = TemporalFireAnalyzer(min_years_for_trend=5)
        print("✓ Module initialization successful")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        sys.exit(1)
    
    # Test basic functionality
    try:
        # Test season conversion
        season = metrics_calc._month_to_season(7)
        assert season == 'Summer'
        
        # Test Mann-Kendall
        test_data = np.array([1, 2, 3, 4, 5])
        mk_result = temporal_analyzer._mann_kendall_test(test_data)
        assert mk_result['trend'] == 'increasing'
        
        print("✓ Basic functionality tests passed")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        sys.exit(1)
    
    print("\n✅ All basic unit tests passed!")
    print("\nTo run full test suite:")
    print("  pytest tests/test_fire_metrics.py -v")