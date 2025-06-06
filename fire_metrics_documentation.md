# Fire Metrics Documentation

This document provides detailed explanations of various fire-related metrics used in wildfire analysis.

## Basic Identification
- **huc12**: Hydrologic Unit Code (HUC) at level 12, which is the smallest watershed unit in the USGS watershed hierarchy.

## Fire Count and Frequency Metrics
- **n_fires**: Total number of fires recorded in the watershed
- **years_with_fire**: Number of years in which at least one fire occurred
- **fire_frequency_per_year**: Average number of fires per year
- **fire_years_fraction**: Proportion of years that experienced at least one fire
- **mean_fire_return_interval_years**: Average number of years between consecutive fires
- **median_fire_return_interval_years**: Middle value of fire return intervals
- **std_fire_return_interval_years**: Standard deviation of fire return intervals
- **min_fire_return_interval_years**: Shortest observed interval between fires
- **max_fire_return_interval_years**: Longest observed interval between fires
- **cv_fire_return_interval**: Coefficient of variation of fire return intervals (std/mean)
- **years_since_last_fire**: Number of years elapsed since the most recent fire
- **max_fire_free_period_years**: Longest period without any fires

## Area and Size Metrics
- **watershed_area_km2**: Total area of the watershed in square kilometers
- **total_estimated_burned_area_km2**: Total area burned by all fires
- **burned_fraction_estimate**: Proportion of watershed area that has burned
- **mean_fire_size_km2**: Average size of fires
- **median_fire_size_km2**: Middle value of fire sizes
- **largest_fire_size_km2**: Size of the largest recorded fire
- **largest_fire_fraction**: Proportion of total burned area accounted for by the largest fire
- **fire_size_cv**: Coefficient of variation of fire sizes

## Multiple Fire Metrics
- **years_with_multiple_fires**: Number of years with more than one fire
- **multiple_fire_years_fraction**: Proportion of years with multiple fires

## Temporal Fire Patterns
- **peak_fire_month**: Month with the highest number of fires
- **peak_fire_season**: Season with the highest number of fires
- **fire_season_length_months**: Duration of the fire season
- **fire_concentration_peak_month**: Month with the highest concentration of fire activity
- **fire_concentration_peak_season**: Season with the highest concentration of fire activity
- **mean_fire_day_of_year**: Average day of year when fires occur
- **std_fire_day_of_year**: Standard deviation of fire occurrence days
- **earliest_fire_day_of_year**: Earliest recorded fire day
- **latest_fire_day_of_year**: Latest recorded fire day

## Seasonal Fire Distribution
- **winter_fire_fraction**: Proportion of fires occurring in winter
- **spring_fire_fraction**: Proportion of fires occurring in spring
- **summer_fire_fraction**: Proportion of fires occurring in summer
- **fall_fire_fraction**: Proportion of fires occurring in fall

## Monthly Fire Distribution
- **month_XX_fire_fraction**: Proportion of fires occurring in month XX (01-12)

## Fire Radiative Power (FRP) Metrics
- **mean_watershed_frp**: Average Fire Radiative Power
- **median_watershed_frp**: Median Fire Radiative Power
- **max_watershed_frp**: Maximum Fire Radiative Power
- **std_watershed_frp**: Standard deviation of Fire Radiative Power
- **cv_watershed_frp**: Coefficient of variation of Fire Radiative Power
- **frp_data_available**: Boolean indicating if FRP data is available

## Fire Duration Metrics
- **mean_fire_duration_days**: Average duration of fires
- **median_fire_duration_days**: Median duration of fires
- **max_fire_duration_days**: Longest recorded fire duration
- **std_fire_duration_days**: Standard deviation of fire durations
- **single_day_fires_fraction**: Proportion of fires lasting only one day
- **long_duration_fires_fraction**: Proportion of fires lasting longer than average

## Fire Detection and Confidence Metrics
- **mean_fire_confidence**: Average confidence level in fire detection
- **min_fire_confidence**: Minimum confidence level in fire detection
- **std_fire_confidence**: Standard deviation of fire detection confidence
- **total_fire_detections**: Total number of fire detections
- **mean_detections_per_fire**: Average number of detections per fire
- **max_detections_per_fire**: Maximum number of detections for a single fire

## Composite Indices
- **fire_activity_index**: Composite measure of overall fire activity
- **fire_regime_stability_index**: Measure of stability in fire patterns
- **fire_intensity_index**: Composite measure of fire intensity
- **fire_regime_type**: Classification of the fire regime
- **fire_risk_category**: Categorization of fire risk level 