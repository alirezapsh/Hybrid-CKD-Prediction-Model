import numpy as np
import pandas as pd
import os
import tempfile
import re
from scipy.io import arff
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class MovingAverageImputer:
    """
    Moving Average Imputer for handling missing values.
    Uses a rolling window to compute the mean of neighboring values.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize Moving Average Imputer.
        
        Args:
            window_size: Size of the rolling window (default: 3)
        """
        self.window_size = window_size
        self.feature_means = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'MovingAverageImputer':
        """
        Fit the imputer on the data.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            self: Returns self for method chaining
        """
        # Store mean values for each column as fallback
        self.feature_means = X.mean().to_dict()
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by imputing missing values using moving average.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transformation.")
        
        X_imputed = X.copy()
        missing_count = X_imputed.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"   Imputing {missing_count} missing values using Moving Average (window={self.window_size})...")
            
            # For each column with missing values
            for col in X_imputed.columns:
                if X_imputed[col].isnull().any():
                    # Use rolling mean with forward and backward fill
                    X_imputed[col] = X_imputed[col].fillna(
                        X_imputed[col].rolling(window=self.window_size, min_periods=1).mean()
                    )
                    # Fill remaining NaN with column mean
                    X_imputed[col] = X_imputed[col].fillna(self.feature_means[col])
        else:
            print("   No missing values found. Skipping imputation.")
        
        return X_imputed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the data.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        return self.fit(X).transform(X)


class KNNImputer:
    """
    K-Nearest Neighbors Imputer for handling missing values.
    Wrapper around sklearn's KNNImputer with DataFrame support.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Initialize KNN Imputer.
        
        Args:
            n_neighbors: Number of neighbors to use for imputation (default: 5)
            weights: Weight function used in prediction ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = SklearnKNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'KNNImputer':
        """
        Fit the KNN imputer on the data.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            self: Returns self for method chaining
        """
        self.feature_names = X.columns.tolist()
        self.imputer.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by imputing missing values.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transformation.")
        
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"   Imputing {missing_count} missing values using KNN (k={self.n_neighbors})...")
            X_imputed = self.imputer.transform(X)
            X_imputed = pd.DataFrame(X_imputed, columns=self.feature_names, index=X.index)
        else:
            print("   No missing values found. Skipping imputation.")
            X_imputed = X.copy()
        
        return X_imputed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the data.
        
        Args:
            X: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        return self.fit(X).transform(X)


def load_arff_file(file_path: str) -> pd.DataFrame:
    """
    Load ARFF file and convert to pandas DataFrame.
    Handles whitespace issues in ARFF files by preprocessing before loading.
    
    Args:
        file_path: Path to the ARFF file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading ARFF file: {file_path}")
    
    # Preprocess ARFF file to handle whitespace issues
    # Read file and strip whitespace from data values
    try:
        # Try loading directly first (for clean files)
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
    except (ValueError, UnicodeDecodeError) as e:
        # If loading fails due to whitespace or encoding issues, use manual parsing
        print(f"   Detected whitespace/encoding issues, using manual ARFF parser...")
        
        # Use manual parsing as fallback
        df = _load_arff_manual(file_path)
        return df
        
        # Process lines: strip whitespace from data values (after @data)
        in_data_section = False
        processed_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if we're entering the data section
            if stripped_line.lower().startswith('@data'):
                in_data_section = True
                processed_lines.append(line)
            elif in_data_section:
                # In data section
                if not stripped_line or stripped_line.startswith('%'):
                    # Empty line or comment - keep as-is
                    processed_lines.append(line)
                else:
                    # Data row - clean whitespace from each value
                    # First strip the entire line to remove leading/trailing whitespace
                    line_clean = stripped_line.strip()  # Ensure it's stripped (no leading/trailing spaces)
                    parts = line_clean.split(',')
                    # Strip whitespace from each value
                    cleaned_parts = []
                    for part in parts:
                        cleaned = part.strip()
                        # Preserve empty values as '?' (ARFF standard for missing)
                        if not cleaned:
                            cleaned = '?'
                        cleaned_parts.append(cleaned)
                    processed_lines.append(','.join(cleaned_parts) + '\n')
            else:
                # Before data section - keep as-is (but could also clean attribute declarations)
                # Clean attribute declarations to remove whitespace from value lists
                if stripped_line.lower().startswith('@attribute') and '{' in line and '}' in line:
                    # Attribute with nominal values - clean the value list
                    # Extract attribute name and type
                    match = re.match(r'@attribute\s+(\S+)\s+(.+)', stripped_line, re.IGNORECASE)
                    if match:
                        attr_name = match.group(1)
                        attr_type = match.group(2)
                        # Clean whitespace in nominal value list
                        if '{' in attr_type and '}' in attr_type:
                            # Extract values between braces
                            start = attr_type.find('{')
                            end = attr_type.find('}')
                            if start != -1 and end != -1:
                                values_str = attr_type[start+1:end]
                                # Split by comma and strip each value, filter out empty
                                values = [v.strip() for v in values_str.split(',') if v.strip()]
                                # Reconstruct attribute declaration
                                cleaned_attr_type = attr_type[:start+1] + ','.join(values) + attr_type[end:]
                                processed_lines.append(f'@attribute {attr_name} {cleaned_attr_type}\n')
                            else:
                                processed_lines.append(line)
                        else:
                            processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
        
        # Write to temporary file and try loading
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.arff', delete=False, encoding='utf-8', newline='') as tmp_file:
                tmp_file.writelines(processed_lines)
                tmp_file_path = tmp_file.name
            
            # Load from temporary file
            data, meta = arff.loadarff(tmp_file_path)
            df = pd.DataFrame(data)
        except Exception as e2:
            # If still fails, try more aggressive whitespace removal
            print(f"   First preprocessing attempt failed, trying more aggressive cleaning...")
            
            # More aggressive approach: remove ALL spaces around commas and in value lists
            processed_lines2 = []
            in_data_section2 = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith('@data'):
                    in_data_section2 = True
                    processed_lines2.append(line)
                elif in_data_section2:
                    if not stripped or stripped.startswith('%'):
                        processed_lines2.append(line)
                    else:
                        # Remove ALL whitespace around commas using regex
                        cleaned = re.sub(r'\s*,\s*', ',', stripped)
                        processed_lines2.append(cleaned + '\n')
                else:
                    # Clean attribute declarations more aggressively
                    if stripped.lower().startswith('@attribute') and '{' in stripped:
                        # Remove all spaces in value lists: { value1 , value2 } -> {value1,value2}
                        cleaned = re.sub(r'\{\s+', '{', stripped)
                        cleaned = re.sub(r'\s+\}', '}', cleaned)
                        cleaned = re.sub(r',\s+', ',', cleaned)
                        cleaned = re.sub(r'\s+,', ',', cleaned)
                        processed_lines2.append(cleaned + '\n')
                    else:
                        processed_lines2.append(line)
            
            # Clean up old temp file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            
            # Write new cleaned file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.arff', delete=False, encoding='utf-8', newline='') as tmp_file2:
                tmp_file2.writelines(processed_lines2)
                tmp_file_path = tmp_file2.name
            
            try:
                data, meta = arff.loadarff(tmp_file_path)
                df = pd.DataFrame(data)
            except Exception as e3:
                print(f"   All preprocessing attempts failed. Error: {e3}")
                raise e3
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    # Decode byte strings to regular strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    
    # Strip any remaining whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings and '?' with NaN
            df[col] = df[col].replace(['', '?', 'nan', 'None'], np.nan)
    
    # Replace '?' with NaN for missing values (in case some weren't caught)
    df = df.replace('?', np.nan)
    
    # Convert numeric columns
    for col in df.columns:
        if col != 'class':  # Keep class as string for now
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    print(f"   Loaded {len(df)} samples with {len(df.columns)} features")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df


def _load_arff_manual(file_path: str) -> pd.DataFrame:
    """
    Manual ARFF parser that handles whitespace issues robustly.
    This is a fallback when scipy's parser fails.
    """
    print(f"   Using manual ARFF parser...")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Parse ARFF file manually
    attributes = {}
    attribute_names = []
    data_started = False
    data_rows = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        
        # Check for @relation
        if stripped.lower().startswith('@relation'):
            continue
        
        # Check for @attribute
        if stripped.lower().startswith('@attribute'):
            # Parse attribute: @attribute name type
            # Handle quoted attribute names: @attribute "name" type
            stripped_clean = stripped[len('@attribute'):].strip()
            
            # Check if name is quoted
            if stripped_clean.startswith('"'):
                # Find closing quote
                end_quote = stripped_clean.find('"', 1)
                if end_quote != -1:
                    attr_name = stripped_clean[1:end_quote].strip()
                    attr_type = stripped_clean[end_quote+1:].strip()
                else:
                    # Malformed, try regular parsing
                    parts = stripped.split(None, 2)
                    if len(parts) >= 3:
                        attr_name = parts[1].strip()
                        attr_type = parts[2].strip()
                    else:
                        continue
            else:
                # Regular parsing - split on whitespace
                parts = stripped.split(None, 2)  # Split on whitespace, max 2 splits
                if len(parts) >= 3:
                    attr_name = parts[1].strip()
                    attr_type = parts[2].strip()
                else:
                    continue
            
            # Clean attribute name (remove any remaining quotes or whitespace)
            attr_name = attr_name.strip('"\'')
            
            # Handle nominal types: {value1, value2, ...}
            if attr_type.startswith('{') and attr_type.endswith('}'):
                # Extract values and clean whitespace
                values_str = attr_type[1:-1]  # Remove braces
                values = [v.strip() for v in values_str.split(',') if v.strip()]
                attributes[attr_name] = {'type': 'nominal', 'values': values}
            else:
                attributes[attr_name] = {'type': attr_type.lower()}
            
            attribute_names.append(attr_name)
            continue
        
        # Check for @data
        if stripped.lower().startswith('@data'):
            data_started = True
            continue
        
        # Parse data rows
        if data_started and stripped:
            # Split by comma and strip each value
            values = []
            for val in stripped.split(','):
                val_clean = val.strip()
                # Handle missing values
                if not val_clean or val_clean == '?':
                    values.append(None)
                else:
                    values.append(val_clean)
            
            # Handle rows with extra commas (common ARFF file issue)
            # Case 1: Row has exactly one extra value (26 for 25 attributes)
            if len(values) == len(attribute_names) + 1:
                # Check if last value is empty/None (from trailing comma like "ckd,")
                if not values[-1] or values[-1] is None or values[-1] == '':
                    values = values[:-1]  # Remove the trailing empty value
                # Check if last value is a duplicate of the class value
                elif len(attribute_names) > 0:
                    class_idx = len(attribute_names) - 1
                    if class_idx < len(values) - 1:
                        # If last two values are the same (duplicate class), remove last
                        if values[class_idx] == values[-1]:
                            values = values[:-1]
                        # Otherwise, if last value is empty, remove it
                        elif not values[-1] or values[-1] == '':
                            values = values[:-1]
                        # Last resort: if we still have 26 values, check for empty values in the middle
                        # and remove the first empty one we find (handles "no,,no" case)
                        if len(values) == len(attribute_names) + 1:
                            # Find first empty value (from double comma) and remove it
                            for i in range(len(values) - 1):  # Don't check last value
                                if not values[i] or values[i] == '' or values[i] is None:
                                    values.pop(i)
                                    break
            
            # Only add if we have the right number of values (after fixing)
            if len(values) == len(attribute_names):
                data_rows.append(values)
            elif len(values) > 0:
                # Log warning for rows that still don't match (for debugging)
                print(f"   [WARNING] Skipping row with {len(values)} values (expected {len(attribute_names)}): {stripped[:80]}...")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=attribute_names)
    
    # Debug: print column names to help diagnose issues
    print(f"   Parsed {len(attribute_names)} attributes: {attribute_names[:5]}{'...' if len(attribute_names) > 5 else ''}")
    if 'class' not in df.columns:
        # Check for case variations or whitespace
        class_variants = [col for col in df.columns if 'class' in col.lower()]
        if class_variants:
            print(f"   [WARNING] 'class' not found, but found similar columns: {class_variants}")
    
    # Convert types
    for col in df.columns:
        if col in attributes:
            attr_info = attributes[col]
            if attr_info['type'] == 'nominal':
                # Keep as string for nominal (will be encoded later)
                df[col] = df[col].astype(str)
                # Replace 'None' and 'nan' strings with actual NaN
                df[col] = df[col].replace(['None', 'nan', 'NaN', '?'], np.nan)
            elif attr_info['type'] in ['numeric', 'real', 'integer']:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # Try to convert to numeric, fallback to string
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # If conversion resulted in all NaN, it's probably categorical
                    if df[col].isna().all():
                        df[col] = df[col].astype(str)
                except:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace(['None', 'nan', 'NaN', '?'], np.nan)
    
    # Replace 'nan' strings and None with NaN
    df = df.replace(['nan', 'None', '?', None], np.nan)
    
    print(f"   Manually loaded {len(df)} samples with {len(df.columns)} features")
    return df


def remove_outliers_iqr(
    X: pd.DataFrame, 
    y: pd.Series, 
    factor: float = 2.5,
    min_outlier_features: int = 3
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers using Interquartile Range (IQR) method (less aggressive).
    Binary columns are excluded from IQR calculation as they represent categorical/risk factors.
    
    Args:
        X: Input features DataFrame
        y: Target Series
        factor: IQR multiplier (default: 2.5, less aggressive than 1.5)
        min_outlier_features: Minimum number of features that must be outliers 
                             for a row to be removed (default: 3)
        
    Returns:
        Tuple of (X_cleaned, y_cleaned) with outliers removed
    """
    X_cleaned = X.copy()
    y_cleaned = y.copy()
    
    initial_count = len(X_cleaned)
    
    # Identify binary columns (columns that only contain 0 and 1, or have ≤2 unique values)
    # These should be excluded from IQR as they represent categorical/risk factors, not continuous variables
    binary_columns = []
    continuous_columns = []
    
    for col in X_cleaned.columns:
        unique_vals = set(X_cleaned[col].dropna().unique())
        # Check if column is binary (only 0 and 1, or only 2 unique values that are 0 and 1)
        if len(unique_vals) <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary_columns.append(col)
        else:
            continuous_columns.append(col)
    
    print(f"   📊 IQR Column Analysis:")
    print(f"      Binary columns (excluded from IQR): {len(binary_columns)} columns")
    print(f"      Continuous columns (IQR applied): {len(continuous_columns)} columns")
    if len(binary_columns) > 0:
        print(f"      Binary columns: {', '.join(binary_columns[:10])}{'...' if len(binary_columns) > 10 else ''}")
    
    # Only calculate IQR for continuous columns
    if len(continuous_columns) == 0:
        print(f"   ⚠️  No continuous columns found. Skipping IQR outlier removal.")
        return X_cleaned, y_cleaned
    
    X_continuous = X_cleaned[continuous_columns]
    
    # Calculate IQR for continuous columns only
    Q1 = X_continuous.quantile(0.25)
    Q3 = X_continuous.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (only for continuous columns)
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Identify outliers per feature (only for continuous columns)
    outlier_per_feature_continuous = (X_continuous < lower_bound) | (X_continuous > upper_bound)
    
    # Create full DataFrame with False for binary columns (they don't contribute to outlier count)
    outlier_per_feature = pd.DataFrame(False, index=X_cleaned.index, columns=X_cleaned.columns)
    outlier_per_feature[continuous_columns] = outlier_per_feature_continuous
    
    # Count how many features are outliers for each row
    outlier_count_per_row = outlier_per_feature.sum(axis=1)
    
    # Only remove rows where at least min_outlier_features are outliers
    outlier_mask = outlier_count_per_row >= min_outlier_features
    
    # Get removed samples information BEFORE removing
    removed_indices = X_cleaned.index[outlier_mask].tolist()
    removed_X = X_cleaned.loc[outlier_mask]
    removed_y = y_cleaned.loc[outlier_mask]
    
    # Count CKD vs No-CKD in removed samples
    removed_class_counts = removed_y.value_counts().to_dict()
    removed_ckd = removed_class_counts.get(1, 0)
    removed_no_ckd = removed_class_counts.get(0, 0)
    
    # For each removed sample, identify which features caused it to be an outlier
    removed_samples_info = []
    for idx in removed_indices:
        sample_outlier_features = outlier_per_feature.loc[idx]
        outlier_feature_names = sample_outlier_features[sample_outlier_features].index.tolist()
        
        # Get detailed outlier information: value, bounds, and reason
        # Only include continuous columns (binary columns are excluded from IQR)
        outlier_details = []
        sample_values = removed_X.loc[idx]
        for col in outlier_feature_names:
            # Only process continuous columns (binary columns shouldn't be here, but check to be safe)
            if col in continuous_columns:
                value = sample_values[col]
                lb = lower_bound[col]
                ub = upper_bound[col]
                if value < lb:
                    reason = f"TOO LOW (value={value:.4f} < lower_bound={lb:.4f})"
                elif value > ub:
                    reason = f"TOO HIGH (value={value:.4f} > upper_bound={ub:.4f})"
                else:
                    reason = "UNKNOWN"
                outlier_details.append({
                    'column': col,
                    'value': value,
                    'lower_bound': lb,
                    'upper_bound': ub,
                    'reason': reason
                })
        
        removed_samples_info.append({
            'index': idx,
            'label': int(removed_y.loc[idx]),
            'outlier_features': outlier_feature_names,
            'outlier_details': outlier_details,
            'num_outlier_features': len(outlier_feature_names)
        })
    
    # Remove outliers
    X_cleaned = X_cleaned[~outlier_mask]
    y_cleaned = y_cleaned[~outlier_mask]
    
    removed_count = initial_count - len(X_cleaned)
    removal_percentage = (removed_count / initial_count) * 100
    
    print(f"   Removed {removed_count} outliers ({removal_percentage:.2f}% of data)")
    print(f"   Remaining samples: {len(X_cleaned)}")
    print(f"   Criteria: IQR factor={factor}, min outlier features={min_outlier_features}")
    print(f"   ✅ Binary columns excluded from IQR: {len(binary_columns)} columns preserved")
    
    # Print detailed information about removed samples
    print(f"\n   📊 DETAILED OUTLIER REMOVAL ANALYSIS:")
    print(f"      Total removed: {removed_count} samples")
    if removed_count > 0:
        print(f"      CKD samples removed: {removed_ckd} ({removed_ckd/removed_count*100:.1f}%)")
        print(f"      No-CKD samples removed: {removed_no_ckd} ({removed_no_ckd/removed_count*100:.1f}%)")
    else:
        print(f"      CKD samples removed: {removed_ckd} (0.0%)")
        print(f"      No-CKD samples removed: {removed_no_ckd} (0.0%)")
        print(f"      ✅ No outliers detected - all samples retained")
    
    # Count which features most commonly cause outliers
    all_outlier_features = []
    for info in removed_samples_info:
        all_outlier_features.extend(info['outlier_features'])
    
    if all_outlier_features:
        from collections import Counter
        feature_outlier_counts = Counter(all_outlier_features)
        top_outlier_features = feature_outlier_counts.most_common(10)
        
        print(f"\n   🔍 Top 10 Features Most Often Causing Outliers:")
        for feature, count in top_outlier_features:
            print(f"      {feature}: {count} samples ({count/removed_count*100:.1f}%)")
    
    return X_cleaned, y_cleaned


def normalize_features(
    X: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using Min-Max Scaling.
    
    Args:
        X: Input features DataFrame
        scaler: Pre-fitted scaler (if None, fits a new one)
        
    Returns:
        Tuple of (X_normalized, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        X_normalized = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
        print(f"   Min-Max normalization completed (range: 0-1)")
    else:
        X_normalized = scaler.transform(X)
        X_normalized = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
        print(f"   Min-Max normalization applied using fitted scaler")
    
    return X_normalized, scaler


def select_features_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    k: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], SelectKBest]:
    """
    Select features using Mutual Information.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        X_test: Test features
        k: Number of top features to select (if None, uses all features with MI > 0)
        
    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selected_features, selector)
    """
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    # Create feature selector
    if k is None:
        # Select features with MI > 0
        k = sum(mi_scores > 0)
        print(f"   Selecting features with MI > 0: {k} features")
    else:
        print(f"   Selecting top {k} features based on Mutual Information")
    
    # Create a wrapper function to pass random_state to mutual_info_classif
    # This ensures SelectKBest uses the same random_state for consistency
    def mi_score_func(X, y):
        return mutual_info_classif(X, y, random_state=42)
    
    selector = SelectKBest(score_func=mi_score_func, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Convert back to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_val_selected = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    # Display top features
    feature_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    print(f"   Top 10 features by Mutual Information:")
    for idx, row in feature_scores.head(10).iterrows():
        print(f"      {row['Feature']}: {row['MI_Score']:.4f}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'class',
    exclude_columns: Optional[List[str]] = None,
    imputation_method: str = 'knn',
    knn_neighbors: int = 5,
    moving_avg_window: int = 3,
    remove_outliers: bool = True,
    outlier_factor: float = 2.5,
    min_outlier_features: int = 3,
    normalize: bool = True,
    use_feature_selection: bool = True,
    n_features: Optional[int] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    handle_categorical: bool = True,
    use_cross_validation: bool = True,
    cv_folds: int = 5
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
    """
    Comprehensive data preprocessing pipeline for CKD dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        exclude_columns: List of columns to exclude from features
        imputation_method: Method for handling missing values ('knn' or 'moving_avg')
        knn_neighbors: Number of neighbors for KNN imputation
        moving_avg_window: Window size for moving average imputation
        remove_outliers: Whether to remove outliers using IQR method
        outlier_factor: IQR multiplier for outlier detection (default: 2.5, less aggressive)
        min_outlier_features: Minimum number of outlier features to remove a row (default: 3)
        normalize: Whether to normalize features using Min-Max scaling
        use_feature_selection: Whether to use Mutual Information for feature selection
        n_features: Number of features to select (None = auto-select based on MI > 0)
        train_size: Proportion of data for training (default: 0.7)
        val_size: Proportion of data for validation (default: 0.15)
        test_size: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
        handle_categorical: Whether to encode categorical variables
        use_cross_validation: Whether to perform cross-validation analysis
        cv_folds: Number of folds for cross-validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_info)
    """
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Default columns to exclude
    if exclude_columns is None:
        exclude_columns = []
    
    # Separate features and target
    # Check for target column (case-insensitive)
    target_col_found = None
    for col in data.columns:
        if col.lower() == target_column.lower():
            target_col_found = col
            break
    
    if target_col_found is None:
        print(f"\n[ERROR] Target column '{target_column}' not found in dataset.")
        print(f"Available columns: {list(data.columns)}")
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(data.columns)}")
    
    # Use the found column name (might have different case)
    if target_col_found != target_column:
        print(f"[INFO] Using target column '{target_col_found}' (case variation of '{target_column}')")
        target_column = target_col_found
    
    # Get feature columns (exclude target and other specified columns)
    feature_columns = [col for col in data.columns 
                      if col not in [target_column] + exclude_columns]
    
    X = data[feature_columns].copy()
    y = data[target_column].copy()
    
    # Encode target variable
    if y.dtype == object or str(y.dtype).startswith('category'):
        y_clean = y.astype(str).str.strip().str.lower()
        class_map = {'ckd': 1, 'notckd': 0, '1': 1, '0': 0}
        y_numeric = y_clean.map(class_map)
    else:
        y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Remove invalid target labels
    valid_mask = y_numeric.isin([0, 1])
    removed = int((~valid_mask).sum())
    if removed > 0:
        print(f"\n⚠️  Detected {removed} samples with invalid target labels. Removing them.")
    
    X = X.loc[valid_mask].copy()
    y = y_numeric.loc[valid_mask].astype(int).copy()
    
    print(f"\n1. Initial Data Shape: {X.shape}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Check for missing values
    missing_count = X.isnull().sum().sum()
    print(f"\n2. Missing Values Check:")
    if missing_count > 0:
        print(f"   Total missing values: {missing_count}")
        missing_by_col = X.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        print(f"   Columns with missing values: {len(missing_cols)}")
    else:
        print(f"   No missing values found in the dataset.")
    
    # Step 0: Handle categorical variables FIRST (before imputation)
    # KNN imputation requires numeric data, so encode categoricals first
    label_encoders = {}
    if handle_categorical:
        print(f"\n2.5. Categorical Encoding (before imputation):")
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        if categorical_cols:
            print(f"   Found {len(categorical_cols)} categorical columns: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values in categorical columns
                X[col] = X[col].astype(str).replace('nan', np.nan)
                # Fit on non-null values only
                non_null_mask = X[col].notna()
                if non_null_mask.sum() > 0:
                    le.fit(X.loc[non_null_mask, col])
                    # Transform all values (NaN will remain NaN)
                    X.loc[non_null_mask, col] = le.transform(X.loc[non_null_mask, col])
                    # Convert to numeric (NaN stays as NaN)
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    label_encoders[col] = le
            print(f"   [OK] Encoded {len(categorical_cols)} categorical columns")
        else:
            print(f"   No categorical columns found (all numeric)")
    
    # Step 1: Handle Missing Values (after categorical encoding)
    if imputation_method.lower() == 'knn':
        print(f"\n3. Missing Value Imputation (KNN):")
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        X = imputer.fit_transform(X)
        print(f"   [OK] KNN imputation completed (k={knn_neighbors})")
    elif imputation_method.lower() == 'moving_avg':
        print(f"\n3. Missing Value Imputation (Moving Average):")
        imputer = MovingAverageImputer(window_size=moving_avg_window)
        X = imputer.fit_transform(X)
        print(f"   [OK] Moving Average imputation completed (window={moving_avg_window})")
    else:
        raise ValueError(f"Unknown imputation method: {imputation_method}")
    
    # Step 2: Remove Outliers using IQR (Modified: Binary columns excluded)
    if remove_outliers:
        print(f"\n4. Outlier Removal (IQR method, factor={outlier_factor}):")
        print(f"   Modified IQR: Binary columns excluded, only continuous columns filtered")
        X, y = remove_outliers_iqr(X, y, factor=outlier_factor, min_outlier_features=min_outlier_features)
        print(f"   [OK] Outlier removal completed")
    else:
        print(f"\n4. Outlier Removal: Skipped")
    
    # Step 3: Feature Normalization (Min-Max Scaling)
    scaler = None
    if normalize:
        print(f"\n5. Normalization (Min-Max Scaling):")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print(f"   [OK] Min-Max normalization completed")
    else:
        print(f"\n5. Normalization: Skipped")
    
    # Step 4: Train-Validation-Test Split (70/15/15) with Cross-Validation Setup
    print(f"\n6. Data Split (Train: {train_size*100:.0f}%, Val: {val_size*100:.0f}%, Test: {test_size*100:.0f}%) with Cross-Validation:")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"   Training set: {X_train.shape[0]} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation set: {X_val.shape[0]} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test set: {X_test.shape[0]} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"   Validation target distribution: {y_val.value_counts().to_dict()}")
    print(f"   Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Cross-Validation Setup
    cv = None
    if use_cross_validation:
        print(f"   Cross-Validation: {cv_folds}-fold stratified CV configured")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Step 5: Feature Selection using Mutual Information (ONLY on training set to avoid data leakage)
    feature_selector = None
    selected_features = feature_columns
    if use_feature_selection:
        print(f"\n7. Feature Selection (Mutual Information) - Applied on Training Set Only:")
        # Calculate MI ONLY on training data to avoid data leakage
        mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
        
        if n_features is None:
            k = sum(mi_scores > 0)
            print(f"   Selecting features with MI > 0: {k} features (based on training set)")
        else:
            k = n_features
            print(f"   Selecting top {k} features based on Mutual Information (from training set)")
        
        # Store original column names BEFORE selection
        original_columns = X_train.columns.tolist()
        
        # Create feature scores DataFrame BEFORE selection
        feature_scores = pd.DataFrame({
            'Feature': original_columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        # Create a wrapper function to pass random_state to mutual_info_classif
        # This ensures SelectKBest uses the same random_state for consistency
        def mi_score_func(X, y):
            return mutual_info_classif(X, y, random_state=random_state)
        
        selector = SelectKBest(score_func=mi_score_func, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected features using boolean mask on column names
        support_mask = selector.get_support()
        # Convert to numpy array for boolean indexing
        original_columns_array = np.array(original_columns)
        selected_features = original_columns_array[support_mask].tolist()
        removed_features = [f for f in original_columns if f not in selected_features]
        feature_selector = selector
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_val = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
        X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        print(f"\n   📊 FEATURE SELECTION DETAILS:")
        print(f"      Total features before selection: {len(original_columns)}")
        print(f"      Selected features: {len(selected_features)}")
        print(f"      Removed features: {len(removed_features)}")
        
        print(f"\n   ✅ SELECTED FEATURES ({len(selected_features)}):")
        for feature in selected_features:
            mi_score = feature_scores[feature_scores['Feature'] == feature]['MI_Score'].values[0]
            print(f"      ✓ {feature}: MI={mi_score:.4f}")
        
        if removed_features:
            print(f"\n   ❌ REMOVED FEATURES ({len(removed_features)}):")
            for feature in removed_features:
                mi_score = feature_scores[feature_scores['Feature'] == feature]['MI_Score'].values[0]
                print(f"      ✗ {feature}: MI={mi_score:.4f}")
        
        print(f"\n   Top 10 features by Mutual Information (from training set):")
        for idx, row in feature_scores.head(10).iterrows():
            status = "✓ SELECTED" if row['Feature'] in selected_features else "✗ REMOVED"
            print(f"      {row['Feature']}: {row['MI_Score']:.4f} [{status}]")
        
        print(f"   [OK] Feature selection completed (fitted on training, applied to val/test)")
        print(f"   Selected {len(selected_features)} features out of {len(feature_columns)}")
    else:
        print(f"\n7. Feature Selection: Skipped")
    
    # Final summary
    if use_cross_validation:
        print(f"\n   [OK] Data split and cross-validation setup completed")
    else:
        print(f"\n   [OK] Data split completed")
    
    # Store preprocessing information
    preprocessing_info = {
        'feature_columns': feature_columns,
        'selected_features': selected_features,
        'n_features_original': len(feature_columns),
        'n_features_selected': len(selected_features),
        'imputer': imputer if 'imputer' in locals() else None,
        'imputation_method': imputation_method,
        'scaler': scaler,
        'scaler_type': 'minmax' if normalize else None,
        'feature_selector': feature_selector,
        'feature_scores': feature_scores if use_feature_selection and 'feature_scores' in locals() else None,
        'target_column': target_column,
        'exclude_columns': exclude_columns,
        'outlier_removed': remove_outliers,
        'cv_folds': cv_folds if use_cross_validation else None,
        'cv': cv,
        'label_encoders': label_encoders if handle_categorical else {}
    }
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_info


if __name__ == "__main__":
    # Example usage
    print("CKD Data Preprocessing Module")
    print("This module provides preprocessing functions for the CKD dataset.")
    print("\nUsage:")
    print("  from preprocessing import load_arff_file, preprocess_data")
    print("  df = load_arff_file('data/chronic_kidney_disease_full.arff')")
    print("  X_train, y_train, X_val, y_val, X_test, y_test, info = preprocess_data(df)")

