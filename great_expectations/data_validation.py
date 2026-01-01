"""
Great Expectations Data Validation
Data quality and schema validation for ML pipelines
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    details: Dict
    severity: str = "error"  # error, warning, info


@dataclass
class DataValidationReport:
    """Complete validation report for a dataset"""
    dataset_name: str
    validation_time: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    overall_success: bool
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "validation_time": self.validation_time.isoformat(),
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "overall_success": self.overall_success,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in self.results
            ]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DataExpectations:
    """
    Great Expectations-style data validation for Telco Churn dataset.
    
    Implements common data quality checks:
    - Schema validation
    - Missing value checks
    - Value range checks
    - Uniqueness checks
    - Statistical checks
    - Referential integrity
    """
    
    def __init__(self, dataset_name: str = "telco_churn"):
        self.dataset_name = dataset_name
        self.results: List[ValidationResult] = []
    
    def _add_result(
        self,
        check_name: str,
        passed: bool,
        details: Dict,
        severity: str = "error"
    ):
        """Add a validation result"""
        self.results.append(ValidationResult(
            check_name=check_name,
            passed=passed,
            details=details,
            severity=severity
        ))
    
    # ═══════════════════════════════════════════════════════════════
    # Schema Expectations
    # ═══════════════════════════════════════════════════════════════
    
    def expect_column_to_exist(self, df: pd.DataFrame, column: str) -> bool:
        """Expect a column to exist in the dataframe"""
        exists = column in df.columns
        self._add_result(
            f"expect_column_to_exist({column})",
            passed=exists,
            details={"column": column, "found": exists}
        )
        return exists
    
    def expect_table_columns_to_match_set(
        self,
        df: pd.DataFrame,
        columns: List[str],
        exact_match: bool = False
    ) -> bool:
        """Expect dataframe to have specified columns"""
        actual_cols = set(df.columns)
        expected_cols = set(columns)
        
        if exact_match:
            passed = actual_cols == expected_cols
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols
        else:
            passed = expected_cols.issubset(actual_cols)
            missing = expected_cols - actual_cols
            extra = set()
        
        self._add_result(
            "expect_table_columns_to_match_set",
            passed=passed,
            details={
                "expected": list(expected_cols),
                "missing": list(missing),
                "extra": list(extra)
            }
        )
        return passed
    
    def expect_column_values_to_be_of_type(
        self,
        df: pd.DataFrame,
        column: str,
        expected_type: str
    ) -> bool:
        """Expect column values to be of a specific type"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_values_to_be_of_type({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        actual_type = str(df[column].dtype)
        
        # Type mapping
        type_map = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "float16"],
            "string": ["object", "string"],
            "bool": ["bool"],
            "datetime": ["datetime64[ns]"]
        }
        
        passed = actual_type in type_map.get(expected_type, [expected_type])
        
        self._add_result(
            f"expect_column_values_to_be_of_type({column})",
            passed=passed,
            details={
                "column": column,
                "expected_type": expected_type,
                "actual_type": actual_type
            }
        )
        return passed
    
    # ═══════════════════════════════════════════════════════════════
    # Null/Missing Expectations
    # ═══════════════════════════════════════════════════════════════
    
    def expect_column_values_to_not_be_null(
        self,
        df: pd.DataFrame,
        column: str,
        mostly: float = 1.0
    ) -> bool:
        """Expect column values to not be null"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_values_to_not_be_null({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        null_count = df[column].isnull().sum()
        total_count = len(df)
        non_null_ratio = 1 - (null_count / total_count) if total_count > 0 else 0
        
        passed = non_null_ratio >= mostly
        
        self._add_result(
            f"expect_column_values_to_not_be_null({column})",
            passed=passed,
            details={
                "column": column,
                "null_count": int(null_count),
                "total_count": total_count,
                "non_null_ratio": non_null_ratio,
                "threshold": mostly
            },
            severity="warning" if null_count > 0 and passed else "error"
        )
        return passed
    
    # ═══════════════════════════════════════════════════════════════
    # Value Range Expectations
    # ═══════════════════════════════════════════════════════════════
    
    def expect_column_values_to_be_between(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        mostly: float = 1.0
    ) -> bool:
        """Expect column values to be within a range"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_values_to_be_between({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        values = df[column].dropna()
        
        in_range = pd.Series([True] * len(values))
        if min_value is not None:
            in_range &= values >= min_value
        if max_value is not None:
            in_range &= values <= max_value
        
        ratio = in_range.mean() if len(values) > 0 else 0
        passed = ratio >= mostly
        
        out_of_range = (~in_range).sum()
        
        self._add_result(
            f"expect_column_values_to_be_between({column})",
            passed=passed,
            details={
                "column": column,
                "min_value": min_value,
                "max_value": max_value,
                "actual_min": float(values.min()) if len(values) > 0 else None,
                "actual_max": float(values.max()) if len(values) > 0 else None,
                "out_of_range_count": int(out_of_range),
                "in_range_ratio": ratio,
                "threshold": mostly
            }
        )
        return passed
    
    def expect_column_values_to_be_in_set(
        self,
        df: pd.DataFrame,
        column: str,
        value_set: List[Any],
        mostly: float = 1.0
    ) -> bool:
        """Expect column values to be in a specific set"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_values_to_be_in_set({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        values = df[column].dropna()
        in_set = values.isin(value_set)
        ratio = in_set.mean() if len(values) > 0 else 0
        passed = ratio >= mostly
        
        unexpected = values[~in_set].unique().tolist()[:10]  # First 10 unexpected
        
        self._add_result(
            f"expect_column_values_to_be_in_set({column})",
            passed=passed,
            details={
                "column": column,
                "expected_set": value_set,
                "unexpected_values": unexpected,
                "in_set_ratio": ratio,
                "threshold": mostly
            }
        )
        return passed
    
    # ═══════════════════════════════════════════════════════════════
    # Uniqueness Expectations
    # ═══════════════════════════════════════════════════════════════
    
    def expect_column_values_to_be_unique(
        self,
        df: pd.DataFrame,
        column: str,
        mostly: float = 1.0
    ) -> bool:
        """Expect column values to be unique"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_values_to_be_unique({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        values = df[column].dropna()
        unique_ratio = len(values.unique()) / len(values) if len(values) > 0 else 0
        passed = unique_ratio >= mostly
        
        duplicate_count = len(values) - len(values.unique())
        
        self._add_result(
            f"expect_column_values_to_be_unique({column})",
            passed=passed,
            details={
                "column": column,
                "unique_count": len(values.unique()),
                "total_count": len(values),
                "duplicate_count": duplicate_count,
                "unique_ratio": unique_ratio,
                "threshold": mostly
            }
        )
        return passed
    
    # ═══════════════════════════════════════════════════════════════
    # Statistical Expectations
    # ═══════════════════════════════════════════════════════════════
    
    def expect_column_mean_to_be_between(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """Expect column mean to be within a range"""
        if column not in df.columns:
            self._add_result(
                f"expect_column_mean_to_be_between({column})",
                passed=False,
                details={"error": "column not found"}
            )
            return False
        
        mean_value = df[column].mean()
        
        passed = True
        if min_value is not None and mean_value < min_value:
            passed = False
        if max_value is not None and mean_value > max_value:
            passed = False
        
        self._add_result(
            f"expect_column_mean_to_be_between({column})",
            passed=passed,
            details={
                "column": column,
                "mean": float(mean_value),
                "min_value": min_value,
                "max_value": max_value
            }
        )
        return passed
    
    def expect_table_row_count_to_be_between(
        self,
        df: pd.DataFrame,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> bool:
        """Expect table row count to be within a range"""
        row_count = len(df)
        
        passed = True
        if min_value is not None and row_count < min_value:
            passed = False
        if max_value is not None and row_count > max_value:
            passed = False
        
        self._add_result(
            "expect_table_row_count_to_be_between",
            passed=passed,
            details={
                "row_count": row_count,
                "min_value": min_value,
                "max_value": max_value
            }
        )
        return passed
    
    # ═══════════════════════════════════════════════════════════════
    # Generate Report
    # ═══════════════════════════════════════════════════════════════
    
    def get_validation_report(self) -> DataValidationReport:
        """Generate complete validation report"""
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = sum(1 for r in self.results if not r.passed)
        
        # Consider overall success based on error-severity failures only
        error_failures = sum(
            1 for r in self.results 
            if not r.passed and r.severity == "error"
        )
        
        return DataValidationReport(
            dataset_name=self.dataset_name,
            validation_time=datetime.now(),
            total_checks=len(self.results),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            results=self.results,
            overall_success=error_failures == 0
        )
    
    def clear_results(self):
        """Clear all validation results"""
        self.results = []


class TelcoChurnExpectations(DataExpectations):
    """
    Pre-configured expectations for Telco Customer Churn dataset.
    """
    
    REQUIRED_COLUMNS = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    GENDER_VALUES = ['Male', 'Female']
    YES_NO_VALUES = ['Yes', 'No']
    CONTRACT_VALUES = ['Month-to-month', 'One year', 'Two year']
    INTERNET_VALUES = ['DSL', 'Fiber optic', 'No']
    PAYMENT_VALUES = [
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ]
    
    def __init__(self):
        super().__init__(dataset_name="telco_churn")
    
    def validate_schema(self, df: pd.DataFrame):
        """Validate schema expectations"""
        self.expect_table_columns_to_match_set(df, self.REQUIRED_COLUMNS)
        
        # Type checks
        self.expect_column_values_to_be_of_type(df, 'tenure', 'int')
        self.expect_column_values_to_be_of_type(df, 'MonthlyCharges', 'float')
        self.expect_column_values_to_be_of_type(df, 'SeniorCitizen', 'int')
    
    def validate_values(self, df: pd.DataFrame):
        """Validate value constraints"""
        # Categorical value sets
        self.expect_column_values_to_be_in_set(df, 'gender', self.GENDER_VALUES)
        self.expect_column_values_to_be_in_set(df, 'Partner', self.YES_NO_VALUES)
        self.expect_column_values_to_be_in_set(df, 'Dependents', self.YES_NO_VALUES)
        self.expect_column_values_to_be_in_set(df, 'PhoneService', self.YES_NO_VALUES)
        self.expect_column_values_to_be_in_set(df, 'Contract', self.CONTRACT_VALUES)
        self.expect_column_values_to_be_in_set(df, 'InternetService', self.INTERNET_VALUES)
        self.expect_column_values_to_be_in_set(df, 'PaymentMethod', self.PAYMENT_VALUES)
        self.expect_column_values_to_be_in_set(df, 'Churn', self.YES_NO_VALUES)
        
        # Numeric ranges
        self.expect_column_values_to_be_between(df, 'tenure', min_value=0, max_value=100)
        self.expect_column_values_to_be_between(df, 'MonthlyCharges', min_value=0, max_value=500)
        self.expect_column_values_to_be_between(df, 'SeniorCitizen', min_value=0, max_value=1)
    
    def validate_completeness(self, df: pd.DataFrame):
        """Validate data completeness"""
        # Critical columns should not be null
        self.expect_column_values_to_not_be_null(df, 'customerID')
        self.expect_column_values_to_not_be_null(df, 'Churn')
        self.expect_column_values_to_not_be_null(df, 'tenure')
        self.expect_column_values_to_not_be_null(df, 'MonthlyCharges')
        
        # Allow some missing in TotalCharges
        self.expect_column_values_to_not_be_null(df, 'TotalCharges', mostly=0.99)
    
    def validate_uniqueness(self, df: pd.DataFrame):
        """Validate uniqueness constraints"""
        self.expect_column_values_to_be_unique(df, 'customerID')
    
    def validate_statistics(self, df: pd.DataFrame):
        """Validate statistical properties"""
        self.expect_table_row_count_to_be_between(df, min_value=1000)
        self.expect_column_mean_to_be_between(df, 'tenure', min_value=10, max_value=50)
        self.expect_column_mean_to_be_between(df, 'MonthlyCharges', min_value=30, max_value=100)
    
    def run_full_validation(self, df: pd.DataFrame) -> DataValidationReport:
        """Run all validation checks"""
        self.clear_results()
        
        self.validate_schema(df)
        self.validate_values(df)
        self.validate_completeness(df)
        self.validate_uniqueness(df)
        self.validate_statistics(df)
        
        return self.get_validation_report()


# ═══════════════════════════════════════════════════════════════
# Validation Runner
# ═══════════════════════════════════════════════════════════════

def validate_telco_dataset(df: pd.DataFrame) -> DataValidationReport:
    """
    Convenience function to validate Telco Churn dataset.
    
    Usage:
        df = pd.read_csv("data.csv")
        report = validate_telco_dataset(df)
        
        if report.overall_success:
            print("Validation passed!")
        else:
            print("Validation failed!")
            print(report.to_json())
    """
    expectations = TelcoChurnExpectations()
    return expectations.run_full_validation(df)


if __name__ == "__main__":
    # Example usage
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    report = validate_telco_dataset(df)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {report.dataset_name}")
    print(f"Validation Time: {report.validation_time}")
    print(f"{'='*60}")
    print(f"Total Checks: {report.total_checks}")
    print(f"Passed: {report.passed_checks}")
    print(f"Failed: {report.failed_checks}")
    print(f"Overall Success: {report.overall_success}")
    print(f"{'='*60}")
    
    if not report.overall_success:
        print("\nFailed Checks:")
        for result in report.results:
            if not result.passed:
                print(f"  ❌ {result.check_name}")
                print(f"     Details: {result.details}")
