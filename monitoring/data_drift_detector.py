"""
Data drift detection module using Evidently AI.
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns
from typing import Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime


class DriftDetector:
    """Detects data drift between reference and current datasets."""
    
    def __init__(self, reference_data_path: str):
        """
        Initialize drift detector.
        
        Args:
            reference_data_path: Path to reference dataset (training data)
        """
        self.reference_data = self._load_data(reference_data_path)
        
    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from path (CSV or Parquet)."""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            # For image data, we might track embeddings or metadata
            # Here we assume we have a CSV of features/metadata
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

    def detect_drift(
        self, 
        current_data_path: str, 
        output_path: str = 'drift_reports'
    ) -> Dict[str, Any]:
        """
        Run drift detection.
        
        Args:
            current_data_path: Path to current inference data
            output_path: Directory to save reports
            
        Returns:
            Drift report dictionary
        """
        current_data = self._load_data(current_data_path)
        
        # Create report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)
        
        # Create test suite for automated checks
        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(lt=3)  # Fail if >= 3 columns drifted
        ])
        tests.run(reference_data=self.reference_data, current_data=current_data)
        
        # Save results
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_path = os.path.join(output_path, f'drift_report_{timestamp}.html')
        report.save_html(report_path)
        
        test_results = tests.as_dict()
        
        return {
            'drift_detected': not test_results['summary']['all_passed'],
            'report_path': report_path,
            'details': test_results
        }

def main():
    # Example usage
    # In a real scenario, we would extract features from images first
    # For now, this is a placeholder for the logic
    print("Drift detector initialized.")

if __name__ == '__main__':
    main()
