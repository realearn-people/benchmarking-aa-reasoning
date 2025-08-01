from typing import Dict, List
import pandas as pd
import openpyxl

class ReportGenerator:
    """Generates an Excel report from the test results using pandas."""

    def __init__(self):
        print("Initialized Report Generator.")
        
    def _format_base_extensions(self, extensions: Dict[str, List[str]]) -> str:
        """Helper to format a list of base extensions into a readable string for a cell."""
        list_base_str = []
        for _type, ext_list in extensions.items():
            list_base_str.append(f"{_type}: {str(ext_list)}")

        if not extensions:
            return "N/A"
        
        return "\n".join(list_base_str)

    def _format_violations(self, violations: Dict[str, List[str]]) -> str:
        """Helper to format a violation dictionary into a readable string for a cell."""
        if not violations:
            return "Pass"
        
        lines = []
        for v_type, messages in violations.items():
            for msg in messages:
                # Shorten long messages for readability in the cell
                if len(msg) > 150:
                    msg = msg[:147] + "..."
                lines.append(f"[{v_type}] {msg}")
        return "\n".join(lines)

    def export_to_excel(self, filename: str = "evaluation_results.xlsx", results: Dict = None):
        """Exports the full results to a multi-sheet Excel file."""
        print(f"\nExporting results to {filename}...")
        if results is None:
            raise ValueError("No results to export. Please provide the results dictionary.")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # sample_result = results['af_name'][n][..tests..]
            for af_name, n_results in results.items():
                
                report_data = []
                for n, tests in sorted(n_results.items()):
                    
                    # Base results
                    row = {'n': n}
                    row['LLM Extensions (Base)'] = self._format_base_extensions(tests.get('base', {}).get('computed', 'N/A'))
                    row['Validity'] = self._format_violations(tests.get('validity', {}).get('violations', {}))
                    row['FP Violations (Base)'] = self._format_violations(tests.get('base', {}).get('violations', {}))
                    
                    # Metamorphic results
                    row['Isomorphism'] = self._format_violations(tests.get('isomorphism', {}).get('violations', {}))
                    row['Fundamental Consistency'] = self._format_violations(tests.get('fundamental_consistency', {}).get('violations', {}))
                    row['Modularity'] = self._format_violations(tests.get('modularity', {}).get('violations', {}))
                    
                    dd_test = tests.get('defense_dynamics', {})
                    if dd_test:
                        dd_info = dd_test.get('info', '')
                        dd_violations = self._format_violations(dd_test.get('violations', {}))
                        row['Defense Dynamics'] = f"コンテクスト「{dd_info}」\n{dd_violations}".strip()
                    else:
                        row['Defense Dynamics'] = "N/A"

                    report_data.append(row)

                if not report_data:
                    continue

                df = pd.DataFrame(report_data).set_index('n')
                
                cols_order = [
                    'LLM Extensions (Base)', 'Validity', 'FP Violations (Base)', 
                    'Isomorphism', 'Fundamental Consistency', 'Modularity', 'Defense Dynamics'
                ]
                # Filter for columns that actually exist in the dataframe
                df = df.reindex(columns=[c for c in cols_order if c in df.columns])

                # Use the AF name for the sheet, trimming if it's too long for Excel
                sheet_name = af_name[:31]
                df.to_excel(writer, sheet_name=sheet_name)
        
        print(f"Export complete. Results saved to '{filename}'.")
