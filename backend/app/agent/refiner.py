from typing import Dict, Any, List, Optional
import re


class QueryRefiner:
    def refine(
        self,
        query: str,
        verifier_output: Dict[str, Any],
        attempt_iter: int,
        target_metrics: Optional[List[str]] = None,
        evidence_so_far: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        checks = verifier_output.get('checks', {})
        cross_check = checks.get('nu_cross', {})
        suff_check = checks.get('nu_suff', {})
        num_check = checks.get('nu_num', {})
        cross_detail = cross_check.get('detail', '')
        cross_passed = cross_check.get('passed', False)
        suff_passed = suff_check.get('passed', False)
        num_passed = num_check.get('passed', False)

        if not cross_passed:
            years = re.findall(r'20\d\d', cross_detail + query)
            year_str = ' '.join(years)
            suffix = ' ' + year_str + ' consolidated financial statements'
        elif not suff_passed:
            evidence_text = ' '.join(
                (item.get('content') or '') for item in (evidence_so_far or [])
            ).lower()
            missing_metrics = [
                metric.replace('_', ' ') for metric in (target_metrics or [])
                if metric.replace('_', ' ') not in evidence_text
            ]
            if missing_metrics:
                suffix = ' ' + ' '.join(missing_metrics)
            else:
                suffix = ' revenue gross profit expenses breakdown'
        elif not num_passed:
            suffix = ' figures numbers financial data'
        else:
            suffix = ' financial report detailed data'
        return query.rstrip() + suffix
