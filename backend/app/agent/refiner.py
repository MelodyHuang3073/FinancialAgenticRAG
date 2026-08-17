from typing import Dict, Any
import re


class QueryRefiner:
    def refine(self, query: str, verifier_output: Dict[str, Any], attempt_iter: int) -> str:
        checks = verifier_output.get('checks', {})
        cross_detail = checks.get('nu_cross', {}).get('detail', '')
        suff_passed = checks.get('nu_suff', {}).get('passed', False)
        num_passed = checks.get('nu_num', {}).get('passed', False)
        is_chinese = bool(re.search(u'[\u4e00-\u9fff]', query))
        if attempt_iter == 1:
            if 'year' in cross_detail.lower() or u'\u5e74\u4efd' in cross_detail:
                years = re.findall(r'20\d\d', cross_detail + query)
                year_str = ' '.join(years)
                suffix = (u' ' + year_str + u' \u5408\u4f75\u8ca1\u52d9\u5831\u8868 \u8a73\u7d30\u6578\u5b57') if is_chinese else (' ' + year_str + ' consolidated financial statements')
            elif not suff_passed:
                suffix = u' \u71df\u696d\u6536\u5165 \u6bdb\u5229 \u8cbb\u7528 \u660e\u7d30' if is_chinese else ' revenue gross profit expenses breakdown'
            elif not num_passed:
                suffix = u' \u6578\u503c \u8ca1\u5831\u6578\u5b57' if is_chinese else ' figures numbers financial data'
            else:
                suffix = u' \u8ca1\u52d9\u5831\u544a \u8a73\u7d30\u6578\u64da' if is_chinese else ' financial report detailed data'
        elif attempt_iter == 2:
            suffix = u' \u7d93\u71df\u8a0e\u8ad6 \u7ba1\u7406\u5c64\u5206\u6790 \u9644\u8a3b' if is_chinese else ' MD&A management discussion notes'
        else:
            suffix = u' \u5e74\u5ea6\u5831\u544a \u5b8c\u6574\u8ca1\u52d9\u5831\u8868' if is_chinese else ' annual report complete financial statements all line items'
        return query.rstrip() + suffix
