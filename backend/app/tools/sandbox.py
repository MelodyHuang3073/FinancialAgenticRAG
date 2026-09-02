import ast
import math
import sys
import io
import traceback
from typing import Dict, Any, Tuple

class SafeSandboxError(Exception):
    pass

class ASTSecurityChecker(ast.NodeVisitor):
    BLOCKED_NODES = {
        ast.Import, ast.ImportFrom,
    }
    ALLOWED_IMPORTS = {'math', 'numpy', 'np', 'datetime'}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                raise SafeSandboxError(f"Security Alert: Import of '{alias.name}' is prohibited in PoT sandbox.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module not in self.ALLOWED_IMPORTS:
            raise SafeSandboxError(f"Security Alert: Import from '{node.module}' is prohibited in PoT sandbox.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in {'eval', 'exec', 'open', '__import__', 'compile', 'getattr', 'setattr', 'delattr'}:
                raise SafeSandboxError(f"Security Alert: Call to prohibited builtin function '{node.func.id}'.")
        self.generic_visit(node)


def execute_pot_code(code_str: str, timeout_sec: float = 5.0) -> Tuple[bool, Any, str, Dict[str, Any]]:
    """
    Executes Program-of-Thought Python code safely.
    Returns: (success: bool, result_value: Any, stdout_or_error: str, locals: Dict[str, Any])
    `locals` is the full sandbox local namespace after execution (empty dict
    on failure) — lets a caller pull out named intermediate variables (e.g.
    a multi-year comparison's per-year values) beyond just the single
    `result` value, without widening the sandbox's own contract further.
    """
    # Clean code block backticks if present
    code_clean = code_str.strip()
    if code_clean.startswith("```"):
        lines = code_clean.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code_clean = "\n".join(lines).strip()

    try:
        parsed_ast = ast.parse(code_clean)
        checker = ASTSecurityChecker()
        checker.visit(parsed_ast)
    except SyntaxError as se:
        return False, None, f"SyntaxError in PoT code: {se}", {}
    except SafeSandboxError as sse:
        return False, None, str(sse), {}
    except Exception as e:
        return False, None, f"AST Parsing Error: {str(e)}", {}

    # Custom helper financial functions inside sandbox
    def cagr(v_begin: float, v_end: float, n_years: float) -> float:
        if v_begin <= 0 or n_years <= 0:
            return 0.0
        return ((v_end / v_begin) ** (1.0 / n_years) - 1.0) * 100.0

    def yoy(v_old: float, v_new: float) -> float:
        if v_old == 0:
            return 0.0
        return ((v_new - v_old) / abs(v_old)) * 100.0

    def margin(numerator: float, revenue: float) -> float:
        if revenue == 0:
            return 0.0
        return (numerator / revenue) * 100.0

    # Restricted Globals Scope
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "round": round, "int": int, "float": float, "str": str,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "range": range, "zip": zip, "enumerate": enumerate,
            "bool": bool, "print": print
        },
        "math": math,
        "cagr": cagr,
        "yoy": yoy,
        "margin": margin,
    }

    try:
        import numpy as np
        safe_globals["np"] = np
        safe_globals["numpy"] = np
    except ImportError:
        pass

    safe_locals: Dict[str, Any] = {}

    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        exec(code_clean, safe_globals, safe_locals)
        output_str = redirected_output.getvalue().strip()

        # Find final result: variable named 'result', 'ans', 'answer', or last mutated variable
        res_val = None
        for res_key in ['result', 'answer', 'ans', 'cagr_val', 'growth_rate', 'final_answer']:
            if res_key in safe_locals:
                res_val = safe_locals[res_key]
                break

        if res_val is None and safe_locals:
            # Pick last variable
            last_key = list(safe_locals.keys())[-1]
            res_val = safe_locals[last_key]

        return True, res_val, output_str or (f"result = {res_val}" if res_val is not None else "Executed successfully."), safe_locals
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        return False, None, err_msg, {}
    finally:
        sys.stdout = old_stdout
