"""Safe Python code execution via subprocess with timeout."""
import os
import subprocess
import tempfile


def execute_python(code: str, timeout: int = 60) -> dict:
    """Execute Python code in an isolated subprocess. Returns stdout, stderr, success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Execution timed out after {timeout}s", "returncode": -1, "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
