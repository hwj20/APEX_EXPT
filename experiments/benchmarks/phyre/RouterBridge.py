import json
import subprocess
import tempfile
from typing import Any, Dict

def router_bridge(data: Dict[str, Any],
                  env_name: str,
                  bridge_script: str,
                  *,
                  timeout: int = 300) -> Dict[str, Any]:
    """
    Bridge to an external 'router' implemented in another Python environment.

    Args:
        data:         The input payload to send to the router.
        env_name:     Name of the conda environment where router_bridge.py runs.
        bridge_script:Path to the router_bridge.py script.
        timeout:      Max seconds to wait for the subprocess.

    Returns:
        The JSON-decoded result from the router.

    Raises:
        RuntimeError: On any failure (non-zero exit, timeouts, JSON errors).
    """
    # 1. Create temporary files for input/output
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fin:
        json.dump(data, fin, ensure_ascii=False)
        fin.flush()
        input_path = fin.name

    output_path = input_path.replace(".json", "_out.json")

    # 2. Invoke the bridge script in the target conda env
    cmd = [
        "conda", "run", "-n", env_name,
        "python", bridge_script,
        input_path,
        output_path
    ]
    # print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Bridge script failed: {e}") from e

    # 3. Read back the output JSON
    try:
        with open(output_path, "r", encoding="utf-8") as fout:
            result = json.load(fout)
    except Exception as e:
        raise RuntimeError(f"Failed to read bridge output: {e}") from e

    return result

# ———— 使用示例 ————
if __name__ == "__main__":
    sample = {"foo": "bar", "numbers": [1,2,3]}
    out = router_bridge(
        data=sample,
        env_name="router_env",
        bridge_script="router_bridge.py"
    )
    print("Router resp：", out)
