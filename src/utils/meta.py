import datetime
import platform
import subprocess
import onnxruntime as ort


def get_git_commit() -> str:
    """Return the current git commit SHA, or 'unknown' if not a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return commit
    except Exception:
        return "unversioned"


def get_hardware_id() -> str:
    """Return a descriptive string for current CPU / platform."""
    return f"{platform.system()}_{platform.machine()}_{platform.processor() or 'generic'}"


def get_ort_version() -> str:
    """Return the installed ONNX Runtime version."""
    return getattr(ort, "__version__", "unknown")


def make_metadata(model_id: str, variant_id: str) -> dict:
    """
    Construct standard metadata dictionary required for every evaluation result.
    model_id: 'r34' or 'mbv3s'
    variant_id: 'fp32', 'int8-minmax', 'int8-entropy', or 'mix-k{N}'
    """
    return {
        "model": model_id,
        "variant": variant_id,
        "git_commit": get_git_commit(),
        "ort_version": get_ort_version(),
        "hardware_id": get_hardware_id(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
