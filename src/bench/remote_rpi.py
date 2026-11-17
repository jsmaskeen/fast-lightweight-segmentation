import argparse
import os
from pathlib import Path
import paramiko
import yaml


def load_rpi_config(config_path: str = "configs/bench.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    rpi_cfg = cfg.get("rpi", {})

    # Overwrite from environment variables if set in .env
    host = os.environ.get("RPI_HOST", rpi_cfg.get("host", "192.168.1.100"))
    port = int(os.environ.get("RPI_PORT", rpi_cfg.get("port", 22)))
    user = os.environ.get("RPI_USER", rpi_cfg.get("user", "pi"))
    password = os.environ.get("RPI_PASSWORD", rpi_cfg.get("password", "raspberry"))
    remote_dir = os.environ.get("RPI_REMOTE_DIR", rpi_cfg.get("remote_dir", "~/busi-int8"))

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "remote_dir": remote_dir,
    }


def get_ssh_client(cfg: dict) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to Raspberry Pi at {cfg['user']}@{cfg['host']}:{cfg['port']}...")
    client.connect(
        hostname=cfg["host"],
        port=cfg["port"],
        username=cfg["user"],
        password=cfg["password"],
        timeout=10,
    )
    return client


def setup_rpi_environment(client: paramiko.SSHClient, remote_dir: str):
    """
    Phase 7:
    - Sets CPU governor to performance.
    - Records hardware/OS/kernel/ORT versions.
    - Creates remote workspace directory.
    """
    print("Setting CPU governor to performance on Raspberry Pi...")
    commands = [
        f"mkdir -p {remote_dir}/models/fp32 {remote_dir}/models/int8 {remote_dir}/models/mixed {remote_dir}/results {remote_dir}/src",
        "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || sudo cpufreq-set -g performance",
        "uname -a",
        "cat /proc/cpuinfo | grep 'model name' | head -n 1",
        "vcgencmd measure_temp 2>/dev/null || true",
        "vcgencmd get_throttled 2>/dev/null || true",
        "python3 -c 'import onnxruntime; print(\"ORT Version:\", onnxruntime.__version__)' 2>/dev/null || true",
    ]

    env_output = {}
    for cmd in commands:
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        print(f"  $ {cmd} -> {out or err}")
        env_output[cmd] = out

    # Append to docs/env.md
    env_doc = Path("docs/env.md")
    cpu_cmd = "cat /proc/cpuinfo | grep 'model name' | head -n 1"
    ort_cmd = 'python3 -c \'import onnxruntime; print("ORT Version:", onnxruntime.__version__)\' 2>/dev/null || true'
    with open(env_doc, "a") as f:
        f.write("\n## Raspberry Pi 4 Edge Target (Connected Details)\n")
        f.write(f"- **Kernel**: `{env_output.get('uname -a', 'N/A')}`\n")
        f.write(f"- **CPU Model**: `{env_output.get(cpu_cmd, 'N/A')}`\n")
        f.write(f"- **Temperature**: `{env_output.get('vcgencmd measure_temp 2>/dev/null || true', 'N/A')}`\n")
        f.write(f"- **Throttle Code**: `{env_output.get('vcgencmd get_throttled 2>/dev/null || true', 'N/A')}`\n")
        f.write(f"- **Remote ORT**: `{env_output.get(ort_cmd, 'N/A')}`\n")

    print(f"Environment documentation updated in {env_doc.resolve()}")


def sync_models_and_code(client: paramiko.SSHClient, remote_dir: str):
    """Sync ONNX models and benchmark scripts to Raspberry Pi via SFTP."""
    print("Syncing ONNX models and benchmark code to Raspberry Pi...")
    sftp = client.open_sftp()

    def upload_dir(local_path, remote_path):
        for item in Path(local_path).rglob("*"):
            if item.is_file():
                rel = item.relative_to(local_path)
                r_file = f"{remote_path}/{rel.as_posix()}"
                r_parent = str(Path(r_file).parent).replace("\\", "/")
                # Ensure remote parent directory exists
                client.exec_command(f"mkdir -p {r_parent}")
                print(f"  Uploading {item.name} -> {r_file}")
                sftp.put(str(item), r_file)

    upload_dir("models", f"{remote_dir}/models")
    upload_dir("src", f"{remote_dir}/src")
    sftp.close()
    print("Sync complete.")


def run_remote_command(client: paramiko.SSHClient, cmd: str):
    """Run command on RPi4 and stream output."""
    print(f"\n[RPi4 Remote Exec] {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 Remote Runner over SSH")
    parser.add_argument("--test-connection", action="store_true", help="Test SSH connection to Raspberry Pi")
    parser.add_argument("--setup-env", action="store_true", help="Set performance governor and record env.md")
    parser.add_argument("--sync", action="store_true", help="Sync models and scripts to RPi")
    parser.add_argument("--bench", action="store_true", help="Run benchmark harness on RPi")
    parser.add_argument("--cine", action="store_true", help="Run cine loop on RPi")
    args = parser.parse_args()

    cfg = load_rpi_config()

    try:
        client = get_ssh_client(cfg)
        print("SSH Connection SUCCESSFUL!")
    except Exception as e:
        print(f"SSH Connection to Raspberry Pi ({cfg['user']}@{cfg['host']}) failed: {e}")
        print("Tip: Ensure Raspberry Pi is powered on, SSH is enabled, and IP/credentials in configs/bench.yaml or .env are set.")
        return

    if args.setup_env or args.test_connection:
        setup_rpi_environment(client, cfg["remote_dir"])

    if args.sync:
        sync_models_and_code(client, cfg["remote_dir"])

    if args.bench:
        run_remote_command(client, f"cd {cfg['remote_dir']} && python3 -m src.bench.harness")

    if args.cine:
        run_remote_command(client, f"cd {cfg['remote_dir']} && python3 -m src.cine.run_cine")

    client.close()


if __name__ == "__main__":
    main()
