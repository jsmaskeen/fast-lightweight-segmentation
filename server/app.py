import argparse
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class WebViewerHandler(SimpleHTTPRequestHandler):
    """
    Custom HTTP Request Handler for the Edge WebViewer.
    Ensures:
    - Modern CORS and Cross-Origin Isolation headers for WebAssembly multi-threading & WebGPU.
    - Correct MIME types for .wasm, .mjs, and .onnx model weights.
    - Serves static assets seamlessly from the server/static directory.
    """

    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            directory = str(Path(__file__).parent / "static")
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        # Enable Cross-Origin Isolation for SharedArrayBuffer & WebAssembly SIMD
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def guess_type(self, path):
        # Ensure correct MIME types for WebAssembly, modules, and models
        p = str(path).lower()
        if p.endswith(".wasm"):
            return "application/wasm"
        elif p.endswith(".mjs") or p.endswith(".js"):
            return "application/javascript"
        elif p.endswith(".onnx"):
            return "application/octet-stream"
        elif p.endswith(".json"):
            return "application/json"
        return super().guess_type(path)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[WebViewer] {self.address_string()} - {format % args}\n")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server_dir = Path(__file__).parent / "static"
    if not server_dir.exists():
        server_dir.mkdir(parents=True, exist_ok=True)

    server_address = (host, port)
    httpd = HTTPServer(server_address, lambda *args, **kwargs: WebViewerHandler(*args, directory=str(server_dir), **kwargs))

    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"\n==================================================================")
    print(f" BUSI Real-Time WebViewer (WebGPU & WebAssembly)")
    print(f" Smallest Distilled-Quantized Model (mbv3s INT8 QDQ - 2.50 MB)")
    print(f"==================================================================")
    print(f" Server running at: http://{display_host}:{port}/")
    print(f" Static directory:  {server_dir.resolve()}")
    print(f" Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebViewer] Server shutting down cleanly...")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="BUSI POCUS Distilled Model WebViewer Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
