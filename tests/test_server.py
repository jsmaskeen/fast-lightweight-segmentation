from http.server import HTTPServer
import threading
import time
import urllib.request

from server.app import WebViewerHandler


def test_server_serves_static_assets_and_model():
    # Setup test server on an ephemeral port
    server = HTTPServer(("127.0.0.1", 0), WebViewerHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test index.html
        with urllib.request.urlopen(f"{base_url}/") as response:
            assert response.status == 200
            content = response.read().decode("utf-8")
            assert "BUSI POCUS Tumor Vision" in content
            assert "WebGPU" in content
            headers = response.headers
            assert headers.get("Cross-Origin-Opener-Policy") == "same-origin"
            assert headers.get("Cross-Origin-Embedder-Policy") == "require-corp"

        # 2. Test model loading
        with urllib.request.urlopen(f"{base_url}/models/mbv3s_distilled_int8_minmax.onnx") as response:
            assert response.status == 200
            assert response.headers.get("Content-Type") == "application/octet-stream"
            # Verify file size matches smallest model ~2.5MB
            assert int(response.headers.get("Content-Length")) > 2_000_000

        # 3. Test static JS
        with urllib.request.urlopen(f"{base_url}/js/app.js") as response:
            assert response.status == 200
            assert "application/javascript" in response.headers.get("Content-Type")

    finally:
        server.shutdown()
        server.server_close()
