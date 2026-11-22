"""
Syntra AI on a Hugging Face Space: the FastAPI engine and the built interface,
served by one uvicorn process on port 7860.

Space settings (Settings > Variables and secrets):
  SYNTRA_OWNER_EMAIL   the email whose sign-up becomes the owner (required: a Space has no local screen)
  SYNTRA_OWNER_PASSWORD  secret: with it the owner account is made again at every start
                       while the disk is empty, so a restart doesn't ask for it again
  SYNTRA_PUBLIC_URL    https://alkamelgis-syntra-ai.hf.space (OAuth and Stripe return addresses)
  SENTINELHUB_INSTANCE_ID, … as in backend/.env.example

Storage: a Space's disk is reset when it restarts or is rebuilt. Accounts, credits,
images and results live in SYNTRA_DATA_DIR: /data when a storage bucket is mounted
there, otherwise ./data (lost on restart).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "backend"))
os.environ.setdefault("SYNTRA_DATA_DIR", "/data" if Path("/data").is_dir() and os.access("/data", os.W_OK) else str(HERE / "data"))
os.environ.setdefault("SYNTRA_STATIC_DIR", str(HERE / "static"))
os.environ.setdefault("SYNTRA_ALLOW_SIGNUP", "1")

try:                                   # ZeroGPU hardware: import spaces before torch, and declare a GPU function
    import spaces

    @spaces.GPU(duration=15)
    def gpu_check() -> bool:
        """ZeroGPU stops a Space that declares no GPU function; detection itself runs on CPU here."""
        import torch
        return bool(torch.cuda.is_available())
except ImportError:                    # CPU hardware or a local run
    spaces = None

from app.main import app  # noqa: E402

try:                                   # a small Gradio page at /gradio: the SDK this Space runs on
    import gradio as gr

    with gr.Blocks(title="Syntra AI status") as status:
        gr.Markdown("Syntra AI is running. Open the app at the Space's own address.")
    app = gr.mount_gradio_app(app, status, path="/gradio")
except Exception:  # noqa: BLE001 - the app works without it
    pass

if __name__ == "__main__":
    import uvicorn

    # The Space routes to 7860 (GRADIO_SERVER_PORT); PORT may name a port the runtime itself holds.
    port = int(os.getenv("GRADIO_SERVER_PORT") or 7860)
    print(f"Syntra on port {port} (PORT={os.getenv('PORT')})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
