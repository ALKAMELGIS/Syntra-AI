"""
Stand-in for app.training.train_worker used by test_training_trainer_offline.py
so the subprocess-supervision logic in app.training.trainer.run_training can
be exercised without ultralytics/torch installed. Speaks the exact same
one-JSON-object-per-line stdout protocol as the real worker.
"""
import json
import sys
import time
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as fh:
    job = json.load(fh)

epochs = int(job.get("epochs", 3))
cancel_flag = job.get("cancel_flag_path")
canceled = False
for epoch in range(1, epochs + 1):
    print(json.dumps({
        "type": "epoch", "epoch": epoch, "epochs": epochs,
        "box_loss": 1.0 / epoch, "precision": 0.5 + 0.01 * epoch, "recall": 0.4 + 0.01 * epoch,
        "map50": 0.3 + 0.02 * epoch, "map50_95": 0.2 + 0.01 * epoch, "elapsed_s": 0.01 * epoch,
    }), flush=True)
    if cancel_flag and Path(cancel_flag).exists():
        canceled = True
        break
    time.sleep(0.001)

print(json.dumps({
    "type": "canceled" if canceled else "done",
    "best": None if canceled else str(Path(job["project"]) / "run" / "weights" / "best.pt"),
    "results_csv": str(Path(job["project"]) / "run" / "results.csv"),
    "elapsed_s": 0.1,
}), flush=True)
