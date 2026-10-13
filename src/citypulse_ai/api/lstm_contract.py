"""Trusted LSTM identity shared across processes; no TensorFlow import."""

from pathlib import Path

MODEL_NAME = "tensorflow-lstm"
MODEL_VERSION = "run-20261012T141848308777Z:cfb8e2c68a87"
WEIGHTS_SHA256 = "cfb8e2c68a87ebd383836b76d7f39a15f93db0d31f6a3a728ee12f84b2d7bc98"
RECORD_SHA256 = "4f4ff0166cf7a6976bac95f3293a201ddf1328bd7e0941f79a688f8a19dfaeb3"
ARTIFACT_DIRECTORY = (Path(__file__).resolve().parents[3] / "outputs" /
    "phase3-tensorflow-lstm-jan-feb-2024" / "run-20261012T141848308777Z")
