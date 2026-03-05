# tests/conftest.py
# ------------------
# pytest configuration file.
# This file is automatically loaded by pytest before any tests run.
# Right now it's minimal — we'll add shared fixtures here later if needed.

import sys
import os

# Make sure Python can find the ingestion/ package when running tests
# from the project root directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
