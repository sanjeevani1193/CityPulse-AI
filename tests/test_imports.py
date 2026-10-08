"""Verify the scaffold imports without data, services, or ML dependencies."""

import importlib
from pathlib import Path
import unittest


class PackageSmokeTests(unittest.TestCase):
    def test_package_imports_from_src(self):
        package = importlib.import_module("citypulse_ai")
        expected = Path(__file__).resolve().parents[1] / "src" / "citypulse_ai"
        self.assertEqual(Path(package.__file__).resolve().parent, expected)

    def test_all_planned_packages_are_importable(self):
        for name in (
            "etl", "features", "forecasting", "evaluation", "experiments", "sqlpilot"
        ):
            with self.subTest(package=name):
                module = importlib.import_module(f"citypulse_ai.{name}")
                self.assertTrue(module.__doc__)
                self.assertIsNotNone(module.__spec__.submodule_search_locations)
