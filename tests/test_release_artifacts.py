import unittest

from scripts.check_release_artifacts import violations


class ReleaseArtifactPolicyTests(unittest.TestCase):
    def test_release_python_module_is_allowed(self):
        self.assertEqual(violations("minato/span.py"), [])

    def test_model_and_package_data_are_rejected(self):
        problems = violations("minato_astro-0.3.0/minato/models/grid/model.txt")

        self.assertIn("contains a development-only or data tree", problems)
        self.assertIn("contains non-Python package data", problems)

    def test_external_adaptation_is_rejected(self):
        self.assertTrue(violations("minato/spdis.py"))
        self.assertTrue(violations("minato/contrib/spdis.py"))
        self.assertTrue(violations("minato_astro-0.3.0/contrib/adaptation.py"))

    def test_development_record_is_rejected(self):
        self.assertIn(
            "contains a development record",
            violations("minato_astro-0.3.0/RELEASE_PREPARATION_PLAN.md"),
        )


if __name__ == "__main__":
    unittest.main()
