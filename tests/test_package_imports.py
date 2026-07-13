import importlib.util
import unittest


class PackageImportTests(unittest.TestCase):
    def test_release_modules_import(self):
        import minato
        import minato.binary_population
        import minato.observing
        import minato.synthetic

        self.assertTrue(hasattr(minato, "__version__"))

    def test_external_adaptation_uses_development_namespace(self):
        from minato.contrib.spdis import SpecDisent

        self.assertTrue(callable(SpecDisent))

    def test_legacy_spdis_namespace_is_absent(self):
        self.assertIsNone(importlib.util.find_spec("minato.spdis"))


if __name__ == "__main__":
    unittest.main()
