import unittest


class PackageImportTests(unittest.TestCase):
    def test_spdis_imports_from_installed_namespace(self):
        from minato import spdis

        self.assertTrue(hasattr(spdis, "SpecDisent"))


if __name__ == "__main__":
    unittest.main()
