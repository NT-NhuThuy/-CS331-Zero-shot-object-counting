import unittest

from zoc import __version__


class ArchivePackageTest(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(__version__, "0.1.0")
