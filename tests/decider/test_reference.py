import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("decider_reference", Path(__file__).with_name("verify_reference.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceProvenance(unittest.TestCase):
    def test_source_hashes_reject_modified_and_missing_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_bytes(b"abc")
            expected = {"model.py": hashlib.sha256(b"abc").hexdigest()}
            self.assertEqual(module.verify_sources(root, expected), expected)
            source.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "reference checksum"):
                module.verify_sources(root, expected)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                module.verify_sources(root, expected)


if __name__ == "__main__":
    unittest.main()
