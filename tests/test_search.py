import unittest

from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):

    def test_clean_text(self):

        text = "  Hello     World  "

        result = clean_text(text)

        self.assertEqual(
            result,
            "Hello World"
        )

    def test_empty_text(self):

        result = clean_text("   ")

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()