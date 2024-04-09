import unittest

import asynctest
from transformers import pipeline
class TestTransformerApi(unittest.TestCase):
    def test_del_file(self):
        classifier = pipeline("sentiment-analysis")
        results = classifier(
            ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
        for result in results:
            print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
