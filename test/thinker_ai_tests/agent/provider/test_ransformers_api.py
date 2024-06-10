import unittest

import asynctest
from transformers import pipeline
from datasets import load_dataset, Audio


class TestTransformerApi(unittest.TestCase):
    def test_sentiment_analysis(self):
        classifier = pipeline("sentiment-analysis")
        results = classifier(
            ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
        for result in results:
            print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    def test_speech_recognition(self):
        model = "facebook/wav2vec2-base-960h"
        speech_recognizer = pipeline("automatic-speech-recognition", model=model)
        dataset = load_dataset("PolyAI/minds14", name="zh-CN", split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
        result = speech_recognizer(dataset[:4]["audio"])
        print([d["text"] for d in result])
if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
