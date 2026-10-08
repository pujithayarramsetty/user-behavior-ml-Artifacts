import unittest
import os
import json


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(
            os.path.exists("user_behavior_model.pkl")
        )

    def test_metrics_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists("user_behavior_results.csv")
        )

    def test_accuracy(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(
            metrics["accuracy"],
            0.90
        )


if __name__ == "__main__":
    unittest.main()
