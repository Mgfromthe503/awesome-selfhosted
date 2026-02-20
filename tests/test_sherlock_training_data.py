import unittest

from sherlock_training_data import (
    EmojiParser,
    anomaly_detection,
    detect_anomalies_in_spiritual_energy,
    interpret_emoji,
    BioinformaticsModule,
    FuturisticQuantumKeyGenerator,
)


class TestSherlockTrainingData(unittest.TestCase):
    def test_emoji_parser_known_and_unknown(self):
        parser = EmojiParser()
        self.assertIn("brain", parser.parse("🧠"))
        self.assertEqual("Unknown Emoji", parser.parse("❓"))

    def test_interpret_emoji_combines_visual_and_symbolic(self):
        interpretation = interpret_emoji("🔥")
        self.assertEqual("🔥", interpretation["visual"]["emoji"])
        self.assertEqual("Transformation and energy", interpretation["symbolic"])
        self.assertIn("🔥 => Transformation and energy", interpretation["combined"])

    def test_anomaly_detection_flags_outlier(self):
        data = [10, 11, 10, 12, 9, 100]
        flags = anomaly_detection(data)
        self.assertEqual(len(data), len(flags))
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))

    def test_detect_anomalies_in_spiritual_energy_summary(self):
        result = detect_anomalies_in_spiritual_energy([1, 2, 1, 2, 20, 2])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["anomalies"], [4])

    def test_bioinformatics_module_parses_fasta(self):
        fasta = ">seq1\nATCG\n>seq2\nGGTA"
        module = BioinformaticsModule()
        parsed = module.analyze_sequence(fasta)
        self.assertEqual(parsed[0]["id"], "seq1")
        self.assertEqual(parsed[0]["sequence"], "ATCG")
        self.assertEqual(parsed[1]["id"], "seq2")
        self.assertEqual(parsed[1]["sequence"], "GGTA")

    def test_quantum_key_generator_returns_expected_marker(self):
        generator = FuturisticQuantumKeyGenerator(num_qubits=5)
        key = generator.generate_quantum_key()
        self.assertEqual(len(key), 5)
        self.assertEqual(generator.get_error_corrected_key(), "Error-corrected quantum key")


if __name__ == "__main__":
    unittest.main()
