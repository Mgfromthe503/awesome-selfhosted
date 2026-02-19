import unittest

from sherlock_training_data import (
    DNA_System,
    EMOJI_MAP,
    PRINCIPLE_ASSET,
    PRINCIPLES,
    Sherlock,
    book_create,
    parse_principles,
    principle_create,
)


class SherlockFrameworkTests(unittest.TestCase):
    def test_principles_and_assets_have_expected_size(self):
        self.assertEqual(len(PRINCIPLES), 12)
        self.assertEqual(len(PRINCIPLE_ASSET), 12)
        self.assertEqual(PRINCIPLE_ASSET[0]["Emoji"], "🧠")

    def test_parse_principles_returns_deduplicated_ids(self):
        parsed = parse_principles("🧠💭⚙️⚙️🌌x")
        self.assertEqual(parsed, [1, 6, 11])

    def test_emoji_map_contains_aliases(self):
        self.assertEqual(EMOJI_MAP["💭"], 1)
        self.assertEqual(EMOJI_MAP["⊕"], 8)

    def test_dna_integrate_returns_new_object(self):
        dna = DNA_System()
        p = principle_create(1, "Mentalism")
        updated = dna.integrate_principle(p)

        self.assertEqual(len(dna.principles), 0)
        self.assertEqual(len(updated.principles), 1)
        self.assertEqual(updated.principles[0]["Name"], "Mentalism")

    def test_sherlock_behaviors(self):
        sher = Sherlock()
        findings = sher.investigate("Some raw data")
        self.assertIn("Findings", findings)
        self.assertEqual(findings["Input"], "Some raw data")

        truth = sher.seek_truth("signal")
        self.assertEqual(truth["Echo"], "signal")
        self.assertEqual(truth["PrinciplesUsed"], 0)

        book = book_create("T", "A", "C")
        result = sher.train_on_book(book)
        self.assertIs(result, sher)
        self.assertEqual(len(sher.knowledge_base), 1)


if __name__ == "__main__":
    unittest.main()
