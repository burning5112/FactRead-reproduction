import unittest
from types import SimpleNamespace
import numpy as np
import torch
from factread.evaluation import score


class ScoringTest(unittest.TestCase):
    def test_prefix_excluded_and_candidates_scored_serially(self):
        seen = []
        class Model:
            def __call__(self, input_ids, **kwargs):
                self_ids = input_ids[0].tolist()
                seen.append(self_ids)
                logits = torch.zeros(1, len(self_ids), 16)
                for i in range(3, len(self_ids) - 1):
                    logits[0, i, self_ids[i + 1]] = 2
                return SimpleNamespace(logits=logits)
        bundle = SimpleNamespace(tokenizer=lambda text, **kwargs: {"input_ids": [1, 2]}, encoder=lambda k, v: (k, v), readout=None, model=Model(), kb_config=None)
        q = {"question": "q", "domain": "cc", "common_prefix_token_ids": [3, 4], "choices": [{"content_token_ids": [i + 5]} for i in range(8)], "banks": {"Near4": {"source_rows": [0, 1, 2, 3], "fact_ids": [0, 1, 2, 3]}}}
        arrays = {"cc": (np.zeros((5, 384), dtype=np.float32), np.zeros((5, 384), dtype=np.float32))}
        actual = score(bundle, q, "Near4", arrays, "cpu")
        expected = torch.log_softmax(torch.tensor([2.] + [0.] * 15), dim=0)[0].item()
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(ids[:4] == [1, 2, 3, 4] and len(ids) == 5 for ids in seen))
        self.assertTrue(all(abs(value - expected) < 1e-7 for value in actual))


if __name__ == "__main__":
    unittest.main()
