"""Reproducibility and canonical hashing of the matched generator.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_generator_reproducibility -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hpop.mcmc_original import matched_synthetic_generator as msg

LENGTHS_TRAIN = (24, 32, 40, 48)
LENGTHS_HELDOUT = (24, 32)


class TestReproducibility(unittest.TestCase):
    def test_same_config_same_seed_is_byte_identical(self):
        a = msg.generate_corpus(4242, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        b = msg.generate_corpus(4242, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        self.assertEqual(msg.canonical_json(msg.corpus_to_jsonable(a)),
                         msg.canonical_json(msg.corpus_to_jsonable(b)))
        self.assertEqual(msg.corpus_hash(a), msg.corpus_hash(b))

    def test_different_seed_changes_the_hash(self):
        a = msg.generate_corpus(4242, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        b = msg.generate_corpus(4243, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        self.assertNotEqual(msg.corpus_hash(a), msg.corpus_hash(b))

    def test_save_load_roundtrip_is_byte_identical(self):
        corpus = msg.generate_corpus(4242, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        payload = msg.corpus_to_jsonable(corpus)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(msg.canonical_json(payload))
            loaded = json.loads(path.read_text())
        self.assertEqual(msg.canonical_json(loaded), msg.canonical_json(payload))
        truth = msg.truth_from_jsonable(loaded["truth"])
        np.testing.assert_array_equal(truth.u_by_skill, corpus.truth.u_by_skill)
        np.testing.assert_array_equal(truth.pi, corpus.truth.pi)
        np.testing.assert_array_equal(truth.transition, corpus.truth.transition)
        for saved, original in zip(loaded["train"], corpus.train):
            trace = msg.trace_from_jsonable(saved)
            self.assertEqual(trace, original)
        for saved, original in zip(loaded["heldout"], corpus.heldout):
            self.assertEqual(msg.trace_from_jsonable(saved), original)

    def test_canonical_serialization_carries_no_paths_or_timestamps(self):
        corpus = msg.generate_corpus(4242, LENGTHS_TRAIN, LENGTHS_HELDOUT)
        text = msg.canonical_json(msg.corpus_to_jsonable(corpus))
        for forbidden in ("/Users/", "/tmp/", "timestamp", "time\":"):
            self.assertNotIn(forbidden, text)

    def test_block_streams_are_reproducible_in_isolation(self):
        """A block regenerated from (master seed, split, trace, block) alone must
        reproduce the stored roles — the corpus is recoverable from the manifest."""
        from hpop.mcmc_original.recurrent_rfs import sample_recurrent_rfs_sequence
        corpus = msg.generate_corpus(4242, (32,), ())
        truth = corpus.truth
        trace = corpus.train[0]
        for l, (width, skill, block) in enumerate(zip(trace.widths, trace.labels,
                                                      trace.role_blocks)):
            rng = msg.block_rng(4242, "train", 0, l)
            regenerated = sample_recurrent_rfs_sequence(
                rng, width, truth.u_by_skill[skill], truth.rfs_parameters())
            self.assertEqual(tuple(regenerated), block)


if __name__ == "__main__":
    unittest.main()
