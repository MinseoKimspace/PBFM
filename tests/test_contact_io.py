import tempfile
import unittest
from pathlib import Path

import torch

from src.contact_flow.io import load_states
from src.contact_flow.physics import PhysicsConfig, geometry


class InputTests(unittest.TestCase):
    def test_procedural_labels_and_nondefault_ground(self):
        physics = PhysicsConfig(y_ground=10, xy_limit=1.5)
        state, radius, labels = load_states("", "train", 6, 42, 5,
                                           physics=physics, return_scene_types=True)
        self.assertEqual(labels, ["vertical_stack", "oblique_stack", "free_flight"] * 2)
        self.assertTrue((state[..., 1] >= 10).all())
        self.assertEqual(float(geometry(state[..., :2], radius, physics)["energy"].sum()), 0.0)

    def test_dataset_source_labels_do_not_require_target(self):
        source = torch.tensor([[[0., 2., 0., 0.]], [[1., 2., 0., 0.]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.pt"
            torch.save({"train": {"source": source, "radius": torch.full((2, 1), .5),
                                   "scene_type": ["a", "b"]}}, path)
            selected, _, labels = load_states(str(path), "train", 2, 42,
                                              return_scene_types=True)
            for item, label in zip(selected, labels):
                self.assertEqual(label, "a" if item[0, 0] == 0 else "b")


if __name__ == "__main__":
    unittest.main()
