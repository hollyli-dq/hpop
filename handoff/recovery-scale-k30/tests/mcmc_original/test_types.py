"""Validation tests for the immutable dataclasses of the original model."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hpop.mcmc_original.types import Segment, Segmentation, SkillTemplate


# --------------------------------------------------------------------------
# Segment
# --------------------------------------------------------------------------


def test_valid_segment():
    segment = Segment(start=0, end=5, skill=2)
    assert segment.start == 0
    assert segment.end == 5
    assert segment.skill == 2
    assert segment.length == 5


def test_segment_is_frozen():
    segment = Segment(start=0, end=3, skill=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.start = 1


def test_segment_of_length_one_is_valid():
    assert Segment(start=7, end=8, skill=0).length == 1


def test_segment_rejects_negative_start():
    with pytest.raises(ValueError, match="start must be >= 0"):
        Segment(start=-1, end=4, skill=0)


@pytest.mark.parametrize("end", [3, 2, 0])
def test_segment_rejects_end_not_greater_than_start(end):
    with pytest.raises(ValueError, match="end must be > start"):
        Segment(start=3, end=end, skill=0)


def test_segment_rejects_negative_skill():
    with pytest.raises(ValueError, match="skill must be >= 0"):
        Segment(start=0, end=4, skill=-1)


@pytest.mark.parametrize("bad", [1.5, "0", None])
def test_segment_rejects_non_integer_fields(bad):
    with pytest.raises(ValueError, match="must be an integer"):
        Segment(start=bad, end=4, skill=0)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def test_valid_contiguous_segmentation():
    segmentation = Segmentation(
        segments=(
            Segment(0, 3, 0),
            Segment(3, 7, 1),
            Segment(7, 10, 0),
        )
    )
    assert segmentation.length == 10
    assert len(segmentation) == 3


def test_single_segment_segmentation_is_valid():
    segmentation = Segmentation(segments=(Segment(0, 4, 0),))
    assert segmentation.length == 4


def test_segmentation_rejects_empty():
    with pytest.raises(ValueError, match="at least one segment"):
        Segmentation(segments=())


def test_segmentation_rejects_gap():
    with pytest.raises(ValueError, match="gap"):
        Segmentation(segments=(Segment(0, 3, 0), Segment(4, 8, 1)))


def test_segmentation_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        Segmentation(segments=(Segment(0, 5, 0), Segment(3, 8, 1)))


def test_segmentation_rejects_first_segment_not_starting_at_zero():
    with pytest.raises(ValueError, match="must start at 0"):
        Segmentation(segments=(Segment(1, 4, 0), Segment(4, 6, 1)))


def test_segmentation_rejects_non_segment_elements():
    with pytest.raises(ValueError, match="must be a Segment"):
        Segmentation(segments=(Segment(0, 3, 0), (3, 6, 1)))


def test_segmentation_end_positions_strictly_increase():
    segmentation = Segmentation(
        segments=(Segment(0, 1, 0), Segment(1, 2, 1), Segment(2, 9, 0))
    )
    ends = [segment.end for segment in segmentation.segments]
    assert ends == sorted(set(ends))


def test_segmentation_is_frozen():
    segmentation = Segmentation(segments=(Segment(0, 2, 0),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        segmentation.segments = ()


def test_segmentation_accepts_a_list_and_stores_a_tuple():
    segmentation = Segmentation(segments=[Segment(0, 2, 0), Segment(2, 5, 1)])
    assert isinstance(segmentation.segments, tuple)
    assert segmentation.length == 5


# --------------------------------------------------------------------------
# SkillTemplate
# --------------------------------------------------------------------------


def valid_template(**overrides) -> SkillTemplate:
    kwargs = {
        "cpa_labels": (0, 1, 2),
        "u": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        "beta": 2.0,
        "epsilon": 0.05,
    }
    kwargs.update(overrides)
    return SkillTemplate(**kwargs)


def test_valid_skill_template():
    template = valid_template()
    assert template.cpa_labels == (0, 1, 2)
    assert template.n_roles == 3
    assert template.latent_dim == 2
    assert template.beta == 2.0
    assert template.epsilon == 0.05


def test_skill_template_is_frozen_and_u_is_read_only():
    template = valid_template()
    with pytest.raises(dataclasses.FrozenInstanceError):
        template.beta = 1.0
    with pytest.raises(ValueError):
        template.u[0, 0] = 99.0


def test_skill_template_does_not_freeze_the_callers_array():
    u = np.zeros((2, 2))
    valid_template(cpa_labels=(0, 1), u=u)
    u[0, 0] = 1.0  # must not raise
    assert u[0, 0] == 1.0


def test_skill_template_accepts_list_input_for_u():
    template = valid_template(cpa_labels=(0, 1), u=[[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(template.u, np.ndarray)
    assert template.u.dtype == float


def test_skill_template_rejects_row_label_mismatch():
    with pytest.raises(ValueError, match="one row per CPA label"):
        valid_template(cpa_labels=(0, 1))


def test_skill_template_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="must be unique"):
        valid_template(cpa_labels=(0, 1, 1))


def test_skill_template_rejects_empty_labels():
    with pytest.raises(ValueError, match="at least one role"):
        valid_template(cpa_labels=(), u=np.zeros((0, 2)))


@pytest.mark.parametrize(
    "bad_u",
    [
        np.array([1.0, 2.0, 3.0]),  # 1-D
        np.zeros((3, 2, 1)),  # 3-D
    ],
)
def test_skill_template_rejects_non_2d_u(bad_u):
    with pytest.raises(ValueError, match="2-D"):
        valid_template(u=bad_u)


def test_skill_template_rejects_zero_latent_dimension():
    with pytest.raises(ValueError, match="latent dimension"):
        valid_template(u=np.zeros((3, 0)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_skill_template_rejects_non_finite_u(bad_value):
    u = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, bad_value]])
    with pytest.raises(ValueError, match="NaN or inf"):
        valid_template(u=u)


def test_skill_template_rejects_negative_beta():
    with pytest.raises(ValueError, match="beta must be >= 0"):
        valid_template(beta=-0.1)


def test_skill_template_accepts_zero_beta():
    assert valid_template(beta=0.0).beta == 0.0


@pytest.mark.parametrize("bad_epsilon", [-0.01, 1.0, 1.5])
def test_skill_template_rejects_invalid_epsilon(bad_epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        valid_template(epsilon=bad_epsilon)


def test_skill_template_accepts_zero_epsilon():
    assert valid_template(epsilon=0.0).epsilon == 0.0


def test_skill_template_rejects_non_numeric_beta_and_epsilon():
    with pytest.raises(ValueError, match="beta must be a real number"):
        valid_template(beta="2.0")
    with pytest.raises(ValueError, match="epsilon must be a real number"):
        valid_template(epsilon=None)


def test_skill_template_rejects_non_integer_labels():
    with pytest.raises(ValueError, match="must be an integer"):
        valid_template(cpa_labels=(0, 1, "2"))
