import numpy as np

from opticell.mask_edit import apply_brush_edit, instance_count


def _labels():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 2
    return labels


def test_add_brush_expands_existing_instance():
    labels = _labels()
    stroke = np.zeros_like(labels, dtype=bool)
    stroke[2:5, 2] = True
    corrected, affected = apply_brush_edit(labels, stroke, mode="add", label_id=1)
    assert affected == (1,)
    assert (corrected[stroke] == 1).all()
    assert instance_count(corrected) == 2


def test_erase_brush_records_affected_ids():
    labels = _labels()
    stroke = labels == 2
    corrected, affected = apply_brush_edit(labels, stroke, mode="erase")
    assert affected == (2,)
    assert not np.any(corrected == 2)
    assert instance_count(corrected) == 1


def test_new_object_assigns_separate_ids_to_disconnected_strokes():
    labels = _labels()
    stroke = np.zeros_like(labels, dtype=bool)
    stroke[0, 7] = True
    stroke[7, 0] = True
    corrected, created = apply_brush_edit(labels, stroke, mode="new_object")
    assert created == (3, 4)
    assert corrected[0, 7] == 3
    assert corrected[7, 0] == 4
    assert instance_count(corrected) == 4


def test_add_rejects_unknown_instance_id():
    labels = _labels()
    stroke = np.zeros_like(labels, dtype=bool)
    stroke[0, 0] = True
    with np.testing.assert_raises(ValueError):
        apply_brush_edit(labels, stroke, mode="add", label_id=999)


def test_empty_stroke_is_noop():
    labels = _labels()
    corrected, affected = apply_brush_edit(labels, np.zeros_like(labels, dtype=bool), mode="erase")
    assert affected == ()
    assert np.array_equal(corrected, labels)
