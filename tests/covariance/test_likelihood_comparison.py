"""Check file conventions and identical scale selection in notebook comparisons."""

import numpy as np
import pytest

from cosmolike_notebook_utils.covariance.likelihood import (
    read_likelihood_covariance,
    select_likelihood_entries,
)


@pytest.mark.parametrize("format", ["three", "four", "ten", "packed"])
def test_supplied_formats_and_cuts(tmp_path, format):
    """All supported files preserve a negative cross and the same row cuts."""
    expected = np.array([[4., -1., 0.5], [-1., 9., 0.2], [0.5, 0.2, 16.]])
    first, second = np.triu_indices(n=3)
    values = expected[first, second]
    filename = "cov.txt"
    if format == "packed":
        filename = "cov.npy"
        np.save(file=tmp_path/filename, arr=values)
    else:
        if format == "three":
            table = np.column_stack(tup=(first, second, values))
        elif format == "four":
            table = np.column_stack(tup=(first, second, values*0.25, values*0.75))
        else:
            table = np.zeros(shape=(len(values), 10))
            table[:, 0] = first
            table[:, 1] = second
            table[:, 8] = values*0.25
            table[:, 9] = values*0.75
        np.savetxt(fname=tmp_path/filename, X=table)
    (tmp_path/"base.dataset").write_text(f"cov_file = {filename}\nmask_file = cut.mask\n")
    (tmp_path/"selected.dataset").write_text("DEFAULT(base.dataset)\n")
    np.savetxt(fname=tmp_path/"cut.mask", X=[[0, 1], [1, 1], [2, 0]])

    supplied = read_likelihood_covariance(dataset=tmp_path/"selected.dataset")
    np.testing.assert_array_equal(supplied["total"], expected)
    forecast = {
        "total": expected*2,
        "gaussian": expected,
        "ssc": expected*0.75,
        "cng": expected*0.25,
    }
    selected = select_likelihood_entries(
        forecast=forecast, supplied=supplied,
        block_sizes=[2, 1], block_labels=["shear", "clustering"],
    )
    np.testing.assert_array_equal(selected["supplied"], expected[:2, :2])
    np.testing.assert_array_equal(selected["total"], expected[:2, :2]*2)
    np.testing.assert_array_equal(selected["indices"], [0, 1])
    assert selected["block_sizes"] == [2]
    assert selected["block_labels"] == ["shear"]
    assert forecast["total"].shape == (3, 3)
    np.testing.assert_array_equal(supplied["total"], expected)

    # A leading submatrix supports the galaxy/shear block in a joint CMB
    # file. It must preserve the original file size in the saved metadata.
    prefix = read_likelihood_covariance(dataset=tmp_path/"selected.dataset", size=2)
    assert prefix["file_size"] == 3
    np.testing.assert_array_equal(prefix["total"], expected[:2, :2])


def test_defined_null_rows_require_a_physical_mask():
    """A selected Y null row is an error, never a diagonal regularization."""
    forecast = {
        "total": np.eye(3),
        "gaussian": np.eye(3),
        "ssc": np.zeros(shape=(3, 3)),
        "cng": np.zeros(shape=(3, 3)),
        "valid_indices": np.array([0, 1]),
    }
    supplied = {"total": np.eye(3), "mask": np.array([True, False, True])}
    with pytest.raises(ValueError, match="Y null row"):
        select_likelihood_entries(
            forecast=forecast, supplied=supplied,
            block_sizes=[3], block_labels=["cluster lensing"],
        )


def test_bad_file_indices_fail_before_selection(tmp_path):
    """Fractional file indices cannot silently become integer row labels."""
    (tmp_path/"example.dataset").write_text("cov_file = cov.txt\nmask_file = cut.mask\n")
    np.savetxt(fname=tmp_path/"cut.mask", X=[[0, 1], [1, 1]])
    np.savetxt(fname=tmp_path/"cov.txt", X=[[0, 0, 1], [0.5, 1, 0.2], [1, 1, 2]])
    with pytest.raises(ValueError, match="covariance indices"):
        read_likelihood_covariance(dataset=tmp_path/"example.dataset")
