"""Regression tests for the Lab 16 provenance and scope guards."""

from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "16_hybrid_retrieval.ipynb"
HERON_CANARY = "HERON-ZEPHYR-4471"


@pytest.fixture(scope="module")
def retrieval():
    """Load the retrieval implementations from the notebook cells under test."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "lab16_scope_guard"}
    for cell_index in (1, 3, 4, 5):
        source = "".join(notebook.cells[cell_index].source)
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    yield namespace
    namespace["client"].close()


def test_search_requires_nonempty_scope(retrieval):
    with pytest.raises(ValueError, match="scope is required") as error:
        retrieval["search"](retrieval["embed"]("customer concentration"), scope="")
    print(f"empty_scope_error={error.value}")


def test_bm25_side_is_scoped_before_fusion(retrieval):
    hits = retrieval["hybrid"](
        HERON_CANARY,
        retrieval["embed"]("customer concentration"),
        scope="deal_a",
        k=20,
    )
    metadata = {chunk["chunk_id"]: chunk for chunk in retrieval["chunks"]}

    assert hits
    assert all(metadata[chunk_id]["deal_id"] != "deal_b" for chunk_id in hits)
    assert all(HERON_CANARY not in metadata[chunk_id]["text"] for chunk_id in hits)
    print(
        f"canary_query_hits={len(hits)} "
        f"deal_b_hits=0 canary_text_hits=0 canary={HERON_CANARY}"
    )
