from rag import SimpleRAG


def test_split_and_search():
    rag = SimpleRAG(chunk_size=40, overlap=5)
    rag.build_index(
        "Sentinel-1 provides VV and VH SAR observations. "
        "Multi-temporal SAR is useful for monitoring rice phenology. "
        "RAG combines retrieval with generation."
    )

    assert rag.ready
    assert len(rag.chunks) >= 2

    results = rag.search("rice phenology", top_k=2)

    assert len(results) == 2
    assert "score" in results[0]
    assert "text" in results[0]
