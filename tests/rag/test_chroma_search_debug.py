from __future__ import annotations

from uuid import uuid4

from app.rag.vector_store import ChromaVectorStore, HashEmbeddingFunction


def test_chroma_search_filters_by_document_id_without_upload_directory(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_debug",
        collection_name=f"test_chroma_debug_{uuid4().hex}",
        embedding_function=HashEmbeddingFunction(),
    )
    store.add_chunks(
        document_id="doc-test-1",
        file_name="manual.txt",
        chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "设备启动前需要检查电源、急停按钮、防护罩和现场安全状态。",
                "start_index": 0,
                "end_index": 32,
            },
            {
                "chunk_id": "chunk-2",
                "text": "启动后应观察指示灯、运行声音和温度，确认设备处于稳定状态。",
                "start_index": 33,
                "end_index": 66,
            },
        ],
    )

    results = store.search(document_id="doc-test-1", question="设备启动前需要检查什么？", top_k=5)

    assert len(results) > 0
    assert results[0]["retrieval_source"] == "chroma"
    assert results[0]["metadata"]["document_id"] == "doc-test-1"

    missing_results = store.search(document_id="doc-not-exist", question="设备启动前需要检查什么？", top_k=5)

    assert missing_results == []
