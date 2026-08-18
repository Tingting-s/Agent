from __future__ import annotations

from contextlib import suppress
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.vector_store import get_vector_store  # noqa: E402
from app.services.document_storage_service import DocumentStorageService  # noqa: E402


QUESTION = "设备启动前需要检查什么？"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    storage_service = DocumentStorageService()
    documents = storage_service.list_documents()
    if not documents:
        print("No document records found in SQLite documents table.")
        return 1

    document = documents[0]
    document_id = str(document.get("document_id") or "")
    print("SQLite latest document:")
    print(f"  document_id: {document_id}")
    print(f"  original_file_name: {document.get('original_file_name') or ''}")
    print(f"  stored_file_name: {document.get('stored_file_name') or ''}")
    print(f"  chunk_count: {document.get('chunk_count') or 0}")
    print(f"  vector_indexed: {bool(document.get('vector_indexed'))}")

    vector_store = get_vector_store()
    print("Vector store:")
    print(f"  persist_directory: {vector_store.persist_directory}")
    print(f"  collection: {vector_store.collection_name}")

    collection_total = _collection_total_count(vector_store.collection)
    if collection_total is None:
        print("  collection_total_count: unavailable")
    else:
        print(f"  collection_total_count: {collection_total}")

    try:
        chroma_chunk_count = vector_store.get_document_chunk_count(document_id)
        print(f"  document_chunk_count: {chroma_chunk_count}")
    except Exception as exc:
        print(f"  document_chunk_count_error: {type(exc).__name__}: {exc}")
        return 1
    if int(document.get("chunk_count") or 0) > 0 and bool(document.get("vector_indexed")) and chroma_chunk_count == 0:
        print(
            "  note: SQLite marks this document as indexed, but Chroma has no chunks for this document_id. "
            "Old documents may need re-upload or re-index after embedding compatibility fixes."
        )

    try:
        results = vector_store.search(document_id=document_id, question=QUESTION, top_k=5)
    except Exception as exc:
        print("Search returned no usable results because search raised an exception.")
        print(f"  search_exception: {type(exc).__name__}: {exc}")
        return 1

    print(f"Raw search result count: {len(results)}")
    if not results:
        if collection_total == 0:
            print("Empty search diagnosis: collection has no data.")
        elif chroma_chunk_count == 0:
            print("Empty search diagnosis: collection has data, but where document_id did not match.")
        else:
            print("Empty search diagnosis: where document_id matched chunks, but query returned no results.")
        return 0

    for index, result in enumerate(results, start=1):
        metadata = dict(result.get("metadata") or {})
        score = result.get("score")
        distance = result.get("distance")
        text = str(result.get("text") or "").replace("\n", " ")[:100]
        print(f"Result {index}:")
        print(f"  chunk_id: {result.get('chunk_id') or metadata.get('chunk_id') or ''}")
        print(f"  document_id: {result.get('document_id') or metadata.get('document_id') or ''}")
        print(f"  retrieval_source: {result.get('retrieval_source') or metadata.get('retrieval_source') or ''}")
        print(f"  score: {score}")
        print(f"  distance: {distance}")
        print(f"  text: {text}")

    return 0


def _collection_total_count(collection: Any) -> int | None:
    count = getattr(collection, "count", None)
    if callable(count):
        with suppress(Exception):
            return int(count())

    items = getattr(collection, "_items", None)
    if isinstance(items, dict):
        return len(items)

    try:
        result = collection.get(include=[])
        return len(result.get("ids") or [])
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
