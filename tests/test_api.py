import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from recallengine import db
from recallengine.embedder import load_model
from recallengine.main import app


@pytest.fixture(scope="session", autouse=True)
def load_embedding_model():
    load_model()


@pytest_asyncio.fixture(autouse=True)
async def setup_db_pool():
    await db.init_db()
    yield
    await db.close_db()


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_app_resources():
    await db.init_db()
    load_model()
    yield
    await db.close_db()


@pytest.mark.asyncio
async def test_document_lifecycle_upload_search_paginate_delete():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Reject unsupported file extension
        bad_resp = await client.post(
            "/documents/upload",
            files={
                "file": ("malware.exe", b"binary content", "application/octet-stream")
            },
        )
        assert bad_resp.status_code == 400

        # 2. Upload a valid Markdown document
        md_content = (
            b"# Podman Networking\n\n"
            b"Rootless Podman containers communicate over a custom bridge network "
            b"and expose PostgreSQL on port 5432 for local integration testing."
        )
        upload_resp = await client.post(
            "/documents/upload",
            files={"file": ("podman_guide.md", md_content, "text/markdown")},
        )
        assert upload_resp.status_code == 201
        created = upload_resp.json()
        doc_id = created["document_id"]
        assert created["title"] == "podman_guide"
        assert created["chunks_created"] == 1

        try:
            # 3. Verify pagination lists the uploaded document
            list_resp = await client.get("/documents?limit=5&offset=0")
            assert list_resp.status_code == 200
            listing = list_resp.json()
            assert listing["total"] >= 1
            assert any(item["id"] == doc_id for item in listing["items"])

            # 4. Semantic search matches meaning without exact wording
            search_resp = await client.post(
                "/search",
                json={
                    "query": "Which port does the database listen on in rootless containers?",
                    "top_k": 3,
                    "min_similarity": 0.35,
                },
            )
            assert search_resp.status_code == 200
            results = search_resp.json()["results"]
            assert len(results) >= 1
            assert results[0]["document_id"] == doc_id
            assert results[0]["similarity"] >= 0.35

            # 5. Unrelated query is filtered out by similarity threshold
            noise_resp = await client.post(
                "/search",
                json={
                    "query": "How do I bake sourdough bread at home?",
                    "top_k": 3,
                    "min_similarity": 0.35,
                },
            )
            assert noise_resp.status_code == 200
            assert noise_resp.json()["results"] == []
        finally:
            # 6. Delete document (verifies ON DELETE CASCADE cleanup) and confirm 404 on repeat
            del_resp = await client.delete(f"/documents/{doc_id}")
            assert del_resp.status_code == 204

            del_again = await client.delete(f"/documents/{doc_id}")
            assert del_again.status_code == 404
