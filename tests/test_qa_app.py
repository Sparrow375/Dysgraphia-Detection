import asyncio
from aiohttp.test_utils import TestServer, TestClient
from qa.app import create_app


def test_api_students():
    async def _test():
        app = create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.get("/api/students")
            assert resp.status == 200
            data = await resp.json()
            assert "students" in data
            assert len(data["students"]) > 0
            s0 = data["students"][0]
            assert "student_id" in s0
            assert "grade" in s0
            assert "sentences_count" in s0
        finally:
            await client.close()

    asyncio.run(_test())


def test_api_student_detail():
    async def _test():
        app = create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.get("/api/student/G4_A_Roll01")
            assert resp.status == 200
            data = await resp.json()
            assert data["student_id"] == "G4_A_Roll01"
            assert "sentences" in data
            assert len(data["sentences"]) >= 4
            assert "page_url" in data
            assert "overlay_url" in data
        finally:
            await client.close()

    asyncio.run(_test())
