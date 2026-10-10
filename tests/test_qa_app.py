import json
import pytest
from aiohttp import web
from qa.app import create_app

@pytest.fixture
def cli(event_loop, aiohttp_client):
    app = create_app()
    return event_loop.run_until_complete(aiohttp_client(app))


def test_api_students(cli):
    resp = cli.get("/api/students")
    assert resp.status == 200
    data = cli.loop.run_until_complete(resp.json())
    assert "students" in data
    assert len(data["students"]) > 0
    s0 = data["students"][0]
    assert "student_id" in s0
    assert "grade" in s0
    assert "sentences_count" in s0


def test_api_student_detail(cli):
    resp = cli.get("/api/student/G4_A_Roll01")
    assert resp.status == 200
    data = cli.loop.run_until_complete(resp.json())
    assert data["student_id"] == "G4_A_Roll01"
    assert "sentences" in data
    assert len(data["sentences"]) >= 4
    assert "page_url" in data
    assert "overlay_url" in data
