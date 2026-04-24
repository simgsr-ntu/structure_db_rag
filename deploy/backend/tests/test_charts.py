# deploy/backend/tests/test_charts.py
import os
import importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def charts_client(test_db, monkeypatch):
    monkeypatch.setenv("DATA_DIR", os.path.dirname(test_db))
    import deploy.backend.api.charts as charts_mod
    import importlib
    importlib.reload(charts_mod)
    app = FastAPI()
    app.include_router(charts_mod.router)
    return TestClient(app)


def test_stats(charts_client):
    r = charts_client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_sermons"] == 6
    assert data["total_speakers"] == 3
    assert data["year_min"] == 2022
    assert data["year_max"] == 2024


def test_by_year(charts_client):
    r = charts_client.get("/api/charts/by-year")
    assert r.status_code == 200
    years = {item["year"]: item["count"] for item in r.json()}
    assert years[2022] == 2
    assert years[2023] == 2
    assert years[2024] == 2


def test_by_speaker(charts_client):
    r = charts_client.get("/api/charts/by-speaker")
    assert r.status_code == 200
    speakers = {item["speaker"]: item["count"] for item in r.json()}
    assert speakers["Pastor A"] == 3
    assert speakers["Pastor B"] == 2


def test_by_verse(charts_client):
    r = charts_client.get("/api/charts/by-verse")
    assert r.status_code == 200
    books = {item["bible_book"]: item["count"] for item in r.json()}
    assert books["John"] == 3


def test_scatter(charts_client):
    r = charts_client.get("/api/charts/scatter")
    assert r.status_code == 200
    points = {(p["year"], p["speaker"]): p["count"] for p in r.json()}
    assert points[(2022, "Pastor A")] == 2
    assert points[(2023, "Pastor B")] == 2
    assert points[(2024, "Pastor A")] == 1
    assert points[(2024, "Pastor C")] == 1
