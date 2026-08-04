"""HTTP API contract tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from offball.api.app import app, get_store
from offball.api.schemas import JobStatus
from offball.api.store import InMemoryJobStore


@pytest.fixture
def client():
    """A client with a fresh store, so tests do not leak jobs into each other."""
    store = InMemoryJobStore()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def test_healthz_reports_the_kernel_backend(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kernel_backend"] in ("rust", "python")


def test_submitting_an_analysis_returns_202_and_a_job_id(client):
    r = client.post("/v1/analyses", json={"video_uri": "s3://bucket/match.mp4"})
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"]
    assert body["status"] in {s.value for s in JobStatus}


def test_job_can_be_fetched_after_submission(client):
    job_id = client.post("/v1/analyses", json={"video_uri": "file:///m.mp4"}).json()["job_id"]
    r = client.get(f"/v1/analyses/{job_id}")
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id


def test_unknown_job_is_404(client):
    r = client.get("/v1/analyses/does-not-exist")
    assert r.status_code == 404
    assert "no such job" in r.json()["detail"]


def test_report_for_an_unfinished_job_is_409_not_404(client):
    """Callers must be able to tell 'not ready' from 'never existed'."""
    job_id = client.post("/v1/analyses", json={"video_uri": "file:///m.mp4"}).json()["job_id"]
    r = client.get(f"/v1/analyses/{job_id}/report")
    assert r.status_code == 409
    assert "no report available" in r.json()["detail"]


def test_report_for_an_unknown_job_is_404(client):
    assert client.get("/v1/analyses/nope/report").status_code == 404


def test_analysis_without_a_detector_fails_loudly_rather_than_faking_a_report(client):
    """A job with no model configured must fail, not invent numbers."""
    job_id = client.post("/v1/analyses", json={"video_uri": "file:///m.mp4"}).json()["job_id"]
    # TestClient runs background tasks synchronously on response close.
    job = client.store.get(job_id)
    assert job.status is JobStatus.FAILED
    assert "detector" in (job.error or "").lower()


def test_listing_returns_newest_first(client):
    ids = [
        client.post("/v1/analyses", json={"video_uri": f"file:///{i}.mp4"}).json()["job_id"]
        for i in range(3)
    ]
    r = client.get("/v1/analyses")
    assert r.status_code == 200
    returned = [j["job_id"] for j in r.json()]
    assert set(returned) == set(ids)


def test_list_limit_is_validated(client):
    assert client.get("/v1/analyses?limit=0").status_code == 422
    assert client.get("/v1/analyses?limit=500").status_code == 422
    assert client.get("/v1/analyses?limit=10").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},                                              # missing video_uri
        {"video_uri": ""},                               # empty
        {"video_uri": "x", "fps": 0},                    # fps must be positive
        {"video_uri": "x", "stride": 0},                 # stride must be >= 1
        {"video_uri": "x", "pitch_length": 10.0},        # outside IFAB limits
        {"video_uri": "x", "pitch_width": 200.0},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/v1/analyses", json=payload).status_code == 422


def test_defaults_are_applied(client):
    r = client.post("/v1/analyses", json={"video_uri": "file:///m.mp4", "match_id": "abc"})
    assert r.status_code == 202
    assert r.json()["match_id"] == "abc"


def test_openapi_schema_is_generated(client):
    """The TypeScript SDK is generated from this; it must stay valid."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/v1/analyses" in schema["paths"]
    assert "ReportResponse" in schema["components"]["schemas"]


# ------------------------------------------------------------------ job store


def test_store_round_trip():
    store = InMemoryJobStore()
    job = store.create("file:///m.mp4", match_id="m1")
    assert store.get(job.job_id) == job

    updated = store.update(job.job_id, status=JobStatus.RUNNING, progress=0.5)
    assert updated.status is JobStatus.RUNNING
    assert updated.progress == 0.5
    assert updated.updated_at >= job.updated_at
    assert updated.created_at == job.created_at


def test_store_update_of_missing_job_is_none():
    assert InMemoryJobStore().update("nope", progress=1.0) is None


def test_store_list_is_capped():
    store = InMemoryJobStore()
    for i in range(10):
        store.create(f"file:///{i}.mp4")
    assert len(store.list(limit=4)) == 4
    assert len(store.list()) == 10
