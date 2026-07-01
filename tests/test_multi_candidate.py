"""Multi-candidate batch tests.

A recruiter CSV export naturally holds many candidate rows. Before per-candidate
grouping, run_pipeline conflated every row into ONE profile. These tests prove a
multi-row CSV now yields N separate, correctly-attributed profiles, while a
single candidate keeps the historical single-object shape.
"""
from __future__ import annotations

from src.transformer.pipeline import PipelineInputs, run_pipeline

_HEADER = (
    "first_name,last_name,headline,current_company,current_title,location,"
    "email,phone,linkedin_url,years_experience,top_skills,tags,date_added,"
    "current_title_start,prev_company,prev_title,prev_start,prev_end,"
    "education_institution,education_degree,education_field,education_end_year"
)

_ROWS = [
    (
        "Priya,Nair,Senior Platform Engineer,Fathom Robotics,Senior Platform Engineer,"
        '"San Francisco, CA, USA",priya.nair@gmail.com,+1 415 555 0173,'
        "https://linkedin.com/in/priya-nair,9,\"Python, Go, Kubernetes\",platform,"
        "2026-06-02,2022-01,Cloudera,Software Engineer,2017-08,2021-12,"
        "University of Illinois Urbana-Champaign,B.S.,Computer Science,2016"
    ),
    (
        "Marcus,Bell,Backend Engineer,Loomly,Backend Engineer,"
        '"Austin, TX, USA",marcus.bell@gmail.com,+1 512 555 0198,'
        "https://linkedin.com/in/marcus-bell,6,\"Java, PostgreSQL\",backend,"
        "2026-06-03,2021-03,Indeed,Engineer,2018-01,2021-02,"
        "University of Texas,B.S.,Computer Science,2017"
    ),
    (
        "Dana,Okoro,Data Engineer,Helix,Data Engineer,"
        '"Seattle, WA, USA",dana.okoro@gmail.com,+1 646 555 0121,'
        "https://linkedin.com/in/dana-okoro,7,\"Python, Spark\",data,"
        "2026-06-04,2020-05,Amazon,Engineer,2016-06,2020-04,"
        "University of Washington,B.S.,Computer Science,2015"
    ),
]


def _csv(*rows: str) -> str:
    return "\n".join([_HEADER, *rows]) + "\n"


def test_multi_row_csv_yields_one_profile_per_candidate():
    result = run_pipeline(PipelineInputs(csv_payload=_csv(*_ROWS)))

    assert isinstance(result, list), "multiple candidates must widen to a JSON array"
    assert len(result) == 3, [p["full_name"] for p in result]

    names = {p["full_name"] for p in result}
    assert names == {"Priya Nair", "Marcus Bell", "Dana Okoro"}

    # Distinct candidates get distinct ids — the conflation bug produced one id.
    ids = {p["candidate_id"] for p in result}
    assert len(ids) == 3

    # Attribution is not crossed: each profile carries only its own email.
    by_name = {p["full_name"]: p for p in result}
    assert by_name["Priya Nair"]["emails"] == ["priya.nair@gmail.com"]
    assert by_name["Marcus Bell"]["emails"] == ["marcus.bell@gmail.com"]
    assert by_name["Dana Okoro"]["emails"] == ["dana.okoro@gmail.com"]


def test_single_row_csv_keeps_single_object_shape():
    result = run_pipeline(PipelineInputs(csv_payload=_csv(_ROWS[0])))

    assert isinstance(result, dict), "one candidate must stay a single object"
    assert result["full_name"] == "Priya Nair"


def test_multi_candidate_include_audit_is_per_candidate():
    result = run_pipeline(PipelineInputs(csv_payload=_csv(*_ROWS)), include_audit=True)

    assert isinstance(result, list)
    assert len(result) == 3
    for element in result:
        assert set(element) == {"output", "audit_log"}
        assert isinstance(element["audit_log"], list)
        # Each element's merge audit belongs to exactly that candidate.
        merge_events = [e for e in element["audit_log"] if e["stage"] == "merge"]
        assert merge_events, "expected per-candidate merge audit"
