import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import geoscience_brain as brain

def test_domain_knowledge_and_guardrails():
    assert len(brain.DOMAIN_KNOWLEDGE) >= 8
    assert "Never fabricate missing evidence" in brain.SYSTEM_PROMPT
    assert "rock physics" in brain.SYSTEM_PROMPT.lower()

def test_evidence_question_relevance():
    questions = brain.get_suggested_questions("Seismic QC", 6)
    assert len(questions) <= 6
    assert len(questions) == len(set(questions))
    assert any("NRMS" in q or "repeatab" in q.lower() for q in questions)

def test_compiles():
    root=Path(__file__).resolve().parents[1]
    for name in ["geoscience_brain.py", "angsi_jr_v9.py", "copilot.py"]:
        compile((root/name).read_text(),str(root/name),'exec')
