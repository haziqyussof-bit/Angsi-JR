"""Offline contract tests for V10.2.1 Gemma chat integration."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_chat


def test_alternating_roles_and_no_system_role():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is NRMS?"},
    ]
    messages = ai_chat.build_messages(history, {"wells": ["Well-1"]})
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"].endswith("hi")
    assert "Well-1" in messages[0]["content"]
    assert messages[-1]["content"] == "what is NRMS?"


def test_bad_consecutive_turns_are_normalized():
    history = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "third"},
    ]
    turns = ai_chat.clean_turns(history)
    assert [m["role"] for m in turns] == ["user", "assistant", "user"]
    assert "second" in turns[0]["content"]
    assert "orphan" not in str(turns)


def test_recent_history_cap():
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"turn{i}"} for i in range(60)]
    # Final 21st user message.
    history.append({"role": "user", "content": "latest"})
    turns = ai_chat.clean_turns(history)
    assert len(turns) <= 20
    assert turns[-1]["content"] == "latest"
    assert turns[0]["role"] == "user"


def test_context_privacy_default_and_opt_in():
    import pandas as pd
    wells = pd.DataFrame({"Well": ["Well-1"], "X": [462000.0]})
    matches = pd.DataFrame({"Well": ["Well-1"], "IL": [950], "XL": [2400], "Distance_m": [5.1]})
    tops = pd.DataFrame({"Well": ["Well-1"], "Marker": ["I-25"], "TVDSS_m": [1540.0],
                         "TWT_ms": [1650.0], "Checkshot_Donor": ["Well-1"],
                         "Checkshot_Type": ["OWN"]})
    data = (None, None, None, wells, None, None, matches, tops)
    state = {"sel_il": 950, "sel_xl": 2400}
    off = ai_chat.build_context(data, state, False)
    assert off["field_evidence_shared"] is False
    assert "wells" not in off
    assert "current_selection" not in off
    on = ai_chat.build_context(data, state, True)
    assert on["wells"] == ["Well-1"]
    assert on["current_selection"]["sel_il"] == 950
    assert on["marker_examples"][0]["Marker"] == "I-25"


class FakeClient:
    def __init__(self, streaming=True):
        self.requests = []
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=ai_chat.MODEL_ID)]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.streaming = streaming

    def create(self, **kwargs):
        self.requests.append(kwargs)
        messages = kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert all(messages[i]["role"] != messages[i + 1]["role"] for i in range(len(messages) - 1))
        if kwargs.get("stream"):
            if not self.streaming:
                raise RuntimeError("SSE streaming not supported")
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]),
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="there"))])]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="CONNECTION_OK"))])


def test_stream_chat_with_mock_organiser_client():
    client = FakeClient()
    reply = "".join(ai_chat.stream_chat([{"role": "user", "content": "hi"}], client=client))
    assert reply == "Hello there"
    assert client.requests[0]["stream"] is True


def test_nonstream_fallback_if_server_does_not_stream():
    client = FakeClient(streaming=False)
    reply = "".join(ai_chat.stream_chat([{"role": "user", "content": "hello"}], client=client))
    assert reply == "CONNECTION_OK"
    assert [c["stream"] for c in client.requests] == [True, False]


def test_inference_connection_check_works_even_without_data():
    result = ai_chat.check_connection(client=FakeClient())
    assert result["connected"]
    assert result["model"] == ai_chat.MODEL_ID
    assert "CONNECTION_OK" in result["reply"]


def test_app_has_only_one_copilot_panel_and_old_form_removed():
    root = Path(__file__).resolve().parents[1]
    src = (root/"angsi_jr_v10.py").read_text()
    assert src.count("def copilot_panel(data):") == 1
    start = src.index("def copilot_panel(data):")
    end = src.index("def nav_sidebar():", start)
    panel = src[start:end]
    assert "ai_chat.render_chat_panel" in panel
    assert "Authorized endpoint" not in panel
    assert "Read-only evidence" not in panel
    assert "Suggested scientific questions" not in panel
    assert (root/"analysis_dashboard.py").exists()
    assert (root/"v10_2_reservoir_engine.py").exists()
