"""Firmware severity and bounded notifications, no device connection."""

from types import SimpleNamespace as S

import pytest

from .hand_faults import HandFaults


def classifier(severity):
    return HandFaults(S(WujiHand2=S(describe_error=lambda code: {
        "severity": severity, "desc": "堵转检测", "cause": "电机低速高电流持续",
        "resolution": "负载移除后自动消除"
    })), "right")


def test_warning_continues_and_rate_limits_then_reappears_after_clear():
    status = classifier("Warning")
    joint = S(nid=8, error_code_current=7)
    status.check([joint], 10)
    messages = status.drain()
    assert len(messages) == 1
    assert "右手 NID=8" in messages[0]
    assert "电机低速高电流持续" in messages[0]
    status.check([joint], 11)
    assert not status.drain()
    status.check([joint], 15)
    assert len(status.drain()) == 1
    status.check([S(nid=8, error_code_current=0)], 16)
    status.check([joint], 17)
    assert len(status.drain()) == 1


@pytest.mark.parametrize("severity", ["DeferredStop", "ImmediateStop", "Fatal", "Unknown", ""])
def test_stopping_or_unknown_severity_rejects(severity):
    status = classifier(severity)
    with pytest.raises(ValueError, match="joint error.*NID=8"):
        status.check([S(nid=8, error_code_current=7)], 10)


def test_decoder_missing_is_terminal():
    status = HandFaults(None, "left")
    with pytest.raises(ValueError, match="Unknown"):
        status.check([S(nid=8, error_code_current=7)], 10)


def test_only_decoded_stall_warning_activates_directional_limit():
    status = HandFaults(S(WujiHand2=S(describe_error=lambda code: {
        "severity": "Warning", "name": "Stall" if code == 7 else "OtherWarning"
    })), "right")
    status.check([S(nid=22, error_code_current=7), S(nid=8, error_code_current=21)], 10)
    assert status.stalled_nids() == {22}
    status.check([S(nid=22, error_code_current=0), S(nid=8, error_code_current=21)], 11)
    assert not status.stalled_nids()
