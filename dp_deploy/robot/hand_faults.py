"""SDK severity classification with bounded warning notifications."""

from collections import deque


class HandFaults:
    def __init__(self, sdk, side):
        self.sdk, self.side = sdk, side
        self.cache = {}
        self.active = {}
        self.pending = deque(maxlen=40)

    def check(self, joints, now):
        present = set()
        for joint in joints:
            code = int(joint.error_code_current)
            if not code:
                continue
            nid = int(joint.nid)
            present.add(nid)
            if code not in self.cache:
                try:
                    self.cache[code] = dict(self.sdk.WujiHand2.describe_error(code))
                except Exception:
                    self.cache[code] = {}
            info = self.cache[code]
            severity = str(info.get("severity", "Unknown"))
            label = "左手" if self.side == "left" else "右手"
            reason = info.get("desc") or info.get("name") or "未知错误"
            text = f"{label} NID={nid}, code={code}, severity={severity}: {reason}"
            if info.get("cause"):
                text += "; 原因: " + str(info["cause"])
            if info.get("resolution"):
                text += "; 建议: " + str(info["resolution"])
            # Only explicitly decoded Warning is nonterminal. Unknown and all
            # stopping severities fail closed, irrespective of auto-clear policy.
            if severity != "Warning":
                raise ValueError("Hand firmware reports a joint error; " + text)
            old = self.active.get(nid)
            if old is None or old[0] != code or now - old[1] >= 5:
                self.pending.append(text)
                self.active[nid] = (code, now)
        for nid in list(self.active):
            if nid not in present:
                del self.active[nid]

    def stalled_nids(self):
        return {nid for nid, (code, _) in self.active.items()
                if self.cache[code].get("severity") == "Warning" and self.cache[code].get("name") == "Stall"}

    def drain(self):
        result = list(self.pending)
        self.pending.clear()
        return result
