import sys, json
from pathlib import Path
sys.path.insert(0, "results/2026-10-17-beyond-vlm/phase2/i09/live")
import driver as D
d = Path(sys.argv[1]); d.mkdir(exist_ok=True)
p = d / "ledger.jsonl"; p.unlink(missing_ok=True)
led = D.Ledger(p, "glm53", 120, 60)
for i in range(5): led.reserve({"i": i})
p.unlink()
try: led.reserve({"i": 9}); print("FAIL: no LedgerLost")
except D.LedgerLost as e: print("U1 floor: LedgerLost; counts kept:", led.total, led.per_model["glm53"], "(expect 5 5)")
p.write_text('{"event": "sent", "model": "glm53"}\n{not json\n')
try: D.Ledger(p, "glm53", 120, 60); print("FAIL: unreadable line accepted")
except D.LedgerLost as e: print("U2 unreadable line: LedgerLost:", str(e)[-60:])
