"""Assemble the sha-stamped R16 circulation file (single part, per-source sha256, END-line tripwires)."""
import hashlib
import pathlib

PARTS = [
    ("PART 1: R16 REPUBLICATION (grinder design, sealing circulation for A14)",
     "learnings/war_room/grinder_design_R16_republication.md"),
    ("PART 2: A17 AMENDMENT DRAFT (72B screen; files on sign-off)",
     "learnings/preregistration_amendment_2026-07-20_A17.md"),
    ("PART 3: LATENT-STATE AUDIT PROTOCOL (R15 blocking prereq, discharged)",
     "learnings/war_room/latent_state_audit_protocol.md"),
    ("PART 4: LATENT-STATE AUDIT RESULTS",
     "runs/latent_state_audit/report.md"),
    ("PART 5: DAILY BRIEF 2026-07-20 (incl. R16 open questions)",
     "learnings/daily_brief_2026-07-20.md"),
]

BAR = "=" * 80
out = []
for title, p in PARTS:
    text = pathlib.Path(p).read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    out.append(f"{BAR}\n{title}\nsource: {p}\nsha256: {sha}\n{BAR}\n\n{text}\n\n--- END OF {title.split(':')[0]} ---\n")

full = "\n".join(out) + "\n=== END OF R16 CIRCULATION (5 parts, all END lines above must be present) ===\n"
dest = pathlib.Path("learnings/panel/r16_circulation.md")
dest.write_text(full, encoding="utf-8")
print(f"{dest} written, {len(full)} chars, sha256 {hashlib.sha256(full.encode()).hexdigest()[:16]}")
