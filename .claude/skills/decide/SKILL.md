---
name: decide
description: Record an architecture or product decision in docs/DECISIONS.md in the house format. Use when choosing between technologies, changing the model, changing a format, or reversing an earlier decision.
argument-hint: "<short decision title>"
---

Record a decision titled: $ARGUMENTS

1. Read `docs/DECISIONS.md`. Find the highest `D-xxx` id; the new one is the next number.
2. Check whether an existing decision already covers this. If the new one changes it, the old entry's status becomes `Superseded by D-xxx` (the only edit allowed to old entries).
3. Append an entry in exactly this shape:
   ```
   ### D-xxx <title in sentence case>
   Status: Proposed (<YYYY-MM-DD>)
   - **Decision:** <one or two sentences, concrete>
   - **Why:** <the forces that make this right now>
   - **Alternatives:** <each with the reason it lost>
   - **Consequences:** <what changes, what to watch, follow-up points>
   ```
4. If the decision changes behaviour described in ARCHITECTURE.md, MODEL.md, DESIGN.md or CLAUDE.md, update those documents in the same change.
5. Show the entry to the user. When they approve, set Status to `Accepted` and commit on a `docs/` branch: `docs(decisions): add d-xxx <slug>`.
