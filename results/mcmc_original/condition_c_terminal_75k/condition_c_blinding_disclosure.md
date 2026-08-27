# Condition C — blinding disclosure

Recovery analysis was sealed for the duration of the run and opened only after
Condition C terminated.

**Incidental recognition.** During truth-free chain-to-chain diagnostics
performed between registered checkpoints, canonical `H` hash prefixes were
incidentally recognised from the earlier Condition B report, which had been read
in the same working session.

**What did and did not follow from that.**

- No truth artifact was deliberately opened at that time.
- No recovery metric was computed before termination.
- No sampler, target, proposal scale, cadence, gate, checkpoint schedule or
  stopping rule was altered as a result.
- Every gate and metric was already frozen in committed code; the recognition
  could not have influenced them.
- Formal recovery was opened only after the termination decision, and uses only
  the metrics predefined before unsealing.

The disclosure is recorded here rather than omitted, because the alternative —
discovering it later in review — would be worse.
