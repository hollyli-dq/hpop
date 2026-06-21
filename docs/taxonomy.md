# Taxonomy

The linguistically-grounded, cognitive-science-backed taxonomy. This is the conceptual inventory;
the machine-actionable versions live in `rules/` (`phases.yaml`, `skills.yaml`).

> **Ownership:** authored / curated by the linguistics PhD. Sections marked _TBD_ are placeholders
> awaiting that content.

---

## Phases (cognitive-science-grounded categories)

A **phase** is a category tag on a skill instance — the "part-of-speech" of a skill. Phases come
from the cognitive-science literature on how agents solve complex tasks. They guide the skill
inventory and the annotation rules, and they define the **phase grammar** of a workflow (which
phases may precede which).

Each phase entry should define:

- **id / name**
- **cognitive definition** + literature grounding (citation)
- **purpose** — what the agent is accomplishing in this phase
- **typical realizing skills** — which skill types fill this phase
- **precedence constraints** — which phases may/may not precede it

_TBD — candidate phases (to be confirmed against the literature):_

| Phase | Working definition (placeholder) |
|-------|----------------------------------|
| Orient | Establish goal and context from the task / initial state. |
| Explore | Gather information; search/navigate to locate what's needed. |
| Plan | Decide the approach / sub-goals. |
| Act | Perform the substantive task operations. |
| Verify | Check results against the goal; detect/repair errors. |
| Deliver | Produce and hand off the final output. |

---

## Skill types (the reusable skill inventory)

A **skill type** is a reusable procedure pattern. A **skill instance** is one invocation of it.
Each skill type carries a **phase** tag.

Open question (DESIGN.md §7): is the inventory a **closed** taxonomy authored by the PhD,
**induced** from data, or **hybrid**? Recorded here once decided.

Each skill type entry should define:

- **id / name**
- **phase** (its category tag)
- **definition** — the reusable procedure it captures
- **typical canonical action tokens** it is composed of
- **local partial-order template** — which steps must precede which within the skill
- **completion / boundary rule** — when an instance ends (success or failure)

_TBD — populated as the inventory is defined._

---

## Canonical action tokens

The minimal meaningful action units. Open question (DESIGN.md §7): fixed vocabulary vs. open set.

_TBD — token vocabulary / labeling scheme._
