# Photo Project Documentation Index

Complete guide to where information lives and how to use it.

## 📚 Core Documentation (Read These First)

| File | Purpose | Status |
|------|---------|--------|
| `CLAUDE.md` | Project constraints, environment, critical rules | ✅ Active |
| `docs/PLAN.md` | Master plan: Phases 1-3, timeline, success criteria | ✅ Active |
| `docs/HANDOFF.md` | Detailed handoff for date extraction pipeline | ✅ Active |
| `docs/DATE_EXTRACTION_APPROACHES.md` | Analysis of 6 approaches with costs | ✅ Complete |

## 🎯 Claude Code Patterns (New - Read Before Major Work)

| File | When to Use |
|------|-------------|
| `docs/CLAUDE_CODE_WORKFLOW.md` | Before starting any new session -- refresher on best practices |
| `docs/IMPLEMENTATION_CHECKLIST.md` | Weekly -- track Phase 1 and Phase 2 progress |
| `~/Documents/CLAUDE_CODE_REFERENCE.md` | Quick lookup for patterns, model selection, checkpoints |

## 🧠 Memory Files (Context Persistence)

Located in `~/.claude/memory/`:

| Memory | Topic | Type |
|--------|-------|------|
| `feedback_token_optimization.md` | Cost reduction via Haiku/Sonnet/Opus delegation | Feedback |
| `feedback_subagent_orchestration.md` | Sequential phases, iterative refinement | Feedback |
| `reference_session_persistence.md` | Hook setup, session logs, cross-session memory | Reference |
| `feedback_verification_checkpoints.md` | Checkpoint gates, photo project phases | Feedback |
| `feedback_parallelization_worktrees.md` | Parallel instances, git worktrees, cascade method | Feedback |
| `feedback_continuous_learning.md` | Skill extraction, /learn command, reflection | Feedback |

**Access**: Automatically loaded in Claude Code. Stored in `MEMORY.md` index.

## 📋 YOLO Fine-Tune Subdirectory

| File | Purpose |
|------|---------|
| `yolo_finetune/CLAUDE.md` | YOLO training config, architecture, classes |
| `yolo_finetune/annotate.py` | HTTP annotation server |
| `yolo_finetune/train.py` | YOLO fine-tuning script |
| `yolo_finetune/index.html` | Browser-based annotation UI |
| `yolo_finetune/dataset/` | YOLO-format training data |

## 🗺️ How to Navigate

### Scenario 1: Starting a New Session
1. Read: `CLAUDE.md` (constraints, environment)
2. Check: `~/.claude/sessions/` (load recent session context)
3. Review: `IMPLEMENTATION_CHECKLIST.md` (progress + next steps)
4. Reference: `~/Documents/CLAUDE_CODE_REFERENCE.md` (patterns quick guide)

### Scenario 2: Implementing a Feature
1. Check: `IMPLEMENTATION_CHECKLIST.md` (which phase? which instance?)
2. Read: Relevant memory file (token optimization, subagent pattern, etc.)
3. Define: Checkpoints before starting (from `feedback_verification_checkpoints.md`)
4. Implement: Follow phase orchestration pattern
5. Extract: `/learn` any non-trivial techniques discovered

### Scenario 3: Debugging or Stuck
1. Check: Relevant memory file (e.g., continuous learning, verification)
2. Review: Recent session logs in `~/.claude/sessions/`
3. Check: Extracted skills in `~/.claude/skills/learned/`
4. Escalate: Ask Claude to review, extract new skill

### Scenario 4: End of Session
1. Save: Session log in `~/.claude/sessions/YYYY-MM-DD-topic.tmp`
2. Extract: Any learned patterns via `/learn`
3. Update: `IMPLEMENTATION_CHECKLIST.md` with progress
4. Commit: Learned skills to git if non-trivial

## 🔍 What's in Memory vs What's in Code

### In Git (Code)
- Project structure
- Scripts, utilities
- Tests, validation
- Configuration files
- Learned skills (after extraction)

### In Memory (Context)
- User preferences
- Non-obvious decisions
- External references
- Long-term constraints
- Cross-session context

### In Session Logs (Temporary)
- Daily progress
- Metrics snapshot
- Current blockers
- Decisions made this session

**Rule**: When in doubt, keep it in git. Only memory-persist what won't change and helps future sessions.

## 📊 Progress Tracking

### Checkpoints (From IMPLEMENTATION_CHECKLIST.md)

**Phase 1: Date Stamp Detection** (Target: Apr 30)
- [ ] Checkpoint 1: 50 samples annotated
- [ ] Checkpoint 2: Model >75% accuracy
- [ ] Checkpoint 3: Generalizes to Disc 1
- [ ] Checkpoint 4: Works across all discs

**Phase 2: Full Consolidation** (Target: Aug 31)
- [ ] Checkpoint 1: Dedup algorithm validated
- [ ] Checkpoint 2: Catalog built
- [ ] Checkpoint 3: Data integrity verified

### Metrics to Track
- Model accuracy, precision, recall (Phase 1)
- Dedup false positive rate (Phase 2)
- Processing time per 1K files
- Edge cases discovered and fixed

## 🛠️ Quick Commands

```bash
# View recent sessions
ls -lt ~/.claude/sessions/ | head -10

# Extract a learned pattern
/learn "Pattern name"

# Review memory files
cat ~/.claude/memory/MEMORY.md

# Create a worktree for parallel work
git worktree add ../project-feature feature-branch

# Name current Claude instance
/rename "Annotation Pipeline"

# Check Claude Code settings
cat ~/.claude/settings.json | grep hooks -A 10
```

## 📞 When to Reach Out

Document and commit:
- New checkpoints discovered
- Major architectural changes
- Integration with external services
- Learned patterns/skills
- Status milestones

## 🔄 Documentation Update Cycle

| Trigger | Action |
|---------|--------|
| Session end | Update session log, commit if major change |
| Checkpoint pass/fail | Update IMPLEMENTATION_CHECKLIST.md |
| Learned pattern | Extract skill, update MEMORY.md |
| Architecture change | Update PLAN.md, create design doc |
| Phase complete | Create PHASEХ_RESULTS.md summary |

## 📚 External References

- Claude Code Official Docs: https://code.claude.com/docs/
- Photo Project Master Plan: `docs/PLAN.md`
- YOLO Fine-Tune Design Spec: `docs/superpowers/specs/2026-04-03-bbox-annotator-yolo-finetune-design.md`
- Claude Code Best Practices: `docs/CLAUDE_CODE_WORKFLOW.md`

---

**Last Updated**: 2026-04-03
**Maintainer**: Will Pike
**Next Review**: When Phase 1 Checkpoint 2 completes (target: 2026-04-15)
