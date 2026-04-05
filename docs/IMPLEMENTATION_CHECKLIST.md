# Photo Project Implementation Checklist

Applying Claude Code best practices patterns to the 77K photo consolidation project.

## 🚀 Phase 1: Date Stamp Detection (Current - Target: April 30)

### Week 1: Foundation & Checkpoints

- [ ] **Define Checkpoints**
  - [ ] Write Checkpoint 1 success criteria (50 samples annotated)
  - [ ] Write Checkpoint 2 criteria (model >75% accuracy)
  - [ ] Write Checkpoint 3 criteria (generalizes to Disc 1)
  - [ ] Write Checkpoint 4 criteria (cross-disc validation)
  - [ ] Add metrics file: `checkpoints.json` with targets

- [ ] **Setup Memory Persistence** (See reference_session_persistence.md)
  - [ ] Create `~/.claude/sessions/` directory
  - [ ] Configure hooks in `~/.claude/settings.json`
  - [ ] Create first session log: `2026-04-03-dating-phase1.tmp`
  - [ ] Document completed work, current state, next steps

- [ ] **Token Optimization**
  - [ ] Review `yolo_finetune/CLAUDE.md` for any monolithic scripts
  - [ ] If scripts >500 lines, plan modular refactor
  - [ ] Verify model selection: Sonnet for training strategy (default)

### Week 2: Annotation & Training (Parallel)

- [ ] **Instance 1: Annotation Pipeline**
  - [ ] Use `/rename "Annotation Pipeline"` in Claude
  - [ ] Continue annotation beyond 50 samples (target: 100-150)
  - [ ] Implement rotation diversity sampling
  - [ ] Document: How many samples in dataset? Coverage by disc?
  - [ ] Update session log with completion metrics

- [ ] **Instance 2: Model Training**
  - [ ] Use `/rename "Model Training"` in Claude
  - [ ] Train YOLO on available samples
  - [ ] Validate at Checkpoint 2 (>75% accuracy)
  - [ ] If fail: Adjust hyperparams, document why
  - [ ] Extract learned pattern: `learned/yolo-cpu-training-recipe.md`
  - [ ] Update session log with accuracy metrics

### Week 3: Edge Cases & Generalization

- [ ] **Checkpoint 2 Review** (Model accuracy)
  - [ ] Run validation script on holdout set
  - [ ] Measure: Accuracy, Precision, Recall
  - [ ] Manual spot-check: 20 random predictions
  - [ ] Document any weak cases (rotations? faint stamps?)

- [ ] **Dataset Augmentation** (If Checkpoint 2 fails)
  - [ ] Extract skill: `learned/rotation-handling-yolo.md`
  - [ ] Add rotated variants to training data
  - [ ] Retrain and revalidate
  - [ ] Update checkpoint criteria if needed

- [ ] **Checkpoint 3: Generalization to Disc 1**
  - [ ] Run inference on full Disc 1 (1,775 photos)
  - [ ] Spot-check 20 random predictions
  - [ ] Verify stamp detection rate ~85%
  - [ ] Document any failure patterns

### Week 4: Cross-Disc & Production Readiness

- [ ] **Checkpoint 4: Cross-Disc Validation**
  - [ ] Run inference on Discs 2-4 (6K total photos)
  - [ ] Spot-check 50 predictions (sample from each disc)
  - [ ] Check for disc-specific issues (scan quality, image size)
  - [ ] Measure: detection rate, edge case failures

- [ ] **Continuous Learning Extraction**
  - [ ] Run `/learn` for each major technique discovered
  - [ ] Extract skills:
    - [ ] Rotation handling
    - [ ] Annotation QC strategy
    - [ ] Training convergence patterns
    - [ ] Edge case validation protocol

- [ ] **Documentation & Handoff**
  - [ ] Update `PLAN.md` with Phase 1 completion
  - [ ] Create `PHASE1_RESULTS.md`: metrics, edge cases, recommendations
  - [ ] Save session logs to `~/.claude/sessions/`
  - [ ] Commit extracted skills to `.claude/skills/learned/`

---

## 📋 Phase 2: Full Consolidation (May - August)

### Pre-Phase Planning

- [ ] **Modular Architecture Design**
  - [ ] Plan module structure: dating/, dedup/, catalog/
  - [ ] Use Explore agent (Haiku) for research
  - [ ] Use Architect agent (Opus) for design
  - [ ] Use Planner agent (Sonnet) for implementation plan
  - [ ] Output: `PHASE2_ARCHITECTURE.md`

- [ ] **Deduplication Algorithm**
  - [ ] Research approaches (hash-based, perceptual, ML)
  - [ ] Define metrics: precision (no false positives), recall (find all dupes)
  - [ ] Write test suite with known duplicates
  - [ ] Extract skill: `dedup-algorithm-testing.md`

- [ ] **Git Worktree Setup** (For 3 parallel instances)
  - [ ] `git worktree add ../project-dating dating-branch`
  - [ ] `git worktree add ../project-dedup dedup-branch`
  - [ ] `git worktree add ../project-catalog catalog-branch`
  - [ ] Use `/rename` for each instance

### Execution (Parallel Instances)

- [ ] **Instance 1: Dedup System**
  - [ ] Implement dedup algorithm
  - [ ] Checkpoint: 100% accuracy on test set
  - [ ] Run on 1K photo sample
  - [ ] Validate no false duplicates

- [ ] **Instance 2: Catalog Builder**
  - [ ] Design photo index schema
  - [ ] Implement indexing pipeline
  - [ ] Checkpoint: Index 1K photos correctly
  - [ ] Validate metadata links

- [ ] **Instance 3: Data Integrity**
  - [ ] Write verification scripts
  - [ ] Create test harness
  - [ ] Checkpoint: Validate checksums, referential integrity
  - [ ] Build diff comparison tools for instance outputs

### Merge & Validate

- [ ] **Merge Worktrees**
  - [ ] Merge dating-branch → main
  - [ ] Merge dedup-branch → main
  - [ ] Merge catalog-branch → main
  - [ ] Resolve any conflicts

- [ ] **Full System Checkpoint**
  - [ ] Run all 77K files through pipeline
  - [ ] Verify: 0 corruption, referential integrity
  - [ ] Spot-check 1% of output (770 photos)
  - [ ] Commit to main with Phase 2 completion notes

---

## 🧠 Memory & Learning Across Phases

### Session Logging (Ongoing)
- [ ] Create session log at start of each work session
- [ ] Update log daily with: completed items, blockers, metrics
- [ ] Save to `~/.claude/sessions/YYYY-MM-DD-topic.tmp`
- [ ] Review at session end, checkpoint completion

### Skill Extraction (Weekly)
- [ ] Run `/learn` after solving non-trivial problems
- [ ] Weekly review: what worked? Extract to skill
- [ ] Organize skills in `~/.claude/skills/learned/`
  - [ ] `rotation-handling-*.md`
  - [ ] `annotation-qa-*.md`
  - [ ] `dedup-algorithm-*.md`
  - [ ] `metadata-schema-*.md`

### Checkpoint Review (Phase-End)
- [ ] Gather all session logs for phase
- [ ] Review metrics progression (accuracy, coverage, quality)
- [ ] Identify patterns that emerged
- [ ] Update `PLAN.md` with lessons learned

---

## 📊 Metrics Tracking

### Phase 1 Success Metrics
```
Checkpoint 1 (Annotation):
✅ 50+ photos labeled
✅ 5+ marked as "no stamp"
✅ 2+ rotated photos included

Checkpoint 2 (Training):
✅ Accuracy ≥ 75% on holdout
✅ Precision ≥ 0.65
✅ Recall ≥ 0.70
✅ Model converges by epoch 50

Checkpoint 3 (Disc 1):
✅ Processes 1,775 photos
✅ ~85% stamp detection rate
✅ 20-photo spot-check all correct

Checkpoint 4 (Cross-Disc):
✅ >80% accuracy on Discs 2-4
✅ No disc-specific failures
✅ Handles image size variations
```

### Phase 2 Success Metrics
```
Checkpoint 1 (Dedup):
✅ 100% accuracy on test set
✅ 0 false positives
✅ Completes 77K files in <4 hours

Checkpoint 2 (Catalog):
✅ All 77K photos indexed
✅ Metadata links valid
✅ Search queries work

Checkpoint 3 (Integrity):
✅ All checksums match
✅ No referential integrity violations
✅ 1% sample spot-check 100% correct
```

---

## 🛠️ Tools & Commands

### Memory Hooks
```bash
# Save session state (PreCompact)
~/.claude/hooks/save-session-state.sh

# Load session context (SessionStart)
~/.claude/hooks/load-session-context.sh

# Finalize session (Stop)
~/.claude/hooks/finalize-session.sh
```

### Instance Management
```bash
# Name instance
/rename "Annotation Pipeline"

# Create worktree
git worktree add ../project-dating dating-branch

# Extract learned pattern
/learn "Pattern description"

# Start with system prompt injection
claude --system-prompt "$(cat ~/.claude/contexts/photo-project.md)"
```

### Validation
```bash
# List recent session logs
ls -lt ~/.claude/sessions/ | head -10

# View current session metrics
cat ~/.claude/sessions/YYYY-MM-DD-topic.tmp

# Check learned skills
ls ~/.claude/skills/learned/ | grep photo-project
```

---

## 📝 Status Tracking

### Current Phase: 1 (Dating Stamp Detection)
- **Start Date**: 2026-04-03
- **Target Completion**: 2026-04-30
- **Current Checkpoint**: 1 (Annotation)
- **Next Action**: Expand from 50 → 100 samples, train model

### Previous Sessions
- [ ] 2026-04-03: Initial setup, 50 samples annotated (Session log: YYYY-MM-DD-dating-phase1.tmp)

### Upcoming Milestones
- 2026-04-15: Checkpoint 2 (Model >75% accuracy)
- 2026-04-22: Checkpoint 3 (Disc 1 generalization)
- 2026-04-30: Checkpoint 4 (Cross-disc validation)
- 2026-05-01: Phase 1 complete, begin Phase 2

---

**Last Updated**: 2026-04-03
**Maintainer**: Will Pike
**References**: CLAUDE_CODE_WORKFLOW.md, PLAN.md, HANDOFF.md
