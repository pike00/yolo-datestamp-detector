# Claude Code Workflow Patterns

Extracted from "The Longform Guide to Everything Claude Code" by @affaanmustafa (Jan 2026).
Adapted for the photo consolidation project.

## Token Optimization

### Primary Strategy: Subagent Architecture

Delegate to the cheapest capable model:
- **Opus 4.6**: Architectural decisions, security-critical code, multi-file refactors, first-attempt failures
- **Sonnet 4.6**: Default for 90% of coding tasks
- **Haiku 4.5**: Repetitive work, clear instructions, "worker" in multi-agent setups

**Cost Math**: Haiku vs Opus is ~5x difference. Sonnet vs Opus is ~1.67x. Haiku+Opus combo makes more sense than Sonnet-heavy.

Example agent definition (in skill YAML):
```yaml
---
name: quick-search
description: Fast file search
tools: Glob, Grep
model: haiku
---
```

### Modular Codebase = Lower Token Cost

Leaner code = fewer tokens to read = cheaper and faster:
- Files in hundreds of lines, not thousands
- Reusable utilities and functions reduce duplication
- Modular structure prevents reading entire monoliths on simple changes
- Intermediate tool calls for reading are cheaper than one massive read

Recommended structure:
```
src/
├── modules/          # Self-contained domains
│   ├── dating/       # Date stamp extraction
│   │   ├── api/      # Public interface
│   │   ├── domain/   # Business logic
│   │   ├── infrastructure/  # DB, file I/O
│   │   └── tests/
│   ├── dedup/        # Deduplication logic
│   └── catalog/      # Catalog management
├── shared/           # Deeply generic helpers
└── main.py          # Bootstrap
```

### Background Processes

Run long operations outside Claude, summarize output:
- Don't stream all output to Claude
- Run `train.py` in tmux, check results later
- Reduces input tokens (cheaper) vs output tokens (more expensive)
- Use `TaskOutput` tool to check async results, not real-time streaming

## Memory Persistence

### Session Log Pattern

Save state at logical intervals:
- **Location**: `~/.claude/sessions/YYYY-MM-DD-topic.tmp`
- **Content**: Current state, completed items, blockers, key decisions, context for next session
- **Trigger**: Manual after each session or via hooks

Example structure:
```markdown
## Session: 2026-04-03 YOLO Annotation Feedback Loop

### Completed
- ✅ Annotated 50 sample photos in feedback.py correct mode
- ✅ Implemented model prediction loading
- ✅ Color-coded annotations (yellow=model, green=user)

### Current Work
- [ ] Train model on 50 annotated samples
- [ ] Evaluate accuracy on holdout set

### Blockers
- None currently

### Key Decisions
- Using stratified sampling for initial 50 samples
- Single-class detector (date stamp region only)

### Context for Next Session
- Model accuracy ~72% after 20 epochs
- Need ~200 samples for production quality
- Rotation handling critical for edge-stamped photos
```

### Memory Hook Configuration

In `~/.claude/settings.json`:
```json
{
  "hooks": {
    "PreCompact": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/save-session-state.sh"
      }]
    }],
    "SessionStart": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/load-session-context.sh"
      }]
    }],
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/finalize-session.sh"
      }]
    }]
  }
}
```

## Parallelization Patterns

### Two-Instance Kickoff

For major work:
- **Instance 1 (Left)**: Scaffolding agent — project structure, configs, conventions
- **Instance 2 (Right)**: Research agent — documentation, architecture, references

Naming pattern: Use `/rename "Instance Name"` to track parallel work clearly.

### Git Worktree Strategy

For 2+ parallel tasks:
```bash
# Create isolated worktrees
git worktree add ../project-phase2 phase2-branch
git worktree add ../project-refactor refactor-branch

# Each gets its own Claude instance
cd ../project-phase2 && claude
```

Benefits:
- No git conflicts
- Each has clean working directory
- Easy to compare outputs/approaches
- Can benchmark same task with different strategies

### Cascade Method

Managing multiple instances:
- Open new tasks in new tabs **to the right**
- Sweep left to right (oldest to newest)
- Maintain consistent direction
- Focus on 3-4 tasks max (more = overhead > productivity)

## Subagent & Orchestration Patterns

### Sequential Phase Orchestration

Structure multi-step work:

```
Phase 1: RESEARCH (Explore agent)
  └─ Output: research-summary.md

Phase 2: PLAN (Planner agent)
  └─ Input: research-summary.md
  └─ Output: plan.md

Phase 3: IMPLEMENT (TDD agent)
  └─ Input: plan.md
  └─ Output: code changes

Phase 4: REVIEW (Code-reviewer agent)
  └─ Input: code changes
  └─ Output: review-comments.md

Phase 5: VERIFY
  └─ Run tests, fix issues
  └─ Done or loop back
```

### Iterative Retrieval for Sub-agents

Sub-agents lack orchestrator context. Fix with iteration:

```
Orchestrator dispatches → Sub-agent returns summary
                              ↓
                        Orchestrator evaluates
                              ↓
                        "Is this sufficient?"
                         ↙         ↘
                       no          yes
                        ↓          ↓
                  Follow-ups   [ACCEPT]
                        ↓
                  Sub-agent fetches
                  answers & returns
                        ↓
                  Loop max 3x
```

Rules:
- Pass both query AND objective context
- Max 3 follow-up cycles to prevent infinite loops
- Evaluate every return before accepting

## Verification & Evaluation

### Checkpoint-Based Evals

For linear workflows with clear milestones:

```
[Task 1] → [Checkpoint #1] → pass? → [Task 2]
                ↓
              fail ──→ fix ──→ loop
```

Use when: Feature implementation with defined stages.

### Continuous Evals

For long-running exploratory work:

```
[Work] → [Timer/Change] → [Run tests + lint]
              ↓
          ┌───┴───┐
        pass     fail
          ↓       ↓
      [Continue] [Stop & fix]
```

Use when: Refactoring, maintenance, long sessions.

### Eval Metrics

- **pass@k**: At least ONE of k attempts succeeds. Use when you need it to work.
- **pass^k**: ALL k attempts succeed. Use when consistency is essential.

Example:
```
k=1: 70%  k=3: 91%  k=5: 97%  (pass@k — odds improve)
k=1: 70%  k=3: 34%  k=5: 17%  (pass^k — gets harder)
```

### Benchmarking Workflow

Compare same task with/without a technique:

```
         [Same Task]
              │
    ┌─────────┴──────────┐
    ▼                    ▼
 Worktree A          Worktree B
 WITH skill         WITHOUT skill
    │                    │
    └────────┬───────────┘
             ▼
        [git diff]
             ↓
  Compare: logs, tokens, quality
```

## Skill & Command Patterns

### Extracting Patterns into Skills

When you solve a non-trivial problem:
1. Use `/learn` mid-session to capture immediately
2. Or let `Stop` hook evaluate session and suggest learned skills
3. Store in `~/.claude/skills/learned/`

Patterns worth extracting:
- Debugging techniques
- Project-specific workarounds
- Consistent error resolutions
- Domain knowledge

### Continuous Learning via Stop Hook

Hook that analyzes sessions for learnable patterns:

```bash
# Runs at session end
# Looks for non-trivial solutions
# Drafts skill files
# Saves to ~/.claude/skills/learned/
```

## Advanced Context Patterns

### Strategic Context Compaction

Instead of auto-compacting (mid-task):
1. Disable auto-compact in settings
2. Compact manually at phase transitions
3. Use PreToolUse hook to suggest after N tool calls

Timing:
- After exploration phase, before implementation
- After completing major milestone
- When transitioning between agents

### Dynamic System Prompt Injection

Inject context at CLI time, not in conversation:

```bash
# Higher authority than @ file references
claude --system-prompt "$(cat memory.md)"

# Scenario-specific aliases
alias claude-dev='claude --system-prompt "$(cat ~/.claude/contexts/dev.md)"'
alias claude-review='claude --system-prompt "$(cat ~/.claude/contexts/review.md)"'
```

Why: System prompt has higher authority than user messages, which beats tool results. For strict behavioral rules or critical context, this ensures proper weighting.

## MCP & Tool Optimization

### Replaceable MCPs

Many MCPs wrap CLIs you already have. Consider CLI + skills instead:

- **GitHub MCP** → `/gh-pr` command wrapping `gh pr create`
- **Supabase MCP** → Skills using Supabase CLI directly
- **Vercel MCP** → Skills wrapping `vercel` CLI

Benefits:
- Frees context window (MCPs are lazy-loaded now, less critical)
- Reduces token usage (CLI operations outside context)
- Same functionality, lower cost

### Tool Selection for Tasks

Replace built-in tools strategically:
- **Grep**: Use `mgrep` for ~50% token reduction vs ripgrep
- **Filesystem**: Use `Glob` + `Read` instead of `find` + `cat`
- **Search**: Use `Grep` files_with_matches mode instead of content mode when you only need paths

## Photo Project Specific Applications

### Phase 1: Date Stamp Detection (Current)

Apply these patterns:

1. **Token Optimization**
   - Use Haiku for annotation tool improvements (clear, repetitive)
   - Use Sonnet for YOLO training strategy (default choice)
   - Use Opus only if training fails or architecture needs rethinking

2. **Memory Persistence**
   - Save session logs after annotation milestones
   - Track model accuracy progression across sessions
   - Document dataset challenges (rotation, missing stamps)

3. **Verification**
   - Checkpoint 1: 50 annotated samples
   - Checkpoint 2: Model achieves >80% accuracy on holdout
   - Checkpoint 3: Model generalizes to full disc 1 (1,775 photos)
   - Continuous: Run validation on each epoch

4. **Parallelization** (Future)
   - Instance 1: Training loop improvements
   - Instance 2: Annotation pipeline expansion
   - Use worktrees for competing annotation strategies

### Phase 2: Full Consolidation

Apply these patterns:

1. **Modular Architecture**
   - `modules/dating/` — Stamp detection + extraction
   - `modules/dedup/` — Duplicate detection + removal
   - `modules/catalog/` — Metadata enrichment + indexing

2. **Subagent Strategy**
   - Explore agent: Assess dedup approaches, storage architectures
   - Architect agent: Design modular structure
   - Worker agents (Haiku): Run dedup, catalog operations
   - Reviewer agent: Validate data integrity

3. **Background Processes**
   - Large batch operations run in tmux, results checked async
   - Don't stream 77K file hashes to Claude
   - Summarize: "Processed 12,547 files. 2,341 duplicates found."

## Continuous Learning

Track patterns that work for this project:
- Photo rotation detection techniques
- Effective annotation strategies
- Dedup heuristics that pass validation
- Successful training hyperparameters

Update memory files as you discover project-specific wisdom.

---

**References**:
- Anthropic: "Claude Code Best Practices" (Apr 2025)
- @affaanmustafa: "The Longform Guide to Everything Claude Code" (Jan 2026)
- Project: `PLAN.md`, `HANDOFF.md`
