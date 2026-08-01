# Long-running optimization

Long runs use the same V1.4 model as short runs. There is no separate long-run engine. ChatGPT retains optimization judgment; Invocation records make repeated operations observable and recoverable.

## User authorization

The user may bound elapsed work, GPU use, modification paths, risk, host actions, or the furthest validation layer. This is an authorization boundary, not a target to consume. Work should stop early when evidence is conclusive or expected value is low.

Normal runs should need at most one planned authorization question after initial analysis. Unattended runs ask none within the granted scope. A new question is reserved for a material change in time, GPU cost, risk, host impact, or modification scope.

## Dynamic investment

Before an expensive operation, ChatGPT compares:

- the removable-time ceiling and minimum useful effect;
- evidence for and against the current mechanism;
- the specific uncertainty the operation can resolve;
- measured build, correctness, benchmark, and profiler costs in the current environment;
- implementation and validation difficulty;
- remaining user authorization.

The estimate should be a range with evidence, not a precise invented duration. If more work is valuable but outside authorization, ChatGPT reports the current evidence and asks once for the additional scope. It does not classify authorization exhaustion as a failed performance direction.

## Invocation lifetime

Each external operation records a request, events, heartbeat, result, elapsed time, stop reason, and cleanup status. Individual command and operation timeouts terminate the process group. SSH or foreground disconnection does not erase the record; use the operation's `status` to inspect the terminal result.

If cleanup cannot be confirmed, conflicting work on the same GPU remains blocked until recovery verifies that no declared task is live. Retrying is explicit, and a completed equivalent result is reused instead of launching duplicate work.

Progress updates should state what is running, what evidence is expected, and the next review point. A long period without a visible heartbeat is a task fault, not normal optimization behavior.
