# Competition Rules

## Submissions

- A submission is a single Python file of at most 1 MiB.
- It runs in an isolated sandbox with 1 CPU, 2 GB of memory, no network access and a 15-minute time limit.
- Some imports and builtins are banned, among them `os`, `sys`, `subprocess`, `socket`, `io`, `pathlib`, `open`, `getattr`, `type`, dunder attributes, `str.format`, private modules, and anything from `simulator` other than the public models and `BaseStrategy`. Banned code is rejected before it runs.
- Only one submission can be evaluated at a time. You can upload the next one once it finishes.
- A failed submission never replaces a working final one. If a submission fails, its detail shows the error and traceback.
- Any attempt to get around the sandbox or its restrictions means disqualification.

## Simulation

- Latency is simulated and is always 5.0 ms.
- The position limit is ±100 and is a hard limit.

## How the competition works

The competition has 4 rounds of increasing difficulty. Each round works the same way:

1. Develop your strategy locally on the round's public backtest, using the simulator and explorer from the GitHub repository.
2. Once you're satisfied, upload your Python strategy file on the dashboard.
3. We evaluate it on the same public backtest data you have locally. This only confirms that your code runs correctly in the server sandbox.
4. You can make up to 10 submissions per round. One of them must be marked as final: by default it's your latest successful submission, and you can pick a different one on the dashboard until submissions close.
5. After submissions close, all final strategies are evaluated on the private backtest and the leaderboard for that round is released.

The last round is different: no leaderboard is released. The final standings are announced at the closing ceremony and on Discord.

## How the winner is decided

Your final score combines your private-backtest PnL from all four rounds:

```
Score = 0.8·z1 + 1.0·z2 + 1.0·z3 + 1.2·z4
```

- `zᵣ` is your PnL in round `r` standardized against the other teams in that round: how many standard deviations you are above or below that round's average.
- Standardizing puts every round on the same scale, so a round with large PnLs doesn't outweigh the others. Later rounds count more.
- Teams with zero or negative PnL in a round (including teams without a final submission) are left out of that round's average and standard deviation, and score as if their PnL were 0.
- The team with the highest score wins.

## Git repository

Every team must keep its work in a git repository and share it with us. This is non-negotiable.

- Commit regularly, at least every 30 minutes, even when you haven't improved the strategy. We want to see your work and research, not just the result.
- Commit failed experiments too, with a short note on what you tried and what came out of it.
- Don't squash, rebase or force-push: we review the history as it happened.
- Keep your research reusable across rounds. A mechanic that doesn't help in round N might still improve your strategy in round N+1, so keep it behind a parameter or flag instead of deleting it.
- Invite this GitHub user to your repository with Read access as soon as possible, ideally before round 1 starts: `antoninkriz`

Qminers reviews every team's repository. Even if your team doesn't finish in the top 3, high-quality research gets noticed and we may reach out to you.
