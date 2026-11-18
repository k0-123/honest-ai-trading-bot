# Running the daily paper trade on GitHub Actions

The workflow `.github/workflows/daily-paper.yml` runs `nsebot daily` every **Monday–Friday at 11:00 UTC (4:30 PM IST)**. It never places real orders.

Each run:
1. Restores the price cache, then installs the bot.
2. Runs the four paper books: v1 baseline, NIFTYBEES buy-and-hold, v2a and v2b. The AI book stays **off** for now (see step 5).
3. Commits `state/` (portfolio, trades, signals, equity, run log) and `reports_out/daily` and `reports_out/weekly` back to the repo, even when the run fails.
4. **Fails** if `nsebot daily` failed **or** any book is HALTED. GitHub then emails you.

NSE holidays and weekends are handled automatically: the run finishes with "Nothing to do".

---

## One-time setup (about 5 minutes)

### 1. Merge the pull requests into `main`
GitHub only runs scheduled workflows from the default branch. Merge PR #2 (v2 books) and then the forward-paper PR (this workflow) into `main`.

### 2. Let the workflow save its results
**Settings → Actions → General → Workflow permissions →** choose **"Read and write permissions"** → **Save**.

### 3. Add your API key as a secret (for later, when you turn the AI book on)
**Settings → Secrets and variables → Actions → Secrets tab → New repository secret**
- Name: `GOOGLE_API_KEY`
- Secret: your Gemini API key

The key lives only in GitHub Secrets. It is never written to the repo or the logs, and it is only passed to the run when the AI book is on.

### 4. Turn the workflow on and test it once
1. Open the **Actions** tab. If GitHub asks, click **"I understand my workflows, go ahead and enable them"**.
2. Click **"Daily paper run"** in the left list.
3. Click **"Run workflow" → "Run workflow"**.
4. After a few minutes you should see a green check and a new commit called "Daily paper run YYYY-MM-DD [skip ci]".

From now on it runs by itself every weekday at 4:30 PM IST. GitHub sometimes starts scheduled runs a few minutes late, which is fine because the bot uses that day's closing data.

### 5. Turn the AI book on (only after 5 successful runs)
The workflow counts successful runs in `state/actions_runs.log`. The AI book stays off until **both** of these are true:
- at least **5** runs with `outcome=success` are in that log, **and**
- you set **Settings → Secrets and variables → Actions → Variables tab → New repository variable**: Name `ENABLE_AI_BOOK`, Value `true`.

To turn it off again, delete the variable or set it to `false`.

---

## Emails and failures

**Getting the email:** GitHub emails failed scheduled runs to the person who last changed the workflow's schedule. Check **your profile picture → Settings → Notifications → Actions** and tick email for failed workflows. Then confirm once with step 4.

**A failed run** means one of two things:
- **The run itself failed** (e.g. Yahoo down). Open the run, then the **"Daily paper run"** step, to see the error. The next weekday's run catches up the missed sessions automatically.
- **A book is HALTED.** Its kill-switch fired: 10% drawdown for v1, 15% for v2a and v2b (see README §6d). The workflow keeps failing every day until you reset that book by hand, so you don't miss it. To reset, on your computer:

  ```bash
  git pull
  ```

  ```bash
  nsebot reset-halt --book v2b_momentum --confirm
  ```

  ```bash
  git add -f state && git commit -m "Reset halt on v2b_momentum" && git push
  ```

  Replace `v2b_momentum` with the book named in the error.

---

## Yearly chore: NSE holidays

`config.yaml` → `nse_holidays` holds NSE's official trading holidays. Only **2026** is listed, because NSE publishes the next year's list around December. When the 2027 list is out, add its weekday dates.

Until then, every run prints `WARNING: ... no 2027 dates`. The bot still runs, but v2b may rebalance one session late around a month-end holiday.

## Where to look

| File | What it shows |
|---|---|
| `reports_out/daily/YYYY-MM-DD.md` | Each book's equity, positions, pending orders and the day's signals with reasons |
| `reports_out/weekly/` | Every Friday: v1 vs v2a vs v2b vs NIFTYBEES (needs 20+ sessions before it means anything) |
| `state/trades.csv` | Every simulated fill, with its reason |
| `state/actions_runs.log` | One line per workflow run (date, outcome, AI on/off) |

## Cost

Each run takes a few minutes. Private repositories on the free plan include 2,000 Actions minutes a month, which is plenty for one run per weekday.
