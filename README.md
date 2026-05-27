# Email Agent — Active Learning

## Project structure

```
email-agent/
├── agent/
│   ├── main.py          ← orchestrator, entry point
│   ├── classifier.py    ← active-learning ML model
│   ├── imap_utils.py    ← IMAP connection + email parsing
│   └── unsub_agent.py   ← brand ID, memory, unsubscribe execution
├── data/                ← created on first run
│   ├── labelled_emails.json   ← training set
│   ├── model.pkl              ← trained sklearn pipeline
│   ├── training_stats.json    ← auto/asked/accuracy metrics
│   └── brand_decisions.json   ← brand choices
├── dashboard.html       ← open in Safari on iPhone
└── requirements.txt
```

## Setup (5 minutes)

### 1. Install dependencies (in a-Shell on iPhone)
```bash
pip install scikit-learn numpy requests beautifulsoup4
```

### 2. Copy files to iCloud Drive
```
iCloud Drive/
└── Shortcuts/
    └── email-agent/
        └── agent/ 
```

### 3. Set credentials
Create `~/Documents/email_agent.env`:
```bash
IMAP_HOST=imap.yourprovider.com
IMAP_PORT=993
IMAP_USER=you@example.com
IMAP_PASS=yourpassword
LIVE_UNSUB=false   #Set true when ready to unsub
```

### 4. Run manually first
```bash
cd ~/Documents/email-agent/agent
source ~/Documents/email_agent.env
python3 main.py classify
```

### 5. Create the two Shortcuts (same as v1 guide)
- **📬 Classify** → runs every 2h automatically
- **🚫 Unsubscribe** → tap manually when you want to review ads

---

## How active learning works in practice

### Zone 1 (0–30 labelled emails): Always asks
```
  ┌─ From   : promo@decathlon.com
  │  Subject: Offre spéciale running
  │  (no model yet — please label this email)
  │
  │  [1] travel  [2] bills  [3] jobs  [4] personal  [5] ads
  └▶ 5
  → [ADS    ] confirmed ✓  Offre spéciale running
```

### Zone 2 (30–80 labelled): Asks only when unsure
```
  ┌─ From   : newsletter@lemonde.fr
  │  Subject: La matinale du Monde
  │  Guess  : ads [████████░░░░] 67%  ← confidence shown
  │
  │  [1] travel  [2] bills  [3] jobs  [4] personal  [5] ads
  │  [↵ Enter] confirm guess (ads)
  └▶ ↵
  → [ADS    ] confirmed ✓  La matinale du Monde
```

### Zone 3 (80+ labelled): Silent and autonomous
```
  → [BILLS  ] auto [████████████] 94%  Votre facture EDF
  → [ADS    ] auto [███████████░] 91%  Vente flash Nike
  → [JOBS   ] auto [██████████░░] 88%  New job matches
```

---

## Admin commands

```bash
python3 main.py status              # print model health + brand memory
python3 main.py forget Decathlon    # reset a brand decision
python3 main.py classify            # classifier only
python3 main.py unsub               # unsubscribe agent only
```

## Privacy

- No email content ever leaves your iPhone (the sklearn model runs 100% locally)
- All model data (`data/`) is plain JSON/pickle — you can inspect or delete it any time

## Cost

| Component | Cost |
|---|---|
| Email classification | Free (sklearn, on-device) |
| Unsubscribe agent | Free |
| Total per month |€0 |
