# EfficientZero Monte Carlo Tree Search

## Algorithm overview
EfficientZero maintains a **policy network** (the *network prior*) and uses **MCTS** to compute an *improved* **target policy** from visit counts.

- **Network policy / prior**: $\pi_\theta(a \mid z)$ — a feed-forward policy head that outputs a distribution over actions given a latent state $z$.
- **MCTS target policy**: $\pi_{\text{target}}(a \mid z)$ — the policy derived from MCTS visit counts at a node (typically proportional to child visit counts).

> **Important:** The *search policy* is $\pi_{\text{target}}$ (from MCTS), not $\pi_\theta$. MCTS uses $\pi_\theta$ as a prior.

---

## Data collection and evaluation
During environment interaction, MCTS is run at the current root state to produce:
- a selected action $a_t$,
- a root value estimate $\hat{V}^{\text{MCTS}}(s_t)$,
- and a root target policy $\pi_{\text{target}}(\cdot \mid z_t)$.

The environment transition is:

$s_t \xrightarrow[\text{MCTS uses }\pi_\theta]{\text{search at root}} 
\Big(a_t,\; \hat{V}^{\text{MCTS}}(s_t),\; \pi_{\text{target}}(\cdot\mid z_t)\Big)
\xrightarrow{\text{env}} (s_{t+1}, r_{t+1})$

The replay buffer stores tuples such as:

$\big(s_t,\; a_t,\; r_{t+1},\; \hat{V}^{\text{MCTS}}(s_t),\; \pi_{\text{target}}(\cdot\mid z_t)\big)$

---

## Training (policy part)
A contiguous chunk of length $T$ is sampled from the replay buffer. If the sample is selected for **re-analyze**, MCTS is re-run using current parameters to refresh $\hat{V}^{\text{MCTS}}$ and $\pi_{\text{target}}$.

The policy head is trained to match the MCTS target policy using **cross-entropy** (not KL):

$L_{\text{root}} \;=\; \sum_{t=t'}^{t'+T-1} 
\mathrm{CE}\Big(\pi_{\text{target}}(\cdot \mid z_t),\; \pi_\theta(\cdot \mid z_t)\Big)$

where

$\mathrm{CE}(p,q) \;=\; -\sum_a p(a)\log q(a)$

---

# Idea: reuse internal MCTS nodes during re-analyze

## Definition: qualifying node (formerly “qualifying subtree root”)
When we **re-analyze** a sample at root state $s$, we build an MCTS tree rooted at $z$ (the latent for $s$).  
A **qualifying node** $h$ is any internal node in this reanalyzed search tree whose **visit count** exceeds a threshold:

$\mathcal{Q}(s) \;=\; \Big\{\, h \in \text{Tree}(s) \;:\; N(h) \ge \texttt{min\_visits} \,\Big\}$

- $h$ denotes a **tree node** (a particular latent-state reached in the search tree).
- $N(h)$ denotes the **visit count** of node $h$ produced by MCTS.
- $z_h$ denotes the **latent state** stored at node $h$.

For each qualifying node $h$, we define its **node-level target policy** from child visit counts:

$\pi_{\text{target}}(a \mid z_h) \;\propto\; N(h,a)$

where $N(h,a)$ is the visit count of child edge/action $a$ from node $h$, and the distribution is normalized over actions.

> Intuition: a “qualifying subtree root” is simply a **qualifying internal node** $h$; the “subtree” is the portion of the MCTS tree below $h$.

---

## What we store / use during re-analyze
This subtree distillation is applied **only when re-analyze is performed**, so we reuse the freshly built MCTS tree.

From the reanalyzed tree rooted at $s$, in addition to the root targets, we also extract:

$\{(z_h,\; \pi_{\text{target}}(\cdot\mid z_h)) : h \in \mathcal{Q}(s)\}$

No additional MCTS runs are needed at $h$; the targets come directly from the already-built reanalyzed tree.

---

## Subtree policy loss (notation clarity + cross-entropy)
We add an auxiliary policy loss that distills the MCTS target policy at qualifying internal nodes:

$L_{\text{subtree}}
\;=\;
\sum_{s \in \mathcal{B}}
\;\frac{1}{|\mathcal{Q}(s)|}
\sum_{h \in \mathcal{Q}(s)}
\mathrm{CE}\Big(\pi_{\text{target}}(\cdot\mid z_h),\; \pi_\theta(\cdot\mid z_h)\Big)$

- $\mathcal{B}$ is the set of root states in the minibatch being reanalyzed.
- The $\frac{1}{|\mathcal{Q}(s)|}$ term prevents samples with many qualifying nodes from dominating training.

Final policy loss:

$L_{\text{policy}} \;=\; L_{\text{root}} + \lambda L_{\text{subtree}}$

---

# Integration requirements
1. Integrate subtree distillation **only during re-analyze**.
2. Add user-configurable:
   - $\lambda$: weight for subtree loss
   - `--min_visits`: qualifying-node threshold
3. Log subtree loss $L_{\text{subtree}}$.
4. Log mean visits of qualifying nodes: $\mathbb{E}_{h\in\mathcal{Q}}[N(h)]$.
5. Log mean depth of qualifying nodes: $\mathbb{E}_{h\in\mathcal{Q}}[\text{depth}(h)]$.
6. Log gradient norm attributable to $L_{\text{subtree}}$ (computed from that term alone).
7. Log entropies:
   - $H(\pi_\theta(\cdot\mid z_h))$
   - $H(\pi_{\text{target}}(\cdot\mid z_h))$

---

# Notation table

| Symbol | Meaning |
|---|---|
| $t$ | Time step index |
| $s_t$ | Environment state / observation at time $t$ (depending on your setup) |
| $a_t$ | Action selected at time $t$ |
| $r_{t+1}$ | Reward observed after taking $a_t$ and transitioning to $s_{t+1}$ |
| $z_t$ | Latent representation of $s_t$ (encoder output) |
| $\pi_\theta(a\mid z)$ | Network policy (prior) over actions given latent $z$ |
| $\pi_{\text{target}}(a\mid z)$ | MCTS-derived target policy (from normalized visit counts) |
| $\hat{V}^{\text{MCTS}}(s)$ | MCTS value estimate at root state $s$ |
| $\text{Tree}(s)$ | MCTS search tree built (during re-analyze) with root corresponding to $s$ |
| $h$ | A node in the MCTS tree (an internal search node) |
| $z_h$ | Latent state stored at node $h$ |
| $N(h)$ | Visit count of node $h$ in MCTS |
| $N(h,a)$ | Visit count of taking action $a$ from node $h$ in MCTS |
| $\mathcal{Q}(s)$ | Set of qualifying nodes in $\text{Tree}(s)$: $N(h)\ge\texttt{min\_visits}$ |
| $\mathcal{B}$ | Set of root states in a reanalyzed minibatch |
| $T$ | Unroll/chunk length sampled from replay |
| $\mathrm{CE}(p,q)$ | Cross-entropy: $-\sum_a p(a)\log q(a)$ |
| $L_{\text{root}}$ | Policy loss at roots (match $\pi_\theta$ to $\pi_{\text{target}}$ at root latents) |
| $L_{\text{subtree}}$ | Auxiliary subtree policy loss at qualifying internal nodes |
| $\lambda$ | Weight for subtree policy loss |
| $H(\pi)$ | Entropy of a policy distribution $\pi$ |
| $\text{depth}(h)$ | Depth of node $h$ in the MCTS tree (root depth $=0$) |
