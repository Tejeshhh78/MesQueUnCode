# Rules

## What you may use

**Any method at all**, subject to two conditions:

1. **The organisers can access and inspect it.** Anything in your pipeline has to be something we
   can look at and run — your repository, a public model, a commercial API, an open dataset.
2. **Your technical document explains it.** Not a full paper; enough that a reader understands what
   you built and why.

Within that, nothing is off the table. Any language, any framework. Pre-trained models, commercial
or open. LLMs, hosted or local. External data about other games. Your own detection, tracking or
event-recognition models. Classical methods with no learning at all.

Both conditions exist for the same reason: the score decides the ranking, and the repository and
document are what make the score mean something. A method nobody can access and nobody can
understand cannot be verified, and an unverifiable result is not a result.

**A private in-house model or an unavailable internal dataset is the one thing this rules out.** If
your approach depends on something you cannot share, ask before you build on it.

### The provided data is a starting point, not a constraint

You are given automatic tracking, tactical tagging and homography so that no team has to rebuild
detection, re-identification and camera calibration before starting on the actual question.

**Improving that data, or replacing it with your own, is entirely legitimate** and needs no
justification. Run your own detector, improve the tracking with your own algorithms, build your own
event recognition — if it produces a better answer, it is a better answer. We do not care where the
intermediate representations come from, as long as they are produced by an algorithm.

Corrections must come out of code. Manually correcting the tracking of the provided game — fixing
ball positions, identity switches or anything else by hand — is hand annotation, and rule 1
applies.

---

## What you may not do

### 1. Annotate the data by hand

**The labels must be produced by an algorithm.** A human watching the video and writing down who
has the ball is not a submission, no matter how accurate it is.

This applies whether or not any answer key was involved. Hand-labelling by eye, hand-correcting an
algorithm's output — frame by frame, segment by segment or at transitions only — and hand-labelling
against a leaked ground truth are all the same thing as far as this rule is concerned: the file did
not come from a system, so there is no system to evaluate and nothing that would work on the next
match.

#### Training and validation data

The line is drawn by **which game** the labels are on, not by how many there are.

**No hand labels on the provided game, in any amount and for any purpose** — not for the
submission, not to train a model and not to validate one. The provided game is the test set: a
model trained on human labels from the game it is then asked to label is memorising, not
generalising, and no threshold on "how much" would make that a fair comparison between teams.

**Hand-labelled data from other games is fine, in any amount.** Label as much as you like. Like
everything else in your pipeline, the organisers must be able to access it and your technical
document must explain it: what was labelled, how, by whom and how much.

What remains fine on the provided game:

- **Learning without human labels** — unsupervised or self-supervised methods, or pseudo-labels
  produced by an algorithm, provided no person decided any of them.
- **Watching the video and inspecting the data** to understand them, debug your pipeline and
  design features. Watching is not the problem; turning what you saw into labels, corrections, or
  rules and constants tied to specific frames of the game is.

### 2. Use the private ground truth, or any human-produced data about the game

You may not use the full-match ground truth, or anything derived from it.

The 5 minutes published in this repository are for **checking that your pipeline is correct** —
frames aligned, file well-formed, labels meaning what you think. They are not training data and not
a tuning set. See [EVALUATION.md](EVALUATION.md) for why tuning on them picks the wrong answer even
without trying.

The same applies to **any human-produced data about the provided game, from any source**: event
data (Opta, StatsBomb, Wyscout, LaLiga or anyone else), corrected tracking, possession data or
annotations of any kind. Data of this kind about other games is fine.

### 3. Submit a file your repository cannot regenerate

Running your repository against the provided data must reproduce your submission. Not
approximately — the same file.

---

## Eligibility and teams

Each team must be composed of 3 to 5 members 
(Teams with less than 3 members will not be eligible for the Innovation Challenge and the Final Pitch.)

---

## The data

The match assets are licensed for this competition only, and the terms you accept at enrolment are
the authority on what that permits. They are not restated here.

Two things follow from them that are easy to get wrong:

- **Do not redistribute the data.** That includes committing it to a public repository. Your
  submission repository has to be public, so make sure the data is not in it — this repository's
  `.gitignore` already excludes `data/`, and yours should too.
- **Do not publish derived data that would reconstruct it.** Frame-level tracking exports, clipped
  video, and similar.

The code and documentation in this repository are separately licensed — see [LICENSE](LICENSE).
That licence covers this repository only and says nothing about the match data.

---

## Questions

If a rule is unclear, or your approach sits near a line, **ask before you build on it**. Open an
issue. An answer before the deadline is worth more to everyone than a disqualification after it,
and a question that needed asking usually means the rule needed clarifying.
