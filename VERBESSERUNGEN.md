# Baseline-Analyse & Ideen für Verbesserungen

Ich habe mir die bestehenden Baselines (`starter/baseline.py`) und den Scorer (`evaluation/run_evaluation.py`) in einer lokalen `.venv` genauer angeschaut und gegen die 5-Minuten-Ground-Truth (`11185..18684`, 17 Ballbesitzphasen, Ø 14,2 s) getestet. Hier sind die wichtigsten Baustellen, die mir aufgefallen sind, und wie wir den Score deutlich nach oben bekommen sollten.

---

## 1. Aktuelle Baseline-Werte (Public 5 Min vs. Full Match)

| Variante | Public `spellF1` | Full-Match `spellF1` | Public `macroF1` | `f1_D` (Dead Ball) | Possessions (GT = 17) | Ø Dauer (GT = 14,2 s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_raw` | 0.0378 | 0.0486 | 0.5972 | 0.4197 | 131 | 2,0 s |
| `baseline_smooth25` | 0.2868 | 0.1996 | 0.6200 | 0.3857 | 36 | 7,6 s |
| `baseline_smooth50` | 0.3256 | **0.2433** | **0.6207** | 0.3225 | 26 | 10,8 s |
| `baseline_smooth125` | **0.4242** | 0.2242 | 0.5812 | 0.1464 | 16 | 18,5 s |

---

## 2. Was in der aktuellen Baseline schiefläuft

### 1. Dead-Ball-Logik in `smooth()` macht `D`-Phasen kaputt
In `starter/baseline.py` wird beim Glätten einfach geprüft:
```python
seen = (home + away) > 0
```
Das heißt: Selbst bei einem großen Fenster (`±50` oder `±125` Frames, also bis zu 10 Sekunden) reicht ein einziger Frame mit erkanntem Ball aus, damit das gesamte Fenster als `H` oder `A` gelabelt wird. Deshalb bricht der `f1_D`-Wert bei `smooth125` auf `0.1464` ein.
**Fix:** Statt `(home + away) > 0` eine Mindestquote verlangen (z. B. dass der Ball in mindestens 30–35 % der Frames im Fenster sichtbar sein muss), sonst `D` bzw. `X`.

### 2. Ruhender Ball bei Unterbrechungen (Freistoß / Einwurf)
In den ersten 5 Minuten gibt es vier echte `D`-Phasen:
- `12851..12989` (5,5 s `D`) – Ball kaum sichtbar (7,2 %).
- `15889..16400` (20,4 s `D`) – Ball ist zu **100 % sichtbar**, liegt aber wegen einer Spielunterbrechung ruhig bzw. wird vor der Ausführung gehalten.
- `17380..17838` (18,3 s `D`) – Ball zu 20,1 % sichtbar.
- `18027..18389` (14,5 s `D`) – Ball wieder zu **100 % sichtbar**, direkt danach (`18389..18685`) führt `H` aus.

Weil `baseline.py` nur schaut, ob der Ball getrackt wird, vergibt es bei `15889..16400` und `18027..18389` durchgehend `H`/`A`. Bei `18027..18685` verschmilzt dadurch die komplette Unterbrechung mit dem nächsten Ballbesitz zu einem langen `H`-Block, was den IoU beim `spellF1` kaputtmacht.
**Fix:** Ballgeschwindigkeit (3. Wert im ATD-Ball-String, wird in `baseline.py` aktuell ignoriert) und Durchschnittsgeschwindigkeit der Spieler mit einbeziehen. Wenn fast alle Spieler stehen/gehen und der Ball sich kaum bewegt oder außerhalb des Feldes (`x, y` nahe/außerhalb `0..1`) ist -> `D`.

### 3. Zweite Halbzeit vor dem Anstoß (`110808..113741`)
Laut Doku (`DATA.md` / `FAQ.md`) wird die 2. Halbzeit schon ab Frame `110808` gewertet, der eigentliche Anstoß ist aber erst bei Frame **`113742`**.
- Die `2.934` Frames dazwischen (knapp 2 Minuten) sind komplett `D`.
- `baseline_smooth50` setzt dort aktuell aber 1.957 Frames auf `H` und 637 Frames auf `A` (nur 11,6 % `D`).
**Fix:** `110808..113741` fest auf `D` setzen und alle unbewerteten Bereiche (`0..11184`, Halbzeit `90652..110807`, nach Abpfiff `190198..199318`) sauber auf `X` setzen.

### 4. Kurze Tracking-Aussetzer im laufenden Spiel (`X` statt `D` nutzen)
Im Segment `16400..17293` (35,7 s Ballbesitz `A`) fehlt der Ball in rund 41 % der Frames (Flugball / Verdeckung). Wenn die Lücke größer als das Smoothing-Fenster wird, fällt die Baseline auf `D` zurück und zerschneidet einen durchgehenden Ballbesitz in mehrere Teile.
- Da der Scorer bei `spellF1` ein `X` innerhalb einer Halbzeit automatisch mit dem letzten Label auffüllt (`fill_abstentions`), sollten wir bei unklaren kurzen Lücken im Spielverlauf `X` statt `D` ausgeben bzw. sehr kurze Gegen-Segmente (`< 1,5–2 s`) herausfiltern.

### 5. Distanz zum Ball & Smart Tagging nutzen
- **Maximaldistanz für Ballkontrolle:** Aktuell bekommt immer der nächstgelegene Spieler den Ballbesitz, selbst wenn der Ball 20 Meter weit weg in der Luft ist. Sinnvoller ist ein echter Kontrollradius (z. B. `< 2,0–2,5 m`). Solange kein Spieler nah genug am Ball ist (Pass/Flugphase), bleibt der Ballbesitz beim letzten Team.
- **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):**
  - Die `SET PIECES`-Einträge markieren ziemlich genau die Fenster, in denen nach einem `D`-Abschnitt wieder angepfiffen wird.
  - Die Phasen (`BUILD UP`, `PROGRESSION`, `DEFENDING IN ...`) liefern über das `Team`-Tag einen guten Prior, welches Team in dem Abschnitt gerade angreift bzw. verteidigt.

---

## 3. Erster Testlauf mit angepasstem Smoothing

Ich habe lokal in der `.venv` kurz getestet, was allein die drei schnellen Anpassungen auf den bestehenden Roh-Codes bringen:
1. Ungraded-Bereiche auf `X` und `110808..113741` auf `D`.
2. Sichtbarkeits-Schwelle `min_vis_ratio = 0.35` im Fenster.
3. Kurze `H`/`A`-Flicker unter 45 Frames (1,8 s) durch `X` überbrücken.

Ergebnis auf den Public 5 Minuten:
- **`w = 50`:** `spellF1` steigt von **0.3256 auf 0.4000**, `macroF1` von **0.6207 auf 0.6487**, `f1_D` von 0.3225 auf 0.4032 (Possessions: 21 statt 26).
- **`w = 75`:** `spellF1` **0.4235**, `macroF1` **0.6544**, exakt **17 Possessions** (wie im GT).
- **`w = 125`:** `spellF1` steigt von **0.4242 auf 0.4966**, `macroF1` von **0.5812 auf 0.6419**.

---

## 4. Nächste Schritte

1. [ ] Eigenes Pipeline-Skript aufbauen, das die schnellen Fixes (Halbzeit-Grenzen, Pre-Kickoff `D`, Sichtbarkeits-Schwelle, Mindestdauer pro Possession) sauber umsetzt.
2. [ ] Im ATD-Parser (`nearest_player_labels`) Ballgeschwindigkeit und minimalen Spielerabstand pro Team auslesen, um Pässe und echte Unterbrechungen zu trennen.
3. [ ] Smart-Tagging-Events (`SET PIECES` und Spielphasen) als zusätzliche Bedingung für Dead-Ball-Ende und Team-Zuordnung einbauen.
4. [ ] Optional: Statt einfachem Sliding-Window einen kleinen Viterbi-/HMM-Decoder über die Zustände `H`, `A`, `D` legen, damit die Übergänge schärfer werden.
