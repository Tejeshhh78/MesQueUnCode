# Was wir von den öffentlichen Challenge-Forks übernehmen können

Stand 2. November 2026, 09:17 Uhr MEZ. Der Crawler fand 189 direkte Forks und zwei Unterforks. Er prüfte 330 öffentliche Branches in 191 Repos, einschließlich MesQueUnCode. In 57 Repos zeigte mindestens ein Branch auf einen Commit außerhalb der Starter-Historie. 49 davon enthielten zusätzliche oder geänderte Quelldateien in den untersuchten Formaten. Zwei Quelldatei-Einträge über 1 MiB wurden ausgelassen. Der Abruf meldete keine Fehler.

## Drei Techniken mit nachvollziehbarem Nutzen

| Team | Was im Code steht | Bedeutung für MesQueUnCode |
|---|---|---|
| GTT-TalentArena | Das Team schätzt Ballbesitz bei verdecktem Ball aus Formation und Bewegung. Klare Ballkontakte liefern automatische Beispiele. Die andere Halbzeit dient zur Überprüfung. [Implementierung](https://github.com/GTT-TalentArena/more-than-a-hack-2027/blob/05b1c34e85a267dfd8e090ac8f7480a40e199f65/solution/a_opus/ballfree.py) | Unser Decoder hat in Balllücken oft gleich starke Indizien für beide Teams. Relative Mannschaftsbreite lässt sich mit dem vorhandenen taktischen Tracking prüfen. Diesen Teil haben wir eigenständig implementiert und gemessen. |
| TrymBardsen | Das Team gleicht die Broadcast-Aufnahme abschnittsweise über Schwerpunkt und Streuung der Spieler mit der taktischen Aufnahme ab. Es verwirft mehrdeutige Zuordnungen und prüft die Ballposition, wo beide Kameras sie sehen. [Implementierung](https://github.com/TrymBardsen/more-than-a-hack-2027/blob/2a7432e5b92bb6f6b00632fbafb80d210724d35a/solution/broadcast_sync.py) | Eine zweite Kamerasicht könnte weitere Lücken schließen. Dafür müssten wir mit beiden lizenzierten Aufnahmen messen, wie viele fehlende Frames zuverlässig zugeordnet werden und wie oft die Positionen widersprechen. |
| 7dracoder | Der Fork liest Ballhöhe aus externen Trackingdaten, prüft die Übereinstimmung der Frame-IDs und nutzt Höhe als zusätzliches Trainingsziel. [Implementierung](https://github.com/7dracoder/more-than-a-hack-2027/blob/d4fc446c8c6de488ff5c9170c39371a114d1c2cf/possession/height.py) | Luftbälle sind ein konkreter Fall, in dem der nächste Spieler möglicherweise keinen Ballbesitz hat. Der Ansatz setzt ein trainierbares Modell und fremde Spiele mit Höhenmessungen voraus. |

Die drei Beispiele beruhen auf gelesenen Implementierungen. Ein Vorkommen im Fork sagt nichts über den privaten Challenge-Score aus. Bei MesQueUnCode enthält `main` den Starter. Dieser Versuch setzt auf dem Integrationsbranch `testing` auf. Der gesonderte Entwicklungsbranch `dev_VG` hat bereits eigene Experimente zur zeitlichen Ballkontrolle. [Vergleichsbasis](https://github.com/Tejeshhh78/MesQueUnCode/tree/ceffe5b4c73f38f9f89563b66dde19da3e8e712d) · [Bestehendes Kontrollexperiment](https://github.com/Tejeshhh78/MesQueUnCode/blob/6f053cbf95a4ad19192faa52e7769d7220ee013f/solution/experiments/temporal-control-003.json)

## Vergleich von v2 und v2-shape

Das neue Profil ergänzt `v2` um ein schwaches Signal für H oder A. Bei mindestens sieben sichtbaren Spielern pro Team misst es die Differenz der seitlichen Streuung. In der anderen Halbzeit liefern klare Ballkontakte automatische Beispiele pro Team. Fehlen ausreichend Beispiele oder zeigt sich dort kein positiver Zusammenhang, bleibt das Zusatzsignal null. Das Profil verwendet es ausschließlich auf Frames mit fehlender Ballposition. Der bestehende Decoder entscheidet weiterhin über die vollständige Zustandsfolge.

Wir haben beide Profile mit dem offiziellen Scorer auf denselben degradierten Trackingdaten der öffentlichen Metrica-Beispielspiele 1, 2 und 3 verglichen. Jedes Spiel lief mit zwei Seeds, einmal ohne Tags und einmal mit simulierten fehlerhaften Tags. Die Referenzlabels stammen aus den Ereignissen dieser anderen Spiele. Diese Spiele wurden im Repo bereits für die Entwicklung von v2 verwendet. [Datenquelle](https://github.com/metrica-sports/sample-data) · [Challenge-Regeln](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/RULES.md)

| Bedingung | Paare | spellF1 v2 | spellF1 v2-shape | Differenz | macroF1 v2 | macroF1 v2-shape |
|---|---:|---:|---:|---:|---:|---:|
| Ohne Tags | 6 | 0,5092 | 0,5198 | +0,0106 | 0,8447 | 0,8524 |
| Fehlerhafte Tags | 6 | 0,5126 | 0,5381 | +0,0255 | 0,8506 | 0,8653 |
| Alle | 12 | 0,5109 | 0,5289 | +0,0180 | 0,8477 | 0,8589 |

Elf von zwölf Paaren verbessern sich bei spellF1. Der Rückgang war Spiel 1, Seed 1, ohne Tags (−0,0052). Mehrere Seeds desselben Spiels sind keine unabhängigen Spiele. Dies ist ein Versuch auf drei externen Spielen mit simuliertem Trackingrauschen. Die Leistung auf dem privaten Challenge-Match ist offen. Das neue Profil bleibt für weitere Vergleiche zuschaltbar.

## Ausführen

Im Root des Team-Repos, mit Git und Python 3.8+ für den Crawler bzw. der dokumentierten Team-Python-Umgebung für das Modell:

```bash
python tools/fork_scout.py
python -m unittest discover -s tests
python -m validation --download --games 1 2 3 --profiles v2 v2-shape --seeds 0 1 --regimes default --tags none noisy
python -m possession --profile v2-shape --output-dir artifacts/shape
```

Der Crawler erzeugt `artifacts/fork-scout.json` mit Repos, Branches, Commit-IDs, Pfaden und Fundstellen sowie `artifacts/fork-scout.md` mit Themenhäufigkeiten. Für das API-Limit ist `GITHUB_TOKEN` optional. Ein Abruffehler führt zu Exitcode 1 und steht im Bericht. Die Methode analysiert die öffentlich sichtbaren Branch-Spitzen, überspringt Datenverzeichnisse und führt keinen Fremdcode aus. Thematische Treffer sind Anhaltspunkte, auch Kommentare und Tests können sie auslösen. Das Modell benötigt für den letzten Befehl das lokale Challenge-Datenpaket.
