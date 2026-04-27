"""
One-time script to normalise speaker names in data/sermons.db.
Run from the project root: python normalize_speakers.py
"""
import sqlite3

DB_PATH = "data/sermons.db"

SPEAKER_MAP = {
    # ── SP Daniel Foo ────────────────────────────────────────────────
    "Pastor Daniel Foo":            "SP Daniel Foo",
    "Ps Daniel Foo":                "SP Daniel Foo",
    "Senior Pastor Daniel Foo":     "SP Daniel Foo",
    "SP Dan Foo":                   "SP Daniel Foo",
    "SP Dani":                      "SP Daniel Foo",

    # ── SP Chua Seng Lee ─────────────────────────────────────────────
    "Senior Pastor Chua Seng Lee":  "SP Chua Seng Lee",
    "DSP Chua Seng Lee":            "SP Chua Seng Lee",
    "Elder DSP Chua Seng Lee":      "SP Chua Seng Lee",
    "SP Designate Chua Seng Lee":   "SP Chua Seng Lee",
    "Elder Chua Seng Lee":          "SP Chua Seng Lee",
    "SP Seng Lee":                  "SP Chua Seng Lee",
    "Ps Chua Seng Lee":             "SP Chua Seng Lee",

    # ── Ps Edric Sng ─────────────────────────────────────────────────
    "DSP Edric Sng":                "Ps Edric Sng",
    "Elder Edric Sng":              "Ps Edric Sng",
    "Elder DSP Edric Sng":          "Ps Edric Sng",
    "DSP Elder Edric Sng":          "Ps Edric Sng",
    "Pastor Edric Sng":             "Ps Edric Sng",
    "Edric":                        "Ps Edric Sng",

    # ── Ps Low Kok Guan ──────────────────────────────────────────────
    "DSP Low Kok Guan":             "Ps Low Kok Guan",
    "Pastor Lok Kok Guan":          "Ps Low Kok Guan",   # typo: Lok → Low
    "Elder Low Kok Guan":           "Ps Low Kok Guan",
    "Pastor Low Kok Guan":          "Ps Low Kok Guan",

    # ── Elder Lok Vi Ming ────────────────────────────────────────────
    "Elder Lok Vi Meng":            "Elder Lok Vi Ming",  # typo: Meng → Ming

    # ── Elder Leon Oei ───────────────────────────────────────────────
    "Leon Oei":                     "Elder Leon Oei",

    # ── Elder David Foo ──────────────────────────────────────────────
    "CS David Foo":                 "Elder David Foo",
    "David Foo":                    "Elder David Foo",
    "Cell Supervisor, David Foo":   "Elder David Foo",

    # ── Elder Mark Tan ───────────────────────────────────────────────
    "CS Mark Tan":                  "Elder Mark Tan",
    "Mark T":                       "Elder Mark Tan",

    # ── Ps Gary Koh ──────────────────────────────────────────────────
    "Pastor Gary Koh":              "Ps Gary Koh",
    "Gary Koh":                     "Ps Gary Koh",
    "Brother Gary Koh":             "Ps Gary Koh",
    "Mr Gary Koh":                  "Ps Gary Koh",
    "Gary & Joanna Koh":            "Ps Gary Koh",

    # ── Ps Jeffrey Aw ────────────────────────────────────────────────
    "Pastor Jeffrey Aw":            "Ps Jeffrey Aw",

    # ── Ps Andrew Tan ────────────────────────────────────────────────
    "Pastor Andrew Tan":            "Ps Andrew Tan",
    "Pastor Andrew Tan & Ps Sarah Khong": "Ps Andrew Tan",

    # ── Ps Darren Kuek ───────────────────────────────────────────────
    "Pastor Darren Kuek":           "Ps Darren Kuek",
    "Ps Darren":                    "Ps Darren Kuek",

    # ── Ps Don Wong ──────────────────────────────────────────────────
    "Pastor Don Wong":              "Ps Don Wong",

    # ── Ps Lawrence Chua ─────────────────────────────────────────────
    "Lawrence Chua":                "Ps Lawrence Chua",
    "Ps Lawrence":                  "Ps Lawrence Chua",
    "Senior Pastor Lawrence Chua":  "Ps Lawrence Chua",
    "SP Lawrence Chua":             "Ps Lawrence Chua",
    "SP LAWRENCE CHUA":             "Ps Lawrence Chua",

    # ── Ps Jason Teo ─────────────────────────────────────────────────
    "Pastor Jason Teo":             "Ps Jason Teo",
    "Jason Teo":                    "Ps Jason Teo",
    "Pastor Jaso":                  "Ps Jason Teo",

    # ── Ps Ng Hua Ken ────────────────────────────────────────────────
    "Pastor Ng Hua Ken":            "Ps Ng Hua Ken",
    "Hua Ken":                      "Ps Ng Hua Ken",
    "Elder Ps Ng Hua Ken":          "Ps Ng Hua Ken",

    # ── Ps Paul Jeyachandran ─────────────────────────────────────────
    "Rev. Paul Jeyachandran":       "Ps Paul Jeyachandran",

    # ── Ps Eugene Seow ───────────────────────────────────────────────
    "Pastor Eugene Seow":           "Ps Eugene Seow",
    "Eugene Seow":                  "Ps Eugene Seow",

    # ── Ps Nicky Raiborde ────────────────────────────────────────────
    "Nicky Raiborde":               "Ps Nicky Raiborde",
    "Ps Nicky S. Raiborde":         "Ps Nicky Raiborde",
    "Ps Nicky S Raiborde":          "Ps Nicky Raiborde",
    "Nicky S. Raiborde":            "Ps Nicky Raiborde",
    "nicky s raiborde":             "Ps Nicky Raiborde",
    "Nicky":                        "Ps Nicky Raiborde",

    # ── Dr John Andrews ──────────────────────────────────────────────
    "DR JOHN ANDREWS":              "Dr John Andrews",
    "D R J O H N A N D R E W S":   "Dr John Andrews",

    # ── Gurmit Singh ─────────────────────────────────────────────────
    "gurmit Singh":                 "Gurmit Singh",

    # ── Joseph Chean ─────────────────────────────────────────────────
    "Brother Joseph Chean":         "Joseph Chean",

    # ── Jeffrey Goh ──────────────────────────────────────────────────
    "Brother Jeffrey Goh":          "Jeffrey Goh",

    # ── Guest Speakers ───────────────────────────────────────────────
    # (real external people — identity confirmed or strongly inferred)
    "John":                         "Guest Speaker",
    "Tal":                          "Guest Speaker",
    "John Bunyan":                  "Guest Speaker",
    "MOSES GIDEON":                 "Guest Speaker",
    "Moses Gideon":                 "Guest Speaker",
    "Marie Tsuruda and Pierre Oosthuizen": "Guest Speaker",
    "Pastor Benny Ho":              "Guest Speaker",
    "Ps Benny Ho":                  "Guest Speaker",
    "Benny Ho":                     "Guest Speaker",
    "Ps Bill Wilson":               "Guest Speaker",
    "Pastor Craig Hill":            "Guest Speaker",
    "Dr Bill Bright":               "Guest Speaker",
    "Dr Victor Wong":               "Guest Speaker",
    "Dr Ng Liang Wei":              "Guest Speaker",
    "Ps William Wood":              "Guest Speaker",
    "Rev. Dr. Philip Huan":         "Guest Speaker",
    "REV. DR. PHILIP HUAN":         "Guest Speaker",
    "Ps Philip Huan":               "Guest Speaker",
    "Dr. Philip Huan":              "Guest Speaker",
    "Rev. Dr. Philip Huan, Rev. Jenni Ho-Huan": "Guest Speaker",
    "Rev David Ravenhill":          "Guest Speaker",
    "Ps David Ravenhill":           "Guest Speaker",
    "Leonard Ravenhill":            "Guest Speaker",
    "Ravenhill":                    "Guest Speaker",
    "Reverend Rick Seaward":        "Guest Speaker",
    "Rev Les Wheeldon":             "Guest Speaker",
    "Rev Daniel Wee":               "Guest Speaker",
    "Ps Daniel Wee":                "Guest Speaker",
    "Ps Daniel Koh":                "Guest Speaker",
    "Floyd McClung":                "Guest Speaker",
    "Dr Chester Kylstra":           "Guest Speaker",
    "Chester and Betsy Kylstra":    "Guest Speaker",
    "Josh McDowell":                "Guest Speaker",
    "Josh D. & Dottie McDowell":    "Guest Speaker",
    "Dr Dan Brewster":              "Guest Speaker",
    "Dr Cassie Carstens":           "Guest Speaker",
    "Dr Ian J":                     "Guest Speaker",
    "Dr Chris Cheech":              "Guest Speaker",
    "Ps Joey Bonifacio":            "Guest Speaker",
    "Ps Jerry Chia":                "Guest Speaker",
    "Ps Jeff Chong":                "Guest Speaker",
    "Ps Henson Lim":                "Guest Speaker",
    "Pastor Henson Lim":            "Guest Speaker",
    "Ps Hakan Gabrielsson":         "Guest Speaker",
    "Brother Hakan Gabrielsson":    "Guest Speaker",
    "Ps Ernest Chow":               "Guest Speaker",
    "Pastor Ernest Chow":           "Guest Speaker",
    "Ps Erne":                      "Guest Speaker",
    "Ps Watson":                    "Guest Speaker",
    "GEORGE BARNA":                 "Guest Speaker",
    "MICHAEL NOVAK":                "Guest Speaker",
    "MARY MA":                      "Guest Speaker",
    "Billy Graham":                 "Guest Speaker",
    "David Pawson":                 "Guest Speaker",
    "David Pawson, Billy Graham, Ravi, Apostle Paul": "Guest Speaker",
    "Martin Luther":                "Guest Speaker",
    "AW Tozer":                     "Guest Speaker",
    "A.W. Tozer":                   "Guest Speaker",
    "Mr. Vuong Dinh Hue":           "Guest Speaker",
    "Mr Lee Kuan Yew":              "Guest Speaker",
    "Prof Freddy Boey":             "Guest Speaker",
    "Pr Dr Chew Weng Chee":         "Guest Speaker",
    "Eugene Shi":                   "Guest Speaker",
    "Blessing Campa":               "Guest Speaker",
    "Dwight L Lord":                "Guest Speaker",

    # Multi-speaker panel rows (all BBTC staff, no single speaker)
    "Ps Low Kok Guan, Elder Lok Vi Ming, Ps Edric Sng, Elder Goh Hock Chye": "Unknown",
    "Ps Low Kok Guan, Elder Lok Vi Ming, DSP Edric Sng": "Unknown",

    # ── Unknown ──────────────────────────────────────────────────────
    # (not a real person name, placeholder, or hopelessly ambiguous)
    "BBTC":                         "Unknown",
    "null":                         "Unknown",
    "Pastor [Name] (assuming Pastor is the speaker, actual name not found)": "Unknown",
    "Pastor":                       "Unknown",
    "Rev":                          "Unknown",
    "Senior Pastor":                "Unknown",
    "Senior Minister":              "Unknown",
    "Pastoral Word":                "Unknown",
    "Pastor's name not mentioned":  "Unknown",
    "Pastor T3":                    "Unknown",
    "PASTOR FRIEND":                "Unknown",
    "Elder y":                      "Unknown",
    "Elder G":                      "Unknown",
    "Elder Lo":                     "Unknown",
    "Elde":                         "Unknown",
    "KK":                           "Unknown",
    "Q.":                           "Unknown",
    "me":                           "Unknown",
    "Lee":                          "Unknown",
    "Mr. Lee":                      "Unknown",
    "Jerry":                        "Unknown",
    "James":                        "Unknown",
    "Jacob":                        "Unknown",
    "Paul":                         "Unknown",
    "JOSHUA":                       "Unknown",
    "Thomas Jefferson":             "Unknown",
    "This is Jesus":                "Unknown",
    "T T Jo S s th m T cr d W th re s c T ti M tr w p W w s se": "Unknown",
    "Simon Peter":                  "Unknown",
    "Simeon Peter":                 "Unknown",
    "Page The":                     "Unknown",
    "Pastor Abraham Jacob Joseph Joshua Moses": "Unknown",
    "Jephthah Fugitive | Fighter | Father": "Unknown",
    "KING ASA":                     "Unknown",
    "Commission F":                 "Unknown",
    "F_______-L________":           "Unknown",
    "Apostle Paul":                 "Unknown",
    "DR LUKE":                      "Unknown",
    "Abraham A Life and Legacy":    "Unknown",
    "Christian":                    "Unknown",
    "Video Wilson Foo":             "Unknown",
    "Rev Wil":                      "Unknown",
    "Mr Jeffr":                     "Unknown",
    "Hezekiah":                     "Unknown",
    "Ehud":                         "Unknown",
    "SP Dani":                      "SP Daniel Foo",   # handled above; listed again for safety
}


def main():
    total_changed = 0

    with sqlite3.connect(DB_PATH) as conn:
        for raw, canonical in SPEAKER_MAP.items():
            cursor = conn.execute(
                "UPDATE sermons SET speaker = ? WHERE speaker = ?",
                (canonical, raw),
            )
            if cursor.rowcount:
                print(f"  {cursor.rowcount:>3}  {raw!r}  →  {canonical!r}")
                total_changed += cursor.rowcount

        # SQL NULLs → "Unknown"
        cursor = conn.execute(
            "UPDATE sermons SET speaker = 'Unknown' WHERE speaker IS NULL"
        )
        if cursor.rowcount:
            print(f"  {cursor.rowcount:>3}  NULL  →  'Unknown'")
            total_changed += cursor.rowcount

        conn.commit()

    print(f"\nDone. {total_changed} row(s) updated.")

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT speaker, COUNT(*) as n FROM sermons "
            "GROUP BY speaker ORDER BY n DESC"
        ).fetchall()
    print(f"\nFinal speaker roster ({len(rows)} distinct):")
    for speaker, count in rows:
        print(f"  {count:>4}  {speaker}")


if __name__ == "__main__":
    main()
