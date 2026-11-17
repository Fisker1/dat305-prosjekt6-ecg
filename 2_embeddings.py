"""
Lager embeddings for de 1292 EKG-ene i utvalg.csv med ECGFounder.

Kjør fra rotmappen i prosjektet, med det virtuelle miljøet aktivt:

    python 2_embeddings.py

Hva skriptet gjør, i fire steg:
  1. Leser utvalg.csv (hvilke EKG vi skal ha, og diagnosen til hvert).
  2. Henter hvert EKG fra PhysioNet med wfdb (hint 3), og lagrer en kopi i
     data/signaler/ så vi slipper å hente på nytt neste gang.
  3. Kjører hvert EKG gjennom den frosne ECGFounder-modellen og tar vare på
     embeddingen (1024 tall) og diagnosesvaret (150 tall).
  4. Lagrer resultatet i data/, i samme rekkefølge som utvalg.csv.

Henting tar omtrent 1,5 sekunder per EKG første gang (rundt en halvtime).
Blir det avbrutt, kjør skriptet igjen: det fortsetter der det slapp.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

# ---------------------------------------------------------------------------
# Oppsett: mapper og fast seed
# ---------------------------------------------------------------------------
ROT = Path(__file__).resolve().parent              # prosjektmappen DAT305
SIGNALMAPPE = ROT / "data" / "signaler"             # kopier av EKG (ikke i git)
UTMAPPE = ROT / "data"                              # resultatene (ikke i git)
VEKTER = ROT / "checkpoint" / "12_lead_ECGFounder.pth"
ECGFOUNDER = ROT / "ecgfounder"                     # forskernes kode (net1d.py, tasks.txt)
PHYSIONET = "ptb-xl/1.0.3/"
BATCH = 32

torch.manual_seed(305)                              # samme resultat hver gang
np.random.seed(305)
sys.path.append(str(ECGFOUNDER))
from net1d import Net1D                             # noqa: E402

SIGNALMAPPE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Steg 1: les utvalget
# ---------------------------------------------------------------------------
utvalg = pd.read_csv(ROT / "utvalg.csv")
print(f"Steg 1: {len(utvalg)} EKG i utvalg.csv, {utvalg['klasse'].nunique()} klasser")


# ---------------------------------------------------------------------------
# Steg 2: hent EKG fra PhysioNet (eller fra kopien i data/signaler/)
# ---------------------------------------------------------------------------
def hent_ekg(ecg_id, filnavn):
    """Gir ett EKG som en tabell med formen (5000, 12): tid x avledning."""
    kopi = SIGNALMAPPE / f"{ecg_id}.npy"
    if kopi.exists():
        return np.load(kopi)

    # filnavn ser slik ut: records500/00000/00016_hr
    mappe, opptak = filnavn.rsplit("/", 1)
    for forsok in range(3):
        try:
            signal, meta = wfdb.rdsamp(opptak, pn_dir=PHYSIONET + mappe)
            break
        except Exception as feil:                   # nettverket kan svikte
            print(f"  feil ved {opptak} (forsøk {forsok + 1}): {feil}")
            time.sleep(3)
    else:
        raise RuntimeError(f"Fikk ikke hentet {opptak}")

    assert meta["fs"] == 500 and signal.shape == (5000, 12), (opptak, signal.shape)
    signal = signal.astype(np.float32)
    np.save(kopi, signal)
    return signal


print("Steg 2: henter EKG (de som allerede er hentet, hoppes over)")
start = time.time()
for i, rad in enumerate(utvalg.itertuples(), start=1):
    hent_ekg(rad.ecg_id, rad.filename_hr)
    if i % 50 == 0 or i == len(utvalg):
        print(f"  {i}/{len(utvalg)} ferdig ({time.time() - start:.0f} s)")


# ---------------------------------------------------------------------------
# Steg 3: bygg modellen og kjør alle EKG gjennom den
# ---------------------------------------------------------------------------
enhet = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Steg 3: kjører ECGFounder på {enhet}")

# Samme oppsett som i forskernes ptbxl_eval.py, pluss return_features=True (hint 2)
modell = Net1D(
    in_channels=12, base_filters=64, ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16, stride=2, groups_width=16, verbose=False,
    use_bn=False, use_do=False, n_classes=150, return_features=True,
)
sjekkpunkt = torch.load(VEKTER, map_location="cpu", weights_only=False)
logg = modell.load_state_dict(sjekkpunkt["state_dict"], strict=False)
assert not logg.missing_keys and not logg.unexpected_keys, logg   # alle vekter passer
modell = modell.to(enhet).eval()                     # bruksmodus, frossen


def forbered(signal):
    """Som i ECGFounder: NaN blir 0, snu til (12, 5000), z-normaliser hele opptaket."""
    x = np.nan_to_num(signal).T
    return (x - x.mean()) / (x.std() + 1e-8)


embeddings, logits = [], []
with torch.no_grad():                                # ingen trening, bare spør
    for fra in range(0, len(utvalg), BATCH):
        del_av_utvalg = utvalg.iloc[fra:fra + BATCH]
        x = np.stack([forbered(hent_ekg(r.ecg_id, r.filename_hr)) for r in del_av_utvalg.itertuples()])
        x = torch.tensor(x, dtype=torch.float32, device=enhet)   # (batch, 12, 5000)
        ut, emb = modell(x)                                       # (batch, 150), (batch, 1024)
        logits.append(ut.cpu().numpy())
        embeddings.append(emb.cpu().numpy())
        print(f"  {min(fra + BATCH, len(utvalg))}/{len(utvalg)}")

embeddings = np.concatenate(embeddings)
logits = np.concatenate(logits)


# ---------------------------------------------------------------------------
# Steg 4: lagre, i samme rekkefølge som utvalg.csv
# ---------------------------------------------------------------------------
assert embeddings.shape == (len(utvalg), 1024) and not np.isnan(embeddings).any()
np.save(UTMAPPE / "embeddings.npy", embeddings)
np.save(UTMAPPE / "logits.npy", logits)
utvalg[["ecg_id", "klasse"]].to_csv(UTMAPPE / "embeddings_rekkefolge.csv", index=False)

# Samme innhold som én CSV (ecg_id + 1024 kolonner), lett å dele med gruppen
tabell = pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(1024)])
tabell.insert(0, "ecg_id", utvalg["ecg_id"].values)
tabell.to_csv(UTMAPPE / "embeddings.csv", index=False)

# Kontroll: modellens beste gjetning for det første EKG-et mot fasiten
diagnoser = [l.strip() for l in open(ECGFOUNDER / "tasks.txt")]
beste = diagnoser[int(np.argmax(logits[0]))]
print(f"Steg 4: lagret embeddings {embeddings.shape} i data/")
print(f"  Kontroll, EKG {utvalg.ecg_id[0]}: fasit {utvalg.klasse[0]}, modellen tror mest på «{beste}»")
