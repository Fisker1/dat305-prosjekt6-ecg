"""
Steg 2: Gjør hvert EKG i utvalg.csv om til 1024 tall (en embedding) med ECGFounder.

Kjør fra prosjektmappen:   python 2_embeddings.py

Første gang hentes hvert EKG fra PhysioNet (ca. 1,5 sek per EKG, rundt en halvtime).
En kopi lagres i data/signaler/, så neste gang går det på få minutter.
Stopper det underveis, er det bare å kjøre på nytt: kopiene som finnes, brukes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from ecgfounder.net1d import Net1D

# ---------------------------------------------------------------- valgene våre
VEKTER = "checkpoint/12_lead_ECGFounder.pth"   # modellen, lastet ned fra Hugging Face
PHYSIONET = "ptb-xl/1.0.3/"                   # versjonen av PTB-XL vi henter fra
SIGNALMAPPE = Path("data/signaler")            # lokale kopier av EKG-ene (ikke i git)
SEED = 305                                     # modellen er frossen, men vi låser tilfeldigheten likevel

torch.manual_seed(SEED)
SIGNALMAPPE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 1. hent ett EKG
def hent_ekg(ecg_id, filnavn):
    """Gir ett EKG som en tabell på 5000 × 12: 10 sekunder à 500 målinger, 12 avledninger."""
    kopi = SIGNALMAPPE / f"{ecg_id}.npy"
    if kopi.exists():
        return np.load(kopi)

    # filnavn ser slik ut: records500/00000/00016_hr  ->  mappe og opptak
    mappe, opptak = filnavn.rsplit("/", 1)
    signal, info = wfdb.rdsamp(opptak, pn_dir=PHYSIONET + mappe)
    np.save(kopi, signal)
    return signal


# ---------------------------------------------------------------- 2. gjør det klart for modellen
def forbered(signal):
    """Samme forbehandling som ECGFounder ble trent med."""
    signal = np.nan_to_num(signal)                              # manglende målinger blir 0
    signal = signal.T                                           # snu: 12 avledninger × 5000 målinger
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)   # z-normaliser hele opptaket
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)   # (1, 12, 5000): én om gangen


# ---------------------------------------------------------------- 3. last inn modellen
def last_modell():
    """Bygger ECGFounder med forskernes innstillinger og laster de ferdigtrente vektene."""
    modell = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        use_bn=False, use_do=False, verbose=False,
        n_classes=150,
        return_features=True,        # hint 2: gi oss de 1024 tallene, ikke bare diagnosesvaret
    )
    sjekkpunkt = torch.load(VEKTER, map_location="cpu", weights_only=False)
    modell.load_state_dict(sjekkpunkt["state_dict"])
    modell.eval()                    # bruksmodus: vi trener ingenting
    return modell


# ---------------------------------------------------------------- 4. kjør alle EKG gjennom modellen
utvalg = pd.read_csv("utvalg.csv")
enhet = "cuda" if torch.cuda.is_available() else "cpu"
modell = last_modell().to(enhet)
print(f"{len(utvalg)} EKG, kjører på {enhet}")

embeddings = []
diagnosesvar = []

with torch.no_grad():                # ingen trening: ikke regn ut gradienter
    for nummer, rad in enumerate(utvalg.itertuples(), start=1):
        signal = hent_ekg(rad.ecg_id, rad.filename_hr)
        x = forbered(signal).to(enhet)
        svar, embedding = modell(x)  # svar: 150 diagnoser, embedding: 1024 tall
        embeddings.append(embedding[0].cpu().numpy())
        diagnosesvar.append(svar[0].cpu().numpy())
        if nummer % 100 == 0:
            print(f"  {nummer} av {len(utvalg)}")

embeddings = np.array(embeddings)
diagnosesvar = np.array(diagnosesvar)
print("Ferdig:", embeddings.shape)


# ---------------------------------------------------------------- 5. lagre, i samme rekkefølge som utvalg.csv
np.save("data/embeddings.npy", embeddings)
np.save("data/logits.npy", diagnosesvar)
utvalg[["ecg_id", "klasse"]].to_csv("data/embeddings_rekkefolge.csv", index=False)

# Samme tall som én CSV-fil (ecg_id + 1024 kolonner), lett å dele med gruppen
tabell = pd.DataFrame(embeddings)
tabell.insert(0, "ecg_id", utvalg["ecg_id"])
tabell.to_csv("data/embeddings.csv", index=False)

# Kontroll: hva tror modellen om det første EKG-et?
diagnosenavn = open("ecgfounder/tasks.txt").read().splitlines()
beste = diagnosenavn[diagnosesvar[0].argmax()]
print(f"Kontroll: EKG {utvalg.ecg_id[0]} er {utvalg.klasse[0]} i fasiten, modellen tror mest på {beste}")
