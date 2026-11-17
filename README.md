# DAT305 Prosjekt 6: What does a pretrained ECG model actually see?

Gruppe 27: Anders, Øystein og Håvard.

Vi kjører 1292 EKG fra PTB-XL gjennom den ferdigtrente EKG-modellen ECGFounder og får
1024 tall (en embedding) per EKG. Så undersøker vi hvilke diagnoser modellen blander sammen,
med PCA, UMAP, k-NN-enighet, silhouette og en matrise over hvilke diagnoser som er naboer.

## Kjør i denne rekkefølgen

| Steg | Fil | Hva den gjør | Resultat |
|---|---|---|---|
| 1 | `1_utvalg.ipynb` | Velger EKG med én diagnose, minst 50 % sikkerhet, ett per pasient, ca. 100 per klasse | `utvalg.csv` (1292 EKG, 13 klasser) |
| 2 | `2_embeddings.py` | Henter EKG-ene fra PhysioNet og kjører dem gjennom ECGFounder | `data/embeddings.npy` (1292 × 1024) |
| 3 | `3_analyse.ipynb` | PCA, UMAP, k-NN, silhouette, naboblanding og sammenligning av metodene | Plott og tall til rapporten |

`utvalg.csv` ligger i repoet, så du kan starte på steg 2.

## Oppsett

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Last ned vektene til modellen (370 MB) fra
https://huggingface.co/PKUDigitalHealth/ECGFounder og legg filen her:
`checkpoint/12_lead_ECGFounder.pth`.

Kjør steg 2 fra rotmappen: `python 2_embeddings.py`. Første gang tar det omtrent en halvtime,
fordi hvert EKG hentes fra PhysioNet. Kopiene lagres i `data/signaler/`, så neste gang går det raskt.

## Mapper

| Mappe | Innhold | I git? |
|---|---|---|
| `ecgfounder/` | Modellkoden fra ECGFounder (`net1d.py`, `tasks.txt`, MIT-lisens) | Ja |
| `utforsking/` | Tidlig utforskning: første embedding, Øysteins første kjøring, kolonnene i PTB-XL | Ja |
| `data/` | EKG-kopier og embeddings | Nei |
| `checkpoint/` | Modellvektene | Nei |

Filene i `utforsking/` viser hvordan vi kom frem til oppsettet. De er ikke en del av
løsningen, og stiene i dem kan være utdaterte.

## Kilder

- Li m.fl. (2025). An electrocardiogram foundation model built on over 10 million recordings. *NEJM AI*. Kode: https://github.com/PKUDigitalHealth/ECGFounder
- Wagner m.fl. (2022). PTB-XL, a large publicly available electrocardiography dataset (versjon 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Wagner m.fl. (2020). PTB-XL, a large publicly available ECG dataset. *Scientific Data*. https://doi.org/10.1038/s41597-020-0495-6
