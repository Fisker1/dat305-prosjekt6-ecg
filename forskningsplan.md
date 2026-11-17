# Forskningsplan · Prosjekt 6: What does a pretrained ECG model actually see?

DAT305, gruppe #27. Utkast 3.11.2026 til gruppemøtet 4.–6.11. Alt i hakeparenteser må avklares.

## 1 Forskningsspørsmål

**Hovedspørsmål.** Hvilke diagnoseklasser i PTB-XL flyter sammen i embeddingrommet til den frosne
ECGFounder-encoderen, og følger sammenblandingen klinisk likhet?

Delspørsmål:

- **F1.** Hvor godt skiller embeddingene diagnoseklassene, målt med k-NN-etikettenighet og silhouette score?
- **F2.** Hvilke klassepar blandes mest, og får vi de samme parene i det fulle rommet, med PCA, med t-SNE og
  med UMAP?
- **F3.** Er parene som blandes klinisk like (samme superklasse eller kjent likt EKG-bilde), eller ser
  mønsteret tilfeldig ut?

Hypotesene skrives ned før vi ser resultatene, slik at tolkningen ikke tilpasses i etterkant:

- **H1.** Klassene skiller seg klart bedre enn tilfeldig. Det er ventet, fordi ECGFounder er trent med
  diagnoseetiketter (hint 4), og det er derfor ikke hovedfunnet.
- **H2.** Par innen samme superklasse blandes mer enn par på tvers av superklasser.
- **H3.** Disse parene blandes mest:

| Par | Hvorfor vi venter sammenblanding |
|---|---|
| IMI og AMI | Begge er hjerteinfarkt, bare lokaliseringen er ulik |
| CRBBB og IRBBB | Komplett og inkomplett høyre grenblokk er grader av samme ledningsforstyrrelse |
| LAFB/LPFB og IVCD | Begge er forstyrrelser i ventrikkelledningen |
| NST_, ISC_ og STTC | Alle er ST- og T-forandringer, ofte uspesifikke |
| LVH og ISC_ | Ved venstre ventrikkelhypertrofi kan repolariseringen ligne iskemi |
| NORM og NST_ | Uspesifikke ST-forandringer kan være små |

Den endelige listen over klasser avgjøres etter EDA (avsnitt 3), fordi noen klasser kan ha for få opptak med
én etikett.

## 2 Data

- **Kilde:** PTB-XL 1.0.3 på PhysioNet. Datasettet har 21 799 tolvavlednings-EKG på 10 sekunder fra
  18 869 pasienter, med 71 SCP-koder som er gruppert i 5 superklasser (NORM, MI, STTC, CD, HYP) og
  23 subklasser.
- **Metadata:** `ptbxl_database.csv` (kolonnen `scp_codes` med sannsynlighet per kode, `patient_id` og
  `strat_fold`) og `scp_statements.csv` (koblingen fra kode til diagnoseklasse).
- **Signaler:** 500 Hz-versjonen (`records500`). De strømmes enkeltvis med `wfdb.rdsamp(..., pn_dir=...)`
  (hint 3), og vi laster ikke ned hele zip-filen på 1,7 GB. Rådata lagres i Google Drive, ikke i git.

## 3 Utforskende analyse og utvalg

1. Les metadata og tell opptak per superklasse og subklasse.
2. Vis at datasettet har flere etiketter per opptak: hvor mange opptak har én, to eller flere
   diagnoseklasser?
3. Vis noen eksempelsignaler per klasse, og sjekk alder, kjønn og signalkvalitet (kolonnene for støy og
   baseline drift).

**Utvalgsregler.** De skal begrunnes i rapporten.

- Vi tar bare med opptak med nøyaktig én diagnostisk subklasse, der koden har sannsynlighet
  [100, eller ≥ 50; velg etter EDA]. Da får hvert punkt én etikett, og k-NN og silhouette blir
  meningsfulle. Ulempen er at vi mister de vanskeligste, sammensatte tilfellene. Det nevner vi som en
  begrensning.
- Vi tar ett opptak per pasient. Ellers kan to opptak fra samme person bli naboer og gi kunstig høy
  k-NN-enighet.
- Utvalget stratifiseres med omtrent [150] opptak per klasse for [8–10] klasser, til sammen 1 000–2 000.
  Klasser med færre enn [50] opptak slås sammen eller tas ut.
- Opptak med kraftig støy tas ut, og vi oppgir hvor mange.
- Utvalget trekkes med fast seed og lagres som en liste med `ecg_id`, slik at alle i gruppen får samme
  utvalg.

## 4 Preprosessering og input til modellen

Vi følger preprosesseringen i ECGFounder-koden (`dataset.py`), så input ligner det modellen er trent på:

- Tolv avledninger i rekkefølgen I, II, III, aVR, aVL, aVF, V1–V6. PTB-XL bruker allerede denne
  rekkefølgen, men det skal sjekkes i koden.
- NaN settes til 0.
- Signalet z-normaliseres over hele opptaket, `(x − mean) / (std + 1e-8)`, slik koden gjør. Vi kan teste
  normalisering per avledning som en variant.
- Formen blir `(12, 5000)`. Vi antar at modellen venter 500 Hz og 10 sekunder, men det står ikke i koden vi
  har lest, så det må bekreftes mot artikkelen før vi kjører hele utvalget.
- README nevner filtrering, men koden vi har lest filtrerer ikke. [Avklar om vi skal ha båndpassfilter.]

## 5 Metode

### 5.1 Embeddings

- **Modell:** ECGFounder (Li m.fl., 2025), 12-avledningsversjonen `12_lead_ECGFounder.pth` (370 MB, MIT-lisens)
  fra Hugging Face. Arkitekturen er en 1D-CNN (`Net1D` i `net1d.py`).
- **Henting:** vi bygger `Net1D` med `return_features=True` (hint 2). Da gir `forward` både logits og
  `deep_features`, som er gjennomsnittet over tid etter siste blokk, med 1024 dimensjoner. Vi lagrer både
  embeddingene (1024-D) og logitene (150 klasser).
- Modellen er frosset (`model.eval()`, `torch.no_grad()`), og vi trener ingenting.
- Embeddingene lagres som `.npy` i Drive, så analysen kan kjøres uten GPU.
- **Hvorfor ECGFounder og ikke ECG-FM:** ECG-FM er forhåndstrent på data som inneholder PTB-XL. Da ville
  god separasjon vært sirkulær (hint 1).

### 5.2 Dimensjonsreduksjon

- **PCA:** forklart varians per komponent, 2D-plott og PCA til 50 dimensjoner som input til t-SNE og UMAP.
- **t-SNE:** perplexity [30 og 50], 5 seeds.
- **UMAP:** `n_neighbors` [15 og 50], 5 seeds.

t-SNE og UMAP bevarer lokale naboskap bedre enn globale avstander. Derfor beregner vi tallene både i det
fulle rommet og i 2D, og 2D-plottene brukes bare som illustrasjon.

### 5.3 Tall som støtter plottene

1. **k-NN-etikettenighet:** andelen av de k = 10 nærmeste naboene (cosinusavstand) som har samme klasse,
   totalt og per klasse. Beregnes i 1024-D, PCA-50, PCA-2, t-SNE-2 og UMAP-2.
2. **Silhouette score:** totalt og per klasse, i de samme rommene.
3. **Sammenblandingsmatrise:** for hver klasse a, andelen av naboene som tilhører klasse b. Matrisen gjøres
   symmetrisk, og parene rangeres. Den svarer på F2.
4. **Konsistens på tvers av metodene:** Spearman-korrelasjon mellom sammenblandingsmatrisene (verdiene over
   diagonalen) for hvert metodepar, og hvor mange av de 5 mest blandede parene som går igjen.
5. **Usikkerhet:** bootstrap med 1 000 trekk gir konfidensintervall for k-NN-enighet og silhouette. For
   t-SNE og UMAP oppgir vi spredningen over seeds.
6. **Nullnivå:** vi stokker etikettene om igjen og beregner metrikkene på nytt. Det viser hva tilfeldig
   ville gitt.

### 5.4 Klinisk likhet (F3)

- Vi sammenligner snittet av sammenblandingen for par innen samme superklasse med par på tvers. Forskjellen
  testes med en permutasjonstest der subklassene byttes tilfeldig mellom superklasser.
- Vi sjekker om de forhåndsbestemte parene i H3 ligger høyt på rangeringen.
- For de mest blandede parene viser vi eksempelsignaler og drøfter hvorfor de ligner. [Finn en klinisk
  kilde på EKG-kjennetegn, for eksempel en lærebok i EKG-tolkning.]

### 5.5 Sammenligningsgrunnlag

Vi trenger et sammenligningsgrunnlag for å se hva encoderen bidrar med:

- **Rå baseline:** samme analyse på signalene direkte, nedsamplet til 100 Hz og redusert med PCA. Den viser
  hvor mye som skiller seg uten noen modell.
- **Logit-baseline:** samme analyse på de 150 logitene. Hvis logitene og embeddingene blandes likt, følger
  embeddingrommet treningsetikettene tett.

## 6 Tolkning og begrensninger

Dette skal drøftes i rapporten:

- Vi har valgt bort opptak med flere diagnoser, og de er en stor del av PTB-XL.
- ECGFounders 150 treningsklasser overlapper med PTB-XL-kodene. Det forklarer god separasjon (hint 4).
- Det kan ligge feil i etikettene i PTB-XL. Foldene 9 og 10 er kvalitetssikret av kardiologer, og vi kan
  sjekke om resultatene holder seg der.
- Utvalget er lite, og noen klasser har få opptak.
- Rekonstruksjon av signalet (Øysteins idé) er utenfor hovedplanen, fordi ECGFounder ikke har noen dekoder.
  Det kan bli en utvidelse hvis vi har tid.

## 7 Leveranser og hvordan de dekker vurderingskriteriene

| Kriterium | Hvor det dekkes |
|---|---|
| 01 Alle stegene | Notebooken følger avsnitt 2–6 i rekkefølge: data, EDA, utvalg, preprosessering, embeddings, metode, resultater og tolkning |
| 02 Forskningsspørsmål | Avsnitt 1, med hypoteser skrevet før vi har resultater |
| 03 Data svarer på spørsmålet | Tallene i 5.3 og 5.4 svarer på F1–F3 |
| 04 Dokumentert notebook og alle bidrar | Markdown-celle før hver hovedblokk, og en tabell over hvem som gjorde hva |
| 05 Kilder | Referanselisten i avsnitt 9 |
| 06 Valg begrunnes | Hvert valg i avsnitt 3–5 har en begrunnelse som går igjen i rapporten |

- **Notebook:** Colab, kan kjøres fra start til slutt på en gratis T4.
- **Rapport:** [avklar format og lengde med Mina eller TA-ene].
- **Presentasjon:** maks 10 minutter. Den går gjennom koden blokk for blokk, og hver av oss presenterer
  delen sin.

## 8 Fremdrift og roller

| Uke | Datoer | Mål |
|---|---|---|
| 45 | 3.–9.11 | Møte og plan vedtatt. Roller fordelt. Colab-oppsett. ECGFounder gir embeddings for 10–20 opptak |
| 46 | 10.–16.11 | EDA, utvalg på 1 000–2 000 opptak, alle embeddings lagret. PCA, t-SNE og UMAP, k-NN og silhouette |
| 47 | 17.–23.11 | Sammenblandingsmatrise, konsistens, klinisk analyse og baseline. Rapporten skrives parallelt |
| 48 | 24.11 | Levering (fristen er 25.11 kl. 23.59) |
| 48–49 | 25.11–4.12 | Presentasjonen øves og holdes. Book tid i regnearket, ikke 3.12 |

| Rolle | Hvem |
|---|---|
| Gruppeleder og koordinering | [navn] |
| Data, EDA og utvalg | [navn] |
| ECGFounder og embeddings | [navn] |
| Dimensjonsreduksjon og metrikker | [navn] |
| Klinisk tolkning og kilder | [navn] |
| Rapport og presentasjon | Alle, med [navn] som redaktør |

**Største risiko:** å få ECGFounder til å kjøre og gi riktige features. Derfor gjør vi det først (uke 45).
Andre risikoer er treg strømming fra PhysioNet (vi cacher til Drive) og for få opptak med én etikett i
enkelte klasser (vi slår sammen klasser eller bytter klasser).

## 9 Kilder

- Li, J., Aguirre, A. D., Moura Junior, V., Jin, J., Liu, C., Zhong, L., Sun, C., Clifford, G., Westover,
  M. B., & Hong, S. (2025). An electrocardiogram foundation model built on over 10 million recordings.
  *NEJM AI*, 2(7), AIoa2401033. Preprint: arXiv:2410.04133. Kode: https://github.com/PKUDigitalHealth/ECGFounder
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly
  available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T.
  (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154.
  https://doi.org/10.1038/s41597-020-0495-6
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., &
  Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*.
  https://doi.org/10.1038/s44360-026-00096-z
- [McInnes m.fl. (2018) UMAP, van der Maaten og Hinton (2008) t-SNE, Rousseeuw (1987) silhouette, og en
  klinisk kilde om EKG-tolkning]
