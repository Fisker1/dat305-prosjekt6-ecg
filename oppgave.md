# Oppgaven

Kopiert ordrett fra kursmateriellet i DAT305 (høst 2026): prosjekt 6 fra `DAT305_Projects.pdf` og
retningslinjene fra `DAT305 Project Guidelines.pdf`.

## Project 6: What does a pretrained ECG model actually see?

You will pick a subset of PTB-XL (say 1,000–2,000 records - small enough to be comfortable on a free-tier T4
Colab GPU, large enough that the clustering result means something) and run it through one frozen pretrained
ECG encoder to get embeddings. Then PCA plus at least one of t-SNE/UMAP, backed by an actual number (k-NN label
agreement or silhouette score), not just a scatter plot and call it evidence.

The real ask is going past "yes it separates" into characterizing where it doesn't: which diagnostic pairs
blur together, and whether that's consistent across methods.

**Dataset:** PTB-XL for signals and labels.
https://physionet.org/content/ptb-xl/1.0.3/
https://physionet.org/content/ptb-xl/1.0.1/ptbxl_database.csv

### Hints

1. Use ECGFounder for the extraction, not ECG-FM, ECG-FM's own pretraining corpus includes PTB-XL, which
   would make any separation finding circular.
2. ECGFounder's own public eval script only exposes classification logits, not embeddings. Getting a feature
   vector out means calling the underlying network with return_features=True and pulling the intermediate
   features instead of the class-probability output, the public script alone won't show them this.
3. No need to download the full 1.7 GB PTB-XL signal zip for a 1–2k subset. wfdb's reader can stream
   individual records straight from PhysioNet's server with the pn_dir argument instead of pulling the whole
   database locally first.
4. The encoder was trained with diagnostic labels, so class separation by itself isn't surprising. What's
   worth reporting is which classes blur together despite that supervision, and whether it lines up with
   clinical similarity (e.g. classes that look alike on paper) or looks arbitrary.

### References

- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly
  available electrocardiography dataset (version 1.0.3). PhysioNet. RRID:SCR_007345.
  https://doi.org/10.13026/kfzx-aw45
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T.
  (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data.
  https://doi.org/10.1038/s41597-020-0495-6
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., &
  Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health.
  https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM

## DAT305 Project Guidelines

The objective of group projects in this course is to solve problems by working with real-world datasets. The
aim is to think like a data scientist and go through all the steps of a typical data science project.

During this project, the student will learn about designing a research plan, preprocessing data, preparing
input for the model, selecting an algorithm and training the model, interpreting the results, and evaluating
and reporting the outcomes scientifically.

### General Rules

- Every project has a distinct dataset, and you must perform the exploratory analysis necessary for the final
  task.
- Each group may consist of 2–5 students (i.e., two, three, four, or five students). In exceptional cases,
  individual students may be allowed to work alone, subject to approval. If you have a specific reason for
  working individually or need to discuss your group arrangement, please contact Mina.
- Sign up for the group on the spreadsheet. A group leader is assigned within each group and is responsible
  for teamwork and coordination.
- You can contact the TAs to meet him/her for refined consultation.

### Project Difficulty and Grading

- All projects will be considered equivalent in terms of difficulty for approval purposes. Choosing a more
  advanced project will not give you additional credit.
- However, if you would like to challenge yourself and explore a more advanced topic, you are welcome to
  choose one of the advanced projects.

### General Assessment Criteria

1. Did the group complete all the steps of the data science project?
2. Did the group formulate a valid research question?
3. Did the group use the data to try to answer the research question? Is the presentation self-contained?
4. Is the notebook submission fully explained and well-documented? Did the entire group contribute to the
   task?
5. Does the work cite relevant sources, and are the authorities acknowledged correctly?
6. Do the report and presentation clearly explain the various aspects of the project and justify the
   decisions made?

### Key Notes for online presentation

- Maximum Duration: 10 minutes
- Focus on demonstrating and explaining the code, not repeating the written report
- Show how the code works, what each major block does.
