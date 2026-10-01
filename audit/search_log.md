# Search log: targeted audit of reported evaluation procedures

**Type of study.** This is a targeted audit of the evaluation procedures reported by selected studies. It is not a systematic review. No protocol was preregistered, no second independent screener was used, and completeness is not claimed.

**Question.** For studies that classify ASD vs. TD using the public eye-tracking scanpath-image release (Carette et al.; Figshare 7073087; 547 images from 59 children), what do the papers report about:
- the split unit;
- when augmentation was applied relative to the split;
- how models, epochs and configurations were selected;
- their headline performance?

## Dates
- **Search and screening:** 30 September 2026.
- **PDF verification:** 30 September 2026. Every coded field was checked against the source PDFs, with the verbatim quote and PDF page recorded (`literature_audit_pdf_verification.csv`, `verification_notes.md`).

## Sources and queries run by the lead analyst
The web search engine was used (general web, US-only). These are the exact query strings:

1. `"eye-tracking scanpath" autism image dataset Carette 2019 figshare classification deep learning`
2. `"Visualization of eye-tracking scanpaths in autism spectrum disorder" image dataset cited by CNN accuracy`
3. `autism eye tracking scanpath images 547 images 59 children transfer learning accuracy`
4. `autism scanpath image dataset Carette "subject-independent" OR "participant-level" OR "leave-one-subject-out" data leakage`
5. `Carette Elbattah autism eye-tracking scanpath images CNN VGG16 accuracy 2023 2024 2025`
6. `"scanpath" images autism "Figshare" ResNet OR "vision transformer" classification 2025 eye-tracking dataset Cilia`
7. `autism eye-tracking scanpath images Kaggle "ETSP" dataset deep learning accuracy 99%`
8. `Benabderrahmane autism eye tracking scanpath arXiv 2026`
9. `Mousli OR Al-Adhaileh OR Alsaidi autism eye-tracking scanpath images CNN`
10. `Cilia 2021 "Computer-Aided Screening of Autism Spectrum Disorder" JMIR Human Factors e27706 PMC`

**Citation chaining.** References cited by Zhang et al. (2026, J Eye Mov Res) as prior work on this dataset were followed: Cilia 2021, Kanhirakadavath 2022, Alsaidi 2024, Benabderrahmane 2024, Mousli 2025 and Al-Adhaileh 2025.

**Assisted search.** Extraction assistants also ran web searches to locate full texts and up to four further studies. They added Thanarajan 2023, Kaloforidis 2025 and Syukron 2026. Their exact query strings were **not logged**. This is a limitation of the audit trail.

## Eligibility (fixed before recounting)
- **Included in counts:**
  - the full text was read;
  - the paper's own analytic sample is the 547-image release;
  - the task is ASD vs. TD classification.
  - Preprints are eligible.
- **Coding:**
  - a code is assigned only from an explicit statement in the paper; otherwise "not reported";
  - codes describe the headline analysis, and secondary analyses are recorded separately;
  - one paper's statements about another study are not used as evidence for that study.

## Screening outcome
**Included in counts (n = 15):**
- Carette 2019
- Cilia 2021
- Elbattah 2021 (VAE)
- Elbattah 2022 (transfer learning)
- Kanhirakadavath 2022
- Ahmed IA 2022
- Thanarajan 2023
- Islam 2024 (preprint)
- Alsaidi 2024
- Alsharif 2024
- Gawish 2025
- Kaloforidis 2025
- Mousli 2025
- Ahmed M 2026
- Zhang 2026

**Excluded, with reason:**

| Study | Reason |
|---|---|
| Jaradat 2025 | Its own analytic sample is 534 images (215 ASD / 319 non-ASD), not the 547-image release |
| Benabderrahmane 2024 | Uses the raw CSV release with self-generated gaze maps, not the image release |
| Al-Adhaileh 2025 | Tabular raw data (Figshare 20113592) |
| Soloh 2025 | Self-generated scanpath images |
| Kasri 2025 | Saliency4ASD dataset |
| Mumenin 2023; Kabir Mehedi 2023; Akshith 2025; Ahmed ZAT 2026; Syukron 2026 | Full text not obtained (abstract or secondary source only) |
| Ahmed & Jadhav 2020 | Not accessible |

## Known limitations
- A single database-free web search was used; Scopus, Web of Science and Google Scholar were not searched.
- Five potentially eligible studies were not coded because their full text was not obtained.
- Papers were coded by the analyst and checked by PDF verification. No blinded dual coding was done.
- "Not reported" means that no explicit statement was found. It is not evidence that leakage occurred.
