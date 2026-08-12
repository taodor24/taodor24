## Fedor Turbin

AI/ML Engineer building verifiable ML systems for medical and biotech data.

I focus on the part of applied AI that actually ships: grounding model output
in real sources, validating it against schemas, and quantifying uncertainty
instead of hiding it — because in healthcare, "the model probably got it
right" isn't good enough.

**Background** — Biomedical Engineering BSc (240 ECTS — medical devices, biosignal processing, 27 ECTS of mathematics) + École 42 engineering core (C, Python, Docker, Linux). 
I can read the biology and write the code — in applied healthcare ML that combination is the bottleneck more often than the modelling is.


### Selected work

**[molecular-generalization-gap](https://github.com/taodor24/molecular-generalization-gap)**
Does scaffold splitting actually measure generalization? Quantifies how much
random train/test splits overstate performance on BBBP and BACE, and shows
model error tracks continuously with Tanimoto distance to the training set —
with cluster bootstrap CIs that reveal the gap is directionally consistent but
not resolvable at benchmark scale.
`RDKit` `scikit-learn` `Morgan fingerprints` `Bemis-Murcko scaffolds` `bootstrap CI`

**[esm2-composition-probe](https://github.com/taodor24/esm2-composition-probe)** 
What does a protein language model actually encode?
Amino acid frequencies alone hit 0.901 AUC on subcellular localization; ESM-2 hits 0.974.
Projecting out the composition subspace leaves 0.836 against a 0.548 floor, and the gap
survives a family-aware split — so the advantage is neither composition nor paralog memory.
`ESM-2` · `PyTorch` · `scikit-learn` · `UniProt API` · `paired bootstrap` · `group-aware CV`

**[calibration-noise-floor](https://github.com/taodor24/calibration-noise-floor)** 
Most of the miscalibration I measured was a default hyperparameter
Started as a study of how calibration drifts under a ten-year temporal shift. sklearn's
default C=1 turned out to swing measured ECE by 5-6x, and a null-ECE check showed what
remained on the pre-shift test set was indistinguishable from finite-sample noise — so
recalibration made it worse rather than better.
`scikit-learn` · `TF-IDF` · `null-ECE simulation` · `bootstrap CI` · `NCBI E-utilities`



### Stack

Python · scikit-learn · pandas · NumPy · RDKit · FastAPI · Pydantic · Docker · Git · Linux · C

Currently building with: LangChain · RAG · Qdrant · OpenAI & Anthropic APIs

---
France · [LinkedIn](https://linkedin.com/in/fedor-turbin) · 
Open to AI/ML Engineer roles in France and the EU
