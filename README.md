## Fedor Turbin

AI/ML Engineer building verifiable ML systems for medical and biotech data.

I focus on the part of applied AI that actually ships: grounding model output
in real sources, validating it against schemas, and quantifying uncertainty
instead of hiding it — because in healthcare, "the model probably got it
right" isn't good enough.

**Background** — BSc Biomedical Systems (biosignals, medical data) · École 42
(C, Python, Docker, Linux)

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
ESM-2 · PyTorch · scikit-learn · UniProt API · paired bootstrap · group-aware CV



### Stack

Python · scikit-learn · pandas · NumPy · RDKit · FastAPI · Pydantic · Docker · Git · Linux · C

Currently building with: LangChain · RAG · Qdrant · OpenAI & Anthropic APIs

---
France · [LinkedIn](https://linkedin.com/in/fedor-turbin) · 
Open to AI/ML Engineer roles in France and the EU
