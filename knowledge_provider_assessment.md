# InNOChem Knowledge Provider Assessment

Assessment date: 2026-05-30

Scope: Evaluate candidate datasets for future InNOChem knowledge-provider integration. This assessment does not implement providers, recommendations, RAG, or AI functionality.

## Executive Recommendation

| Rank | Dataset | Recommendation | Business Value |
|---:|---|---|---|
| 1 | PubChemAnnotationTopics_Hazardous-Substances-Data-Bank-HSDB.json | Core Provider, using PubChem/HSDB annotation retrieval as the real data source | Highest |
| 2 | Chemical Components Hazard Carbon Ranking.xlsx | Derived Dataset, supplemental only after provenance validation | Medium |
| 3 | Chemical Components Carbon Footprint Hazard Ranking.xlsx | Reject as provider; keep only as prototype/reference | Low |

The HSDB topic catalog is the strongest strategic fit because it points to authoritative, compound-level safety, regulatory, toxicology, exposure, and environmental intelligence through PubChem annotations. The two carbon/hazard spreadsheets are useful concept prototypes, but they appear to be generated derivative workbooks with limited provenance traceability.

## Dataset Coverage Summary

| Dataset | Row Count | Chemical Count | CAS Coverage | CID Coverage | Hazard Coverage | Sustainability Coverage | Regulatory Coverage | Carbon Footprint Coverage | Provenance Quality |
|---|---:|---:|---|---|---|---|---|---|---|
| HSDB PubChem annotation topics JSON | 148 topic rows | No direct compound rows; largest topic count 6,060 HSDB records | Topic metadata only; CAS topic count 5,934 | Topic metadata only; PubChem compound annotations implied, no per-CID rows in file | Strong topic coverage | Strong environmental fate/ecotoxicity topics | Strong US regulatory topics | None direct | High for publisher/source, low for compound-level extraction from this file alone |
| Chemical Components Hazard Carbon Ranking.xlsx | 200 primary compound rows | 200 unique CAS values | 200/200 | None | Strong derived score/classification fields | Moderate environmental fate/PBT fields | Moderate regulatory-status field | 200/200 | Medium-low; source list present, row-level citations absent |
| Chemical Components Carbon Footprint Hazard Ranking.xlsx | 25 primary compound rows | 25 named compounds | None | None | Moderate derived score/classification fields | Weak | Weak/none | 25/25 | Low; source list present, no row-level citations or identifiers |

## Provider Assessments

### PubChem HSDB Annotation Topics

Provider Name: HSDB PubChem Annotation Provider

Provider Type: Core Provider

Source: `PubChemAnnotationTopics_Hazardous-Substances-Data-Bank-HSDB.json`

Publisher: PubChem / Hazardous Substances Data Bank (HSDB). HSDB historically comes from the U.S. National Library of Medicine; PubChem now exposes HSDB-derived annotation topics.

Methodology: The provided file is a catalog of PubChem annotation topics, not compound-level facts. Each row has a topic heading, type, description, and count of compounds with that annotation. A future provider should use this topic catalog to select high-value PubChem annotation headings, then retrieve compound-level annotations by CID.

Coverage:

| Metric | Assessment |
|---|---|
| Row count | 148 annotation topics |
| Chemical count | No direct chemical rows in file. Topic counts indicate up to 6,060 HSDB-numbered compound records. |
| CAS coverage | CAS topic count: 5,934 compounds. This file does not contain the CAS values themselves. |
| CID coverage | No explicit CID rows in file. PubChem annotation retrieval should provide CID-linked facts. |
| Hazard coverage | Very strong. Includes GHS Classification, NFPA Hazard Classification, toxicity values, carcinogenicity, IDLH, TLV, emergency treatment, reactivity, irritation, and toxicity summaries. |
| Sustainability coverage | Strong for environmental fate, biodegradation, bioconcentration, ecotoxicity, soil mobility, volatilization, water concentrations, and pollution sources. |
| Regulatory coverage | Strong for OSHA, TSCA, CERCLA, RCRA, DOT, CWA, FDA, drinking-water standards, shipping regulations, and other standards. |
| Carbon footprint coverage | None direct. |
| Provenance quality | Strong publisher provenance and topic descriptions. Compound-level provenance depends on retrieving the actual PubChem annotation records, not this topic list alone. |

Strengths:

- Authoritative and high-value for safety, toxicity, exposure, emergency response, regulatory intelligence, and environmental fate.
- Naturally fits current PubChem-centric validation pipeline through CID-linked annotations.
- Provides many topic-level counts, which helps prioritize implementation.
- Strong long-term value for compliance and explainable chemical intelligence.

Weaknesses:

- Current JSON is only a topic catalog, not the fact payload.
- Requires a future annotation retrieval layer to fetch per-CID facts.
- Topic payloads may be text-heavy and will need careful normalization, evidence retention, and citation display.
- Not a carbon-footprint data source.

Recommended Integration Priority: Highest. Implement as a Core Provider after the current provider interface is ready for PubChem annotation retrieval.

### Chemical Components Hazard Carbon Ranking.xlsx

Provider Name: Chemical Components Hazard Carbon Ranking

Provider Type: Derived Dataset

Source: `Chemical_Components_Hazard_Carbon_Ranking.xlsx`

Publisher: No authoritative publisher identified in workbook metadata. Workbook creator is `openpyxl`; source line cites IPCC AR6, EPA GHG, GHS/CLP, IARC Monographs, NFPA 704, Montreal Protocol, Stockholm Convention, ECHA SVHC, and WFD.

Methodology: Derived ranking spreadsheet. Primary sheet contains 200 compounds with formula, CAS, category, carbon footprint, GWP-100, acute toxicity, carcinogenicity, NFPA flammability/reactivity, ozone depletion potential, PBT flag, environmental fate, regulatory status, overall hazard score, risk level, and notes. Workbook includes category summaries and top-50 derived views. Methodology sheet defines risk levels and broad score logic, but row-level citations are not present.

Coverage:

| Metric | Assessment |
|---|---|
| Row count | 200 primary compound rows; additional summary/top-ranking sheets are derived views. |
| Chemical count | 200 compounds with unique CAS values. |
| CAS coverage | 200/200 primary rows. |
| CID coverage | None. |
| Hazard coverage | Strong within the spreadsheet: GHS acute toxicity, IARC carcinogen group, NFPA flammability/reactivity, PBT, ODP, hazard score, risk level. |
| Sustainability coverage | Moderate: environmental fate, PBT, ODP, carbon footprint, GWP. |
| Regulatory coverage | Moderate: regulatory status field populated for 200 rows. |
| Carbon footprint coverage | Strong within file: 200/200 rows have kg CO2-eq/kg and GWP-100. |
| Provenance quality | Medium-low: source families are listed, but values are not row-level cited and workbook metadata indicates generated/openpyxl output. |

Strengths:

- Broad and compact coverage across hazard and carbon dimensions.
- Full CAS coverage makes it easy to match against validated chemicals.
- Useful for prototyping carbon/hazard display and derived screening experiments.
- Includes both compound-level rows and aggregate/top-ranking views.

Weaknesses:

- Not an authoritative primary source as provided.
- No CID coverage.
- No row-level provenance or citations for individual hazard/carbon values.
- Overall hazard score and risk level are derived rankings; they should not be treated as official classifications.
- Potential overlap with existing deterministic scoring may confuse users unless clearly labeled as a derived dataset.

Recommended Integration Priority: Medium. Use only as a Derived Dataset or Supplemental Provider after adding explicit provenance warnings and preferably validating values against primary sources.

### Chemical Components Carbon Footprint Hazard Ranking.xlsx

Provider Name: Chemical Components Carbon Footprint Hazard Ranking

Provider Type: Reject

Source: `Chemical_Components_Carbon_Footprint_Hazard_Ranking.xlsx`

Publisher: No authoritative publisher identified in workbook metadata. Workbook creator is `openpyxl`; source line cites IPCC AR6, EPA GHG Reporting, GHS/CLP Hazard Classification, ECHA Database, and NIST Chemistry WebBook.

Methodology: Small derived ranking workbook with 25 compounds. Includes carbon footprint, GWP, GWP rating, GHS acute toxicity label, IARC group, NFPA flammability/reactivity, ODP, PBT, overall hazard score, and risk level. Contains dashboard and methodology sheets.

Coverage:

| Metric | Assessment |
|---|---|
| Row count | 25 primary compound rows. |
| Chemical count | 25 named compounds. |
| CAS coverage | None. |
| CID coverage | None. |
| Hazard coverage | Moderate derived fields. |
| Sustainability coverage | Weak beyond GWP/carbon and PBT/ODP. |
| Regulatory coverage | Weak/none in primary data. |
| Carbon footprint coverage | 25/25 rows. |
| Provenance quality | Low: no row-level provenance, no identifiers, generated workbook metadata. |

Strengths:

- Simple prototype for presenting carbon-footprint and hazard-score concepts.
- Contains methodology/legend text that can inform UX copy for a future derived scoring display.

Weaknesses:

- Too small for provider use.
- No CAS or CID identifiers, making reliable matching difficult.
- Appears superseded by the 200-compound workbook.
- Derived scores without row-level evidence should not be exposed as authoritative intelligence.

Recommended Integration Priority: Low. Reject as a provider. Keep only as an internal concept/reference if needed.

## HSDB High-Value Topic Priorities

The HSDB JSON contains 148 annotation topics. The following topics are especially useful for future InNOChem intelligence.

| Topic | Count | Best Use |
|---|---:|---|
| GHS Classification | 680 | Safety Intelligence, Regulatory Intelligence |
| Evidence for Carcinogenicity | 1,441 | Safety Intelligence, Regulatory Intelligence |
| NFPA Hazard Classification | 840 | Safety Intelligence |
| NIOSH Recommendations | 1,006 | Safety Intelligence, Regulatory Intelligence |
| OSHA Standards | 851 | Regulatory Intelligence, Safety Intelligence |
| TSCA Requirements | 605 | Regulatory Intelligence |
| CERCLA Reportable Quantities | 926 | Regulatory Intelligence |
| DOT Emergency Guidelines | 1,310 | Safety Intelligence, Regulatory Intelligence |
| Human Toxicity Values | 189 | Safety Intelligence |
| Human Toxicity Excerpts | 4,437 | Safety Intelligence |
| Non-Human Toxicity Values | 3,124 | Safety Intelligence, Sustainability Intelligence |
| Non-Human Toxicity Excerpts | 5,007 | Safety Intelligence, Sustainability Intelligence |
| Ecotoxicity Values | 1,757 | Sustainability Intelligence |
| Ecotoxicity Excerpts | 1,202 | Sustainability Intelligence |
| Environmental Fate | 3,301 | Sustainability Intelligence, Alternative Discovery |
| Environmental Fate/Exposure Summary | 3,200 | Sustainability Intelligence, Alternative Discovery |
| Environmental Biodegradation | 2,373 | Sustainability Intelligence, Alternative Discovery |
| Environmental Bioconcentration | 3,182 | Sustainability Intelligence |
| Soil Adsorption/Mobility | 3,148 | Sustainability Intelligence |
| Volatilization from Water/Soil | 3,127 | Sustainability Intelligence |
| LogP | 2,143 | Sustainability Intelligence, Alternative Discovery |
| Hazardous Reactivities and Incompatibilities | 2,341 | Safety Intelligence, Alternative Discovery |
| Antidote and Emergency Treatment | 4,178 | Safety Intelligence |
| Disposal Methods | 5,017 | Safety Intelligence, Sustainability Intelligence |
| EPA Hazardous Waste Number | 649 | Regulatory Intelligence, Sustainability Intelligence |
| RCRA Requirements | 621 | Regulatory Intelligence |
| Shipment Methods and Regulations | 1,678 | Regulatory Intelligence |
| Shipping Name/ Number DOT/UN/NA/IMO | 1,413 | Regulatory Intelligence |

### Regulatory Intelligence

Highest-value topics:

- OSHA Standards
- TSCA Requirements
- CERCLA Reportable Quantities
- RCRA Requirements
- EPA Hazardous Waste Number
- DOT Emergency Guidelines
- Shipment Methods and Regulations
- Shipping Name/ Number DOT/UN/NA/IMO
- Clean Water Act Requirements
- FDA Requirements
- Federal Drinking Water Standards

Why: These topics can support compliance flags, jurisdiction-aware warnings, reporting obligations, transportation constraints, and waste-handling constraints.

### Safety Intelligence

Highest-value topics:

- GHS Classification
- NFPA Hazard Classification
- Evidence for Carcinogenicity
- Human Toxicity Values
- Human Toxicity Excerpts
- Non-Human Toxicity Values
- Immediately Dangerous to Life or Health (IDLH)
- Threshold Limit Values (TLV)
- NIOSH Recommendations
- OSHA Standards
- Antidote and Emergency Treatment
- Fire Fighting Procedures
- Hazardous Reactivities and Incompatibilities

Why: These topics support practical safety review, hazard communication, workplace exposure checks, acute response, and incompatibility screening.

### Sustainability Intelligence

Highest-value topics:

- Ecotoxicity Values
- Ecotoxicity Excerpts
- Environmental Fate
- Environmental Fate/Exposure Summary
- Environmental Biodegradation
- Environmental Bioconcentration
- Soil Adsorption/Mobility
- Volatilization from Water/Soil
- Environmental Water Concentrations
- Artificial Pollution Sources
- Disposal Methods
- EPA Hazardous Waste Number

Why: These topics help describe environmental persistence, mobility, bioaccumulation, disposal burden, and ecological toxicity. HSDB does not directly provide carbon footprint, so it should complement, not replace, carbon datasets.

### Alternative Discovery

Highest-value topics:

- Associated Chemicals
- Uses
- Methods of Manufacturing
- General Manufacturing Information
- LogP
- Solubility
- Environmental Fate
- Environmental Biodegradation
- Environmental Bioconcentration
- Hazardous Reactivities and Incompatibilities
- Disposal Methods

Why: These topics can help compare candidate alternatives by use context, structural/associated chemicals, physicochemical properties, process relevance, persistence, and incompatibility. They should support discovery filters, not automated recommendations yet.

## Integration Guidance

1. Build HSDB as the next serious provider, but use the JSON file only as a topic allowlist and priority map.
2. Retrieve compound-level HSDB facts by PubChem CID and selected topic heading.
3. Preserve source topic, PubChem CID, evidence text, and PubChem/HSDB provenance for every fact.
4. Keep HSDB facts separate from deterministic scoring until the team defines explicit decision rules.
5. Treat the 200-compound carbon/hazard workbook as a derived supplemental dataset only after adding warning labels and row-level provenance checks.
6. Reject the 25-compound workbook as a provider because it lacks CAS/CID identifiers and is superseded by the larger workbook.

## Final Business-Value Ranking

1. HSDB PubChem Annotation Provider: highest value because it unlocks authoritative safety, toxicology, regulatory, emergency, and environmental intelligence for validated PubChem compounds.
2. Chemical Components Hazard Carbon Ranking.xlsx: medium value as a compact derived carbon/hazard dataset with CAS matching, but insufficient row-level provenance for authoritative decisions.
3. Chemical Components Carbon Footprint Hazard Ranking.xlsx: low value; useful only as an early prototype artifact, not a provider.
