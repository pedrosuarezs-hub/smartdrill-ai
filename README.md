# SmartDrill AI

Functional Streamlit research prototype for adaptive mineral resource-definition drilling.

## Current capabilities

- Built-in synthetic copper demo dataset
- Upload COLLAR.csv, optional SURVEY.csv, ASSAY.csv and optional HISTORY.csv
- Validate drilling tables
- Convert assay midpoints to 3D coordinates
- Run local ordinary kriging
- Display estimated Cu and kriging-variance diagnostic slices
- Generate vertical candidate drillholes
- Rank candidates with the frozen Phase 7 active-learning acquisition function
- Recommend the next information-rich drillhole
- Evaluate the autonomous CONTINUE / STOP rule

## Run locally

1. Create a Python virtual environment.
2. Install dependencies with: pip install -r requirements.txt
3. Run: streamlit run app.py

## Deploy on Streamlit Community Cloud

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Create a new app.
3. Choose app.py as the main file.
4. Deploy.

## Upload schemas

COLLAR.csv:
BHID, XCOLLAR, YCOLLAR, ZCOLLAR

SURVEY.csv:
BHID, AT_M, AZIMUTH, DIP

ASSAY.csv:
BHID, FROM_M, TO_M, CU_PCT

HISTORY.csv:
ITERATION, MEAN_VAR, MEAN_GRADE, METAL_PROXY

## Frozen research settings

Acquisition weights:
- global variance reduction: 0.20
- threshold entropy reduction: 0.20
- grade-weighted uncertainty reduction: 0.60

Autonomous stopping:
- marginal uncertainty reduction <= 1.8%
- mean-grade change <= 2.0%
- metal-proxy change <= 3.0%
- persist for 3 consecutive updates
- do not stop before iteration 4

## Important limitations

SmartDrill AI is research software, not a mineral-resource reporting package. Current v2 limitations include vertical candidate holes, prototype desurvey approximation, heuristic covariance parameters, no geological domains, and no density model.

The 0.20% Cu value used by the demo is a research threshold, not an economic cutoff grade.
