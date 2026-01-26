# Australian Open 2026 — ATP \& WTA Prediction Engine



A full end-to-end machine learning pipeline to predict ATP and WTA Australian Open 2026 outcomes, using historical match data, rolling player form, ATP/WTA rankings, and bracket-aware Monte Carlo simulation.

Past ATP and WTA data through: http://www.tennis-data.co.uk/alldata.php

---



## What this project does



\- Builds separate ML models for ATP and WTA

\- Learns from:

&nbsp; - recent player form (rolling windows)

&nbsp; - match odds

&nbsp; - tour rankings 

\- Simulates the entire Grand Slam bracket round by round

\- Outputs title probabilities for every player

\- Supports both ATP and WTA tours in one pipeline



---



## Modelling approach (high level)



1\. **Raw data ingestion**

&nbsp;  - ATP \& WTA match results (2000–2025)

&nbsp;  - Betting odds (Bet365)

&nbsp;  - Rankings (ATP \& WTA)



2\. **Feature engineering**

&nbsp;  - Rolling form metrics (last \*N\* matches)

&nbsp;  - ATP/WTA Ranking difference (log-scaled)

&nbsp;  - Symmetric winner/loser training rows



3\. **Model**

&nbsp;  - Logistic Regression

&nbsp;  - Time-aware train/test split

&nbsp;  - Separate models for ATP and WTA



4\. **Tournament simulation**

&nbsp;  - Uses real AO 2026 draws

&nbsp;  - Simulates R128 → Final

&nbsp;  - Monte Carlo simulation (10,000 runs)



---



## Model performance (typical)



| Tour | Accuracy | Log Loss |

|------|----------|----------|

| ATP  |   ~63%   |   ~0.66  |

| WTA  |  ~58–61% |   ~0.67  |





