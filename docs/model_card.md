# Model card: credit default probability

Format follows Mitchell et al. (2019), "Model Cards for Model Reporting." Every quantitative statement below is filled from `results/tables/` by `python -m src.run_all`. This card describes a research model. It is not a lending policy and it is not a conformity assessment.

Seeds: [42, 43, 44, 45, 46]. Bootstrap resamples: 1000. Fast mode: False. This run uses the full cleaned dataset. 

## Model details

- Models: L2 logistic regression (primary) and an XGBoost classifier with fixed hyperparameters from `configs/config.yaml`. Differentially private logistic regression uses IBM diffprivlib.
- Training objective: probability of default next month. The positive class is default (`default = 1`).
- Decision rule: the threshold maximizes F1 of the default class on the validation split, separately for each fitted model. The primary full-feature logistic threshold is stated in the results paragraph below.
- Recommended configuration selected by the written rule `smallest_epsilon_within_auc_tolerance_and_eo_no_worse`: epsilon 10 on feature set full, reweighing=False. Smallest epsilon whose mean ROC-AUC is within tolerance of its paired non-private model and whose mean sex equalized-odds difference is no worse Mean ROC-AUC 0.7290; mean sex equalized odds difference 0.0605.
- License and contact: course research project. No deployer is identified because the model is not released for use.

## Intended use

The model is intended for a graduate audit of fairness, explanation, and privacy on a public credit-default dataset, including a comparison with the questions raised by the New York Department of Financial Services Apple Card investigation.

Out of scope:

- Deciding whether a person receives credit, a credit limit, or a price.
- Any production underwriting, marketing, or collection system.
- Use outside the population and period represented by this dataset.
- Compliance sign-off under the EU AI Act or fair-lending law.

## Factors

The audit treats recorded sex and age as the relevant factors. Sex in the source file is a binary code (1 male, 2 female). Age is grouped as under 30, 30–44, and 45 and older, and a binary under-30 attribute is used for reweighing. Education and marriage are model inputs. The no-sex feature set is the fairness-through-unawareness comparison. The no-sex-no-age feature set removes both attributes.

## Metrics

- Performance: accuracy, balanced accuracy, ROC-AUC, PR-AUC, precision, recall, F1, and Brier score. The positive class is default.
- Favorable decision for demographic parity: approval, defined as predicted non-default.
- Equalized odds: the larger of the true-positive-rate gap and the false-positive-rate gap for the default class.
- Uncertainty: percentile bootstrap intervals on the primary test split for the baseline audit, and mean ± sample standard deviation across seeds for reweighing and differential privacy.
- Explanation: mean absolute SHAP, Spearman rank correlation, top-k Jaccard overlap, and mean cosine similarity.
- Privacy: the epsilon argument of diffprivlib's logistic regression, which is not an end-to-end guarantee once preprocessing is included.

## Evaluation data

Seeds: [42, 43, 44, 45, 46]. Bootstrap resamples: 1000. Fast mode: False. This run uses the full cleaned dataset. 

The evaluation split is 20% of the modeling frame, stratified on default × sex. The validation split, also 20%, is used only to choose the F1 threshold. Test labels are not used to fit the preprocessor, the model, the reweighing weights, or the threshold.

## Training data

UCI Default of Credit Card Clients (Yeh and Lien 2009): 30,000 Taiwanese credit-card clients observed in 2005, when the full file is used. Default is the recorded payment outcome, not a credit-limit decision. Education codes 0, 5, and 6 are mapped to 4 (other). Marriage code 0 is mapped to 3 (other). `PAY_0` is kept because the source schema has no `PAY_1`.

| group_type | group | n | share | default_rate |
| --- | --- | --- | --- | --- |
| overall | all | 30000 | 1.0000 | 0.2212 |
| sex | female | 18112 | 0.6037 | 0.2078 |
| sex | male | 11888 | 0.3963 | 0.2417 |
| age_group | <30 | 9618 | 0.3206 | 0.2284 |
| age_group | 30-44 | 15096 | 0.5032 | 0.2083 |
| age_group | 45+ | 5286 | 0.1762 | 0.2448 |


## Quantitative analyses

### Overall performance

On the modeling frame of 30000 rows, the default rate is 0.2212 (22.12%). By sex, female n=18112, default rate 0.2078 (20.78%); male n=11888, default rate 0.2417 (24.17%). By age, <30 n=9618, default rate 0.2284 (22.84%); 30-44 n=15096, default rate 0.2083 (20.83%); 45+ n=5286, default rate 0.2448 (24.48%). The primary logistic regression on all features has test ROC-AUC 0.7286, F1 0.5235, accuracy 0.7855, precision 0.5146, recall 0.5328, Brier score 0.1442, and a validation-chosen default threshold of 0.2712. Removing sex changes ROC-AUC to 0.7269 and F1 to 0.5117. The tree model on all features has ROC-AUC 0.7896 and F1 0.5580. For sex, on the full logistic model, the demographic parity difference is 0.0755 (95% bootstrap CI 0.0536 to 0.0975; the interval excludes 0) and the equalized odds difference is 0.0818 (95% bootstrap CI 0.0497 to 0.1327; the interval excludes 0). The approval-rate ratio is 0.9057, which meets the four-fifths threshold of 0.80. Approval rates are female 0.8009 (80.09%), male 0.7254 (72.54%). After sex is removed, the demographic parity difference is 0.0222 (95% bootstrap CI 0.0035 to 0.0455; the interval excludes 0) and the equalized odds difference is 0.0260 (95% bootstrap CI 0.0040 to 0.0798; the interval excludes 0); the approval ratio is 0.9710. A logistic model predicting sex from the no-sex features has test ROC-AUC 0.5873, which is above the weak-proxy cutoff in config, so the attribute is partly recoverable. The strongest training-set associations with sex are AGE (point_biserial 0.0910); PAY_2 (point_biserial 0.0670); PAY_3 (point_biserial 0.0663); PAY_4 (point_biserial 0.0573); PAY_5 (point_biserial 0.0535). For age group, on the full logistic model, the demographic parity difference is 0.0959 (95% bootstrap CI 0.0669 to 0.1310; the interval excludes 0) and the equalized odds difference is 0.1188 (95% bootstrap CI 0.0735 to 0.1952; the interval excludes 0). Approval rates by age are <30 0.7789 (77.89%), 30-44 0.7919 (79.19%), 45+ 0.6960 (69.60%). With sex and age removed, the age-group demographic parity difference is 0.0402 and the equalized odds difference is 0.0888 (95% CI 0.0311 to 0.1627). A model predicting age under 30 from the remaining features has test ROC-AUC 0.8026, which is at or above the moderate-proxy cutoff in config, so the attribute is recoverable from the other inputs. The strongest associations with being under 30 are MARRIAGE (cramers_v 0.4222); LIMIT_BAL (point_biserial -0.2283); EDUCATION (cramers_v 0.1201); PAY_3 (point_biserial 0.0889); PAY_2 (point_biserial 0.0874).

| model | feature_set | roc_auc | pr_auc | f1 | recall | precision | accuracy | brier | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | full | 0.7286 | 0.5063 | 0.5235 | 0.5328 | 0.5146 | 0.7855 | 0.1442 | 0.2712 |
| tree | full | 0.7896 | 0.5787 | 0.5580 | 0.5923 | 0.5275 | 0.7925 | 0.1305 | 0.2727 |
| lr | no_sex | 0.7269 | 0.5052 | 0.5117 | 0.5365 | 0.4890 | 0.7735 | 0.1443 | 0.2635 |
| tree | no_sex | 0.7913 | 0.5810 | 0.5555 | 0.5810 | 0.5321 | 0.7943 | 0.1301 | 0.2792 |
| lr | no_sex_no_age | 0.7272 | 0.5058 | 0.5241 | 0.5154 | 0.5331 | 0.7930 | 0.1443 | 0.2744 |
| tree | no_sex_no_age | 0.7904 | 0.5798 | 0.5538 | 0.6262 | 0.4964 | 0.7768 | 0.1306 | 0.2485 |


### Group gaps

| model | feature_set | attribute | metric | value | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| lr | full | sex_label | dp_difference | 0.0755 | 0.0536 | 0.0975 |
| lr | full | sex_label | dp_ratio | 0.9057 | 0.8790 | 0.9324 |
| lr | full | sex_label | eo_difference | 0.0818 | 0.0497 | 0.1327 |
| lr | full | sex_label | four_fifths_pass | 1.0000 |  |  |
| lr | full | age_group | dp_difference | 0.0959 | 0.0669 | 0.1310 |
| lr | full | age_group | dp_ratio | 0.8789 | 0.8374 | 0.9147 |
| lr | full | age_group | eo_difference | 0.1188 | 0.0735 | 0.1952 |
| lr | full | age_group | four_fifths_pass | 1.0000 |  |  |
| lr | full | sex_age | dp_difference | 0.1803 | 0.1343 | 0.2308 |
| lr | full | sex_age | dp_ratio | 0.7819 | 0.7229 | 0.8361 |
| lr | full | sex_age | eo_difference | 0.1944 | 0.1400 | 0.3018 |
| lr | full | sex_age | four_fifths_pass | 0.0000 |  |  |
| tree | full | sex_label | dp_difference | 0.0616 | 0.0387 | 0.0840 |
| tree | full | sex_label | dp_ratio | 0.9206 | 0.8934 | 0.9496 |
| tree | full | sex_label | eo_difference | 0.0504 | 0.0321 | 0.1028 |
| tree | full | sex_label | four_fifths_pass | 1.0000 |  |  |
| tree | full | age_group | dp_difference | 0.0725 | 0.0413 | 0.1040 |
| tree | full | age_group | dp_ratio | 0.9069 | 0.8681 | 0.9466 |
| tree | full | age_group | eo_difference | 0.1144 | 0.0572 | 0.1886 |
| tree | full | age_group | four_fifths_pass | 1.0000 |  |  |
| tree | full | sex_age | dp_difference | 0.1202 | 0.0945 | 0.1659 |
| tree | full | sex_age | dp_ratio | 0.8521 | 0.7965 | 0.8816 |
| tree | full | sex_age | eo_difference | 0.1683 | 0.1129 | 0.2747 |
| tree | full | sex_age | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex | sex_label | dp_difference | 0.0222 | 0.0035 | 0.0455 |
| lr | no_sex | sex_label | dp_ratio | 0.9710 | 0.9413 | 0.9953 |
| lr | no_sex | sex_label | eo_difference | 0.0260 | 0.0040 | 0.0798 |
| lr | no_sex | sex_label | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex | age_group | dp_difference | 0.1139 | 0.0862 | 0.1502 |
| lr | no_sex | age_group | dp_ratio | 0.8538 | 0.8087 | 0.8892 |
| lr | no_sex | age_group | eo_difference | 0.1392 | 0.0960 | 0.2146 |
| lr | no_sex | age_group | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex | sex_age | dp_difference | 0.1297 | 0.1070 | 0.1845 |
| lr | no_sex | sex_age | dp_ratio | 0.8364 | 0.7724 | 0.8641 |
| lr | no_sex | sex_age | eo_difference | 0.1701 | 0.1251 | 0.2785 |
| lr | no_sex | sex_age | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex | sex_label | dp_difference | 0.0388 | 0.0161 | 0.0601 |
| tree | no_sex | sex_label | dp_ratio | 0.9498 | 0.9230 | 0.9789 |
| tree | no_sex | sex_label | eo_difference | 0.0304 | 0.0094 | 0.0838 |
| tree | no_sex | sex_label | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex | age_group | dp_difference | 0.0736 | 0.0416 | 0.1047 |
| tree | no_sex | age_group | dp_ratio | 0.9060 | 0.8666 | 0.9461 |
| tree | no_sex | age_group | eo_difference | 0.1264 | 0.0597 | 0.1964 |
| tree | no_sex | age_group | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex | sex_age | dp_difference | 0.0995 | 0.0766 | 0.1506 |
| tree | no_sex | sex_age | dp_ratio | 0.8770 | 0.8167 | 0.9040 |
| tree | no_sex | sex_age | eo_difference | 0.1568 | 0.1012 | 0.2603 |
| tree | no_sex | sex_age | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex_no_age | sex_label | dp_difference | 0.0150 | 0.0009 | 0.0361 |
| lr | no_sex_no_age | sex_label | dp_ratio | 0.9811 | 0.9550 | 0.9989 |
| lr | no_sex_no_age | sex_label | eo_difference | 0.0111 | 0.0038 | 0.0659 |
| lr | no_sex_no_age | sex_label | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex_no_age | age_group | dp_difference | 0.0402 | 0.0144 | 0.0708 |
| lr | no_sex_no_age | age_group | dp_ratio | 0.9498 | 0.9120 | 0.9816 |
| lr | no_sex_no_age | age_group | eo_difference | 0.0888 | 0.0311 | 0.1627 |
| lr | no_sex_no_age | age_group | four_fifths_pass | 1.0000 |  |  |
| lr | no_sex_no_age | sex_age | dp_difference | 0.0562 | 0.0356 | 0.1021 |
| lr | no_sex_no_age | sex_age | dp_ratio | 0.9309 | 0.8758 | 0.9554 |
| lr | no_sex_no_age | sex_age | eo_difference | 0.1029 | 0.0696 | 0.2266 |
| lr | no_sex_no_age | sex_age | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex_no_age | sex_label | dp_difference | 0.0331 | 0.0098 | 0.0558 |
| tree | no_sex_no_age | sex_label | dp_ratio | 0.9549 | 0.9251 | 0.9866 |
| tree | no_sex_no_age | sex_label | eo_difference | 0.0223 | 0.0085 | 0.0596 |
| tree | no_sex_no_age | sex_label | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex_no_age | age_group | dp_difference | 0.0427 | 0.0209 | 0.0770 |
| tree | no_sex_no_age | age_group | dp_ratio | 0.9424 | 0.8976 | 0.9711 |
| tree | no_sex_no_age | age_group | eo_difference | 0.0882 | 0.0432 | 0.1600 |
| tree | no_sex_no_age | age_group | four_fifths_pass | 1.0000 |  |  |
| tree | no_sex_no_age | sex_age | dp_difference | 0.0803 | 0.0530 | 0.1252 |
| tree | no_sex_no_age | sex_age | dp_ratio | 0.8945 | 0.8372 | 0.9297 |
| tree | no_sex_no_age | sex_age | eo_difference | 0.1239 | 0.0850 | 0.2300 |
| tree | no_sex_no_age | sex_age | four_fifths_pass | 1.0000 |  |  |


### Reweighing

Reweighing is estimated over 5 seeds. For logistic regression on all features, reweighing on sex moves ROC-AUC from 0.7288 ± 0.0087 to 0.7261 ± 0.0086 (lower) and moves equalized odds difference from 0.0697 ± 0.0222 to 0.0214 ± 0.0123 (lower). Demographic parity difference moves from 0.0726 ± 0.0136 to 0.0068 ± 0.0046 (lower). On the no-sex feature set, ROC-AUC moves from 0.7276 ± 0.0086 to 0.7276 ± 0.0086 and equalized odds difference moves from 0.0264 ± 0.0147 to 0.0253 ± 0.0136.

| model | feature_set | mitigation | protected_attribute | roc_auc_mean | roc_auc_std | eo_mean | eo_std | dp_mean | dp_std | f1_mean | f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | full | baseline | age_young | 0.7288 | 0.0087 | 0.0245 | 0.0047 | 0.0120 | 0.0054 | 0.5200 | 0.0152 |
| lr | full | baseline | intersection | 0.7288 | 0.0087 | 0.0911 | 0.0230 | 0.0863 | 0.0115 | 0.5200 | 0.0152 |
| lr | full | baseline | sex | 0.7288 | 0.0087 | 0.0697 | 0.0222 | 0.0726 | 0.0136 | 0.5200 | 0.0152 |
| lr | full | reweighed | age_young | 0.7282 | 0.0086 | 0.0190 | 0.0128 | 0.0104 | 0.0114 | 0.5195 | 0.0129 |
| lr | full | reweighed | intersection | 0.7254 | 0.0084 | 0.0535 | 0.0128 | 0.0210 | 0.0097 | 0.5143 | 0.0133 |
| lr | full | reweighed | sex | 0.7261 | 0.0086 | 0.0214 | 0.0123 | 0.0068 | 0.0046 | 0.5154 | 0.0154 |
| lr | no_sex | baseline | age_young | 0.7276 | 0.0086 | 0.0262 | 0.0061 | 0.0131 | 0.0088 | 0.5156 | 0.0163 |
| lr | no_sex | baseline | intersection | 0.7276 | 0.0086 | 0.0584 | 0.0216 | 0.0411 | 0.0133 | 0.5156 | 0.0163 |
| lr | no_sex | baseline | sex | 0.7276 | 0.0086 | 0.0264 | 0.0147 | 0.0273 | 0.0066 | 0.5156 | 0.0163 |
| lr | no_sex | reweighed | age_young | 0.7270 | 0.0085 | 0.0251 | 0.0113 | 0.0146 | 0.0108 | 0.5172 | 0.0136 |
| lr | no_sex | reweighed | intersection | 0.7270 | 0.0085 | 0.0512 | 0.0208 | 0.0407 | 0.0121 | 0.5164 | 0.0134 |
| lr | no_sex | reweighed | sex | 0.7276 | 0.0086 | 0.0253 | 0.0136 | 0.0277 | 0.0065 | 0.5173 | 0.0155 |
| tree | full | baseline | age_young | 0.7843 | 0.0088 | 0.0330 | 0.0176 | 0.0154 | 0.0088 | 0.5504 | 0.0149 |
| tree | full | baseline | intersection | 0.7843 | 0.0088 | 0.0929 | 0.0240 | 0.0759 | 0.0111 | 0.5504 | 0.0149 |
| tree | full | baseline | sex | 0.7843 | 0.0088 | 0.0544 | 0.0180 | 0.0566 | 0.0074 | 0.5504 | 0.0149 |
| tree | full | reweighed | age_young | 0.7841 | 0.0088 | 0.0209 | 0.0075 | 0.0052 | 0.0039 | 0.5487 | 0.0124 |
| tree | full | reweighed | intersection | 0.7834 | 0.0091 | 0.0637 | 0.0121 | 0.0367 | 0.0145 | 0.5495 | 0.0123 |
| tree | full | reweighed | sex | 0.7830 | 0.0090 | 0.0249 | 0.0163 | 0.0245 | 0.0100 | 0.5511 | 0.0130 |
| tree | no_sex | baseline | age_young | 0.7837 | 0.0095 | 0.0259 | 0.0154 | 0.0124 | 0.0065 | 0.5497 | 0.0095 |
| tree | no_sex | baseline | intersection | 0.7837 | 0.0095 | 0.0696 | 0.0177 | 0.0572 | 0.0136 | 0.5497 | 0.0095 |
| tree | no_sex | baseline | sex | 0.7837 | 0.0095 | 0.0372 | 0.0183 | 0.0406 | 0.0077 | 0.5497 | 0.0095 |
| tree | no_sex | reweighed | age_young | 0.7831 | 0.0091 | 0.0242 | 0.0050 | 0.0062 | 0.0042 | 0.5494 | 0.0107 |
| tree | no_sex | reweighed | intersection | 0.7829 | 0.0088 | 0.0721 | 0.0107 | 0.0466 | 0.0124 | 0.5449 | 0.0098 |
| tree | no_sex | reweighed | sex | 0.7831 | 0.0089 | 0.0307 | 0.0239 | 0.0347 | 0.0110 | 0.5492 | 0.0126 |


### Explanations

On the full logistic model the largest mean absolute SHAP feature is PAY_0 (0.4520). SEX has rank 13. On the no-sex model the largest is PAY_0. Spearman correlation of mean absolute SHAP ranks is 0.9862 between the baseline and the sex-reweighed model, and 0.9977 between the full and no-sex models on their shared features.

Individual reason lists:

denied_female (row 6753): September 2005 repayment status = 8 (payment delay of 8 months) increased the default score by +4.532 SHAP units (log-odds of default). August 2005 repayment status = 7 (payment delay of 7 months) increased the default score by +0.701 SHAP units (log-odds of default). July 2005 repayment status = 6 (payment delay of 6 months) increased the default score by +0.406 SHAP units (log-odds of default). September 2005 bill amount (NT dollar) = 16942 increased the default score by +0.267 SHAP units (log-odds of default). June 2005 repayment status = 5 (payment delay of 5 months) increased the default score by +0.195 SHAP units (log-odds of default). male_neighbor_approved (row 23986): July 2005 repayment status = 4 (payment delay of 4 months) increased the default score by +0.276 SHAP units (log-odds of default). September 2005 bill amount (NT dollar) = 29206 increased the default score by +0.191 SHAP units (log-odds of default). June 2005 repayment status = 3 (payment delay of 3 months) increased the default score by +0.121 SHAP units (log-odds of default). credit limit (NT dollar) = 50000 increased the default score by +0.086 SHAP units (log-odds of default). recorded sex code (1 male, 2 female) = 1 (male) increased the default score by +0.070 SHAP units (log-odds of default). borderline (row 16444): September 2005 bill amount (NT dollar) = 25627 increased the default score by +0.213 SHAP units (log-odds of default). marriage code = 1 (married) increased the default score by +0.111 SHAP units (log-odds of default). credit limit (NT dollar) = 30000 increased the default score by +0.101 SHAP units (log-odds of default). age in years = 53 increased the default score by +0.081 SHAP units (log-odds of default). June 2005 bill amount (NT dollar) = 28827 increased the default score by +0.062 SHAP units (log-odds of default). clear_approval (row 14513): August 2005 bill amount (NT dollar) = 261444 increased the default score by +0.572 SHAP units (log-odds of default). July 2005 bill amount (NT dollar) = 309622 increased the default score by +0.546 SHAP units (log-odds of default). May 2005 bill amount (NT dollar) = 258056 increased the default score by +0.482 SHAP units (log-odds of default). April 2005 bill amount (NT dollar) = 188083 increased the default score by +0.213 SHAP units (log-odds of default). marriage code = 1 (married) increased the default score by +0.111 SHAP units (log-odds of default).

### Differential privacy and the recommended setting

The plotted privacy curve is logistic regression on no_sex, reweighing=False, with finite epsilon values 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10. The paired non-private model has ROC-AUC 0.7303 ± 0.0084 and sex equalized odds difference 0.0208 ± 0.0116. At epsilon 0.01, ROC-AUC is 0.5711 ± 0.0579, sex equalized odds difference is 0.0413 ± 0.0371, and sex demographic parity difference is 0.0384. At epsilon 10, ROC-AUC is 0.7285 ± 0.0109 and sex equalized odds difference is 0.0273. Relative to the non-private model, the change in group ROC-AUC at epsilon 0.01 is female 0.1549 (from 0.7230 to 0.5681); male 0.1656 (from 0.7380 to 0.5724). The smaller sex group in the modeling frame is male. SHAP Spearman correlation with the non-private model is 0.1684 at epsilon 0.01 (top-k Jaccard 0.3381) and 0.7606 at epsilon 10 (top-k Jaccard 0.6457). The recommendation rule selected epsilon 10 on full with reweighing=False: ROC-AUC 0.7290, sex equalized odds difference 0.0605. Smallest epsilon whose mean ROC-AUC is within tolerance of its paired non-private model and whose mean sex equalized-odds difference is no worse.

| epsilon | feature_set | reweighed | roc_auc_mean | roc_auc_std | eo_difference_sex_mean | eo_difference_sex_std | dp_difference_sex_mean | shap_spearman_mean | shap_jaccard_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0100 | full | False | 0.5834 | 0.0590 | 0.1243 | 0.1802 | 0.1149 | 0.2960 | 0.3767 |
| 0.0100 | full | True | 0.5833 | 0.0590 | 0.1284 | 0.1862 | 0.1194 | 0.1508 | 0.3507 |
| 0.0100 | no_sex | False | 0.5711 | 0.0579 | 0.0413 | 0.0371 | 0.0384 | 0.1684 | 0.3381 |
| 0.0100 | no_sex | True | 0.5713 | 0.0582 | 0.0419 | 0.0367 | 0.0386 | 0.0636 | 0.3905 |
| 0.0500 | full | False | 0.6056 | 0.0509 | 0.2162 | 0.1910 | 0.2024 | 0.3209 | 0.3767 |
| 0.0500 | full | True | 0.6047 | 0.0513 | 0.2312 | 0.1762 | 0.2172 | 0.1621 | 0.3434 |
| 0.0500 | no_sex | False | 0.5868 | 0.0406 | 0.0336 | 0.0288 | 0.0290 | 0.1386 | 0.3548 |
| 0.0500 | no_sex | True | 0.5866 | 0.0404 | 0.0337 | 0.0278 | 0.0284 | 0.0344 | 0.3024 |
| 0.1000 | full | False | 0.6164 | 0.0464 | 0.2044 | 0.1649 | 0.1922 | 0.3186 | 0.3767 |
| 0.1000 | full | True | 0.6152 | 0.0468 | 0.2046 | 0.1635 | 0.1937 | 0.1779 | 0.3601 |
| 0.1000 | no_sex | False | 0.5958 | 0.0302 | 0.0347 | 0.0269 | 0.0310 | 0.1431 | 0.3190 |
| 0.1000 | no_sex | True | 0.5954 | 0.0307 | 0.0266 | 0.0205 | 0.0215 | 0.0620 | 0.3190 |
| 0.5000 | full | False | 0.6525 | 0.0263 | 0.0885 | 0.0669 | 0.0756 | 0.3049 | 0.3767 |
| 0.5000 | full | True | 0.6497 | 0.0266 | 0.1031 | 0.0644 | 0.0879 | 0.2221 | 0.4267 |
| 0.5000 | no_sex | False | 0.6458 | 0.0068 | 0.0366 | 0.0186 | 0.0303 | 0.1867 | 0.3524 |
| 0.5000 | no_sex | True | 0.6449 | 0.0086 | 0.0400 | 0.0235 | 0.0277 | 0.0925 | 0.3601 |
| 1.0000 | full | False | 0.6770 | 0.0196 | 0.0590 | 0.0505 | 0.0504 | 0.3824 | 0.4791 |
| 1.0000 | full | True | 0.6737 | 0.0197 | 0.0709 | 0.0382 | 0.0574 | 0.3020 | 0.4725 |
| 1.0000 | no_sex | False | 0.6693 | 0.0079 | 0.0449 | 0.0192 | 0.0388 | 0.3723 | 0.4344 |
| 1.0000 | no_sex | True | 0.6686 | 0.0091 | 0.0432 | 0.0174 | 0.0399 | 0.2538 | 0.4125 |
| 2.0000 | full | False | 0.7049 | 0.0103 | 0.0596 | 0.0460 | 0.0607 | 0.4417 | 0.4564 |
| 2.0000 | full | True | 0.7014 | 0.0104 | 0.0563 | 0.0310 | 0.0350 | 0.4121 | 0.5165 |
| 2.0000 | no_sex | False | 0.6974 | 0.0091 | 0.0423 | 0.0090 | 0.0446 | 0.4853 | 0.4755 |
| 2.0000 | no_sex | True | 0.6969 | 0.0094 | 0.0429 | 0.0122 | 0.0394 | 0.3653 | 0.4945 |
| 5.0000 | full | False | 0.7222 | 0.0082 | 0.0643 | 0.0264 | 0.0613 | 0.5478 | 0.5724 |
| 5.0000 | full | True | 0.7184 | 0.0075 | 0.0432 | 0.0221 | 0.0210 | 0.4573 | 0.5421 |
| 5.0000 | no_sex | False | 0.7206 | 0.0124 | 0.0354 | 0.0117 | 0.0387 | 0.5756 | 0.5165 |
| 5.0000 | no_sex | True | 0.7197 | 0.0126 | 0.0322 | 0.0140 | 0.0338 | 0.5316 | 0.5641 |
| 10.0000 | full | False | 0.7290 | 0.0099 | 0.0605 | 0.0277 | 0.0621 | 0.7389 | 0.5981 |
| 10.0000 | full | True | 0.7256 | 0.0094 | 0.0338 | 0.0185 | 0.0127 | 0.6743 | 0.6200 |
| 10.0000 | no_sex | False | 0.7285 | 0.0109 | 0.0273 | 0.0140 | 0.0323 | 0.7606 | 0.6457 |
| 10.0000 | no_sex | True | 0.7280 | 0.0109 | 0.0241 | 0.0179 | 0.0272 | 0.7348 | 0.6970 |
| inf | full | False | 0.7315 | 0.0085 | 0.0618 | 0.0194 | 0.0628 | 1.0000 | 1.0000 |
| inf | full | True | 0.7288 | 0.0079 | 0.0310 | 0.0105 | 0.0123 | 1.0000 | 1.0000 |
| inf | no_sex | False | 0.7303 | 0.0084 | 0.0208 | 0.0116 | 0.0268 | 1.0000 | 1.0000 |
| inf | no_sex | True | 0.7301 | 0.0085 | 0.0253 | 0.0204 | 0.0253 | 1.0000 | 1.0000 |


## Ethical considerations

Credit decisions affect access to payments and liquidity. A default model is only a proxy for the Apple Card setting, which was about credit limits offered to applicants, not about observed default in Taiwan in 2005. Sex is recorded as a binary administrative code and does not represent gender identity. Fairness metrics can conflict: equalizing one conditional error rate can move another (Chouldechova 2017; Kleinberg, Mullainathan, and Raghavan 2017). Demographic parity and equalized odds are both reported so that conflict stays visible. Differential privacy can change group accuracy; the per-group AUC curves are the measurement of that possibility, not an assumption about its direction.

The favorable outcome used for parity is being predicted not to default. That is a research definition of approval. It is not a statement that a low default score should be treated as a credit offer.

## Caveats and recommendations

- Single institution, one country (Taiwan), calendar period 2005.
- Sex is recorded as binary.
- The dataset records default outcomes, not credit-limit decisions. It is a proxy setting for the questions in the Apple Card investigation, not a reconstruction of that portfolio.
- Fairness metrics can conflict. Improving demographic parity does not imply improving equalized odds, and the reverse is also true.
- diffprivlib scaler-bound caveat: StandardScaler means and variances are fit on the private training split before row clipping. Those statistics are not differentially private. After scaling, each coordinate is clipped to the configured feature_clip and each row is clipped so its L2 norm is at most data_norm. data_norm is a fixed configuration value, not a statistic of the sample. The epsilon reported for a diffprivlib model is the epsilon of that model's objective perturbation only. It does not account for the scaler fit. Inside this sweep, reweighing is done by resampling with replacement according to the Kamiran-Calders weights, including for the non-private arm, because diffprivlib ignores sample_weight. Phase 4 is the sample_weight result.
- Do not use the recommended epsilon, the threshold, or the reweighing weights outside this study. If the same measurement pipeline is repeated on a new window, the monitoring margins in the governance mapping are the review triggers, not an authorization to deploy.

Config hash: `6249e877ea37c1095dc589ffcd10d8bfc50e19eaaa93ec3bc06d5c40978cd152`.
