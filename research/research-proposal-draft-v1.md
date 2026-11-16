# Investigating the Combined Effects of Missing Values and Feature Noise on Traditional Machine Learning Performance in Tabular Data

**Document type:** Preliminary research proposal / working draft  
**Status:** Draft 1 — research design to be refined  
**Author:** Hunesh Kumar  
**Programme:** MPhil Data Science  
**Institution:** University of Sindh  

> **Draft note:** This document is a starting point for developing the research. The literature review, research gap, dataset choices, experimental settings, and references must be checked and refined before academic submission. No experimental results are claimed in this proposal.

---

## Abstract

The quality of data can influence how well machine learning models perform. In real-world tabular datasets, values may be missing, and recorded feature values may contain noise. These problems are often considered separately, but they can also occur together. When both are present, their combined effect on model performance may differ from the effect of either problem on its own.

This proposed study will investigate how missing values and feature noise affect selected traditional machine learning algorithms on tabular classification datasets. It will compare a clean-data baseline with conditions containing missing values only, feature noise only, and both types of data corruption. The study will also examine whether the choice of imputation method changes model performance when feature noise is present. Candidate imputation methods include mean, median, and K-nearest-neighbours (KNN) imputation. Candidate classifiers include Logistic Regression, Decision Tree, and Random Forest.

The experiments will use controlled data-corruption procedures and consistent evaluation settings. Classification performance will be assessed using measures such as accuracy, precision, recall, F1-score, and, where appropriate, ROC-AUC or balanced accuracy. Repeated experimental runs will be used to examine variability. The main focus will be the interaction between missingness and feature noise, rather than only their separate effects.

The expected outcome is a clearer understanding of how these two data-quality problems affect traditional machine learning models and whether some imputation methods are more suitable than others when noisy features are also present. The study may help practitioners make more informed preprocessing decisions when working with imperfect tabular data.

**Keywords:** data quality, missing values, feature noise, imputation, tabular data, machine learning, classification.

---

## 1. Introduction

Machine learning is widely used to identify patterns in data and make predictions. The quality of the data supplied to a model is an important part of this process. In practical applications, datasets are not always complete or perfectly measured. Some records may contain missing values, while other values may be inaccurate, inconsistent, or affected by measurement errors.

Missing values occur when information is not recorded or is unavailable for one or more variables. A common response is to remove incomplete records or replace missing values through an imputation method. However, removing records can reduce the amount of available data, while imputation can introduce assumptions or change the distribution of a feature. The effect of an imputation method may depend on the dataset and the machine learning algorithm being used.

Feature noise is another data-quality problem. In this study, feature noise refers to unwanted variation or perturbation in the input feature values. For example, a measurement may be slightly higher or lower than the true value because of a sensor limitation, recording error, or other source of measurement uncertainty. Noise in input features can make it more difficult for a model to learn useful patterns.

Missing values and feature noise may occur in the same dataset. A dataset could contain incomplete measurements as well as inaccurate recorded measurements. If these problems are handled independently, the resulting preprocessing decisions may not be suitable for data affected by both. For example, an imputation method may perform differently when the observed values used to estimate missing entries are themselves noisy.

This research therefore proposes a controlled investigation of missing values and feature noise, both separately and together. It will focus on traditional machine learning algorithms applied to tabular classification data. By comparing different corruption conditions and imputation methods, the study aims to understand whether the combined presence of missing values and feature noise has an effect beyond their individual effects.

## 2. Problem Statement

Many machine learning studies evaluate model performance using datasets that have been cleaned or prepared before training. However, real-world tabular datasets may contain more than one type of data-quality problem at the same time. Missing values and noisy feature measurements are two examples.

The effect of missing values can depend on the proportion and pattern of missingness, the method used to fill missing entries, and the learning algorithm. Similarly, the effect of feature noise can depend on the amount of noise, the affected features, and the model's sensitivity to changes in input values.

Although previous research has examined missing-data handling, noisy data, and broader data-quality issues, it remains useful to study their interaction under a clearly controlled experimental design. In particular, the effect of applying imputation when the observed feature values are also noisy needs to be examined across selected traditional classifiers.

The problem addressed by this study is therefore the limited practical understanding that may result when missing values and feature noise are evaluated only as separate data-quality conditions. The research will compare their individual and combined effects and examine whether different imputation methods lead to different outcomes under these conditions.

## 3. Research Aim

The aim of this study is to investigate the individual and combined effects of missing values and feature noise on the performance of selected traditional machine learning algorithms using tabular classification datasets.

## 4. Research Objectives

The study will pursue the following objectives:

1. To evaluate the effect of different levels of missing values on selected traditional machine learning classifiers.
2. To examine how different levels of feature noise affect classifier performance.
3. To compare model performance when missing values and feature noise occur separately and together.
4. To compare mean, median, and K-nearest-neighbours imputation under different data-quality conditions.
5. To examine whether the effect of combined data corruption differs across the selected machine learning algorithms.

## 5. Research Questions

The proposed study will address these questions:

1. How do different levels of missing values affect the performance of selected traditional machine learning algorithms?
2. How does feature noise affect the performance of these algorithms?
3. Does the combined presence of missing values and feature noise have a different effect from their individual effects?
4. How does the choice of imputation method influence model performance when feature noise is also present?
5. Do the selected algorithms respond differently to the same missingness and feature-noise conditions?

## 6. Research Hypotheses

The following hypotheses will guide the experimental investigation. They are provisional and will be refined as the literature review and experimental design develop.

- **H1:** Increasing the proportion of missing values will reduce the predictive performance of at least some selected classifiers.
- **H2:** Increasing feature-noise intensity will reduce the predictive performance of at least some selected classifiers.
- **H3:** The combined presence of missing values and feature noise will have an effect on predictive performance that is not fully explained by considering each factor separately.
- **H4:** The relative performance of mean, median, and KNN imputation will differ across data-quality conditions.
- **H5:** The effect of missing values and feature noise will vary across the selected machine learning algorithms.

These hypotheses do not assume that every model or dataset will show the same pattern. The experiments will test whether the effects are consistent and under which conditions they occur.

## 7. Preliminary Literature Review

### 7.1 Missing values and imputation

Missing values are a common problem in data analysis. Their causes can differ, and the mechanism behind missingness can influence how the data should be handled. Three commonly discussed mechanisms are:

- **Missing Completely at Random (MCAR):** the probability that a value is missing is unrelated to observed or unobserved data.
- **Missing at Random (MAR):** missingness may depend on other observed variables, but not on the missing value itself after accounting for the observed information.
- **Missing Not at Random (MNAR):** missingness depends on the value that is missing or on other unobserved information.

Imputation methods replace missing entries with estimated values. Simple methods include mean and median imputation. Mean imputation replaces a missing numerical value with the mean of the observed values for that feature. Median imputation uses the median and is generally less affected by extreme values. KNN imputation estimates a missing value using information from similar observations.

No single imputation method is guaranteed to be best for every dataset. The suitability of a method can depend on the missingness mechanism, feature distributions, relationships between variables, and the downstream learning algorithm.

### 7.2 Feature noise

Feature noise refers to unwanted variation in input variables. It may arise from measurement devices, data-entry mistakes, data integration, or other collection processes. Noise can alter the patterns available to a learning algorithm and may affect different algorithms in different ways.

For a controlled experiment, noise needs an explicit operational definition. One possible approach for numerical features is additive Gaussian noise:

\[
x'_{ij} = x_{ij} + \epsilon_{ij}
\]

where \(x_{ij}\) is the original value for observation \(i\) and feature \(j\), and \(\epsilon_{ij}\) is a random noise term. The noise scale can be defined relative to the standard deviation of each feature. This is only a proposed simulation method; the final method should be justified and reported precisely.

This study focuses on noise in input features. It does not refer to label noise, which occurs when the target labels are incorrect or unreliable.

### 7.3 Related research and the research gap

Prior research has examined the effects of missing values, imputation methods, and machine learning classifiers. For example, Garciarena and Santana (2017) investigated interactions among missing-data types, imputation methods, and supervised classifiers. Other work has reviewed the effect of imputation on machine learning performance and evaluated the impact of imputation quality on classifiers.

Research has also considered data perturbations and broader data-quality dimensions. These studies show that data quality is not a single issue and that the effect of corrupted data can depend on the learning task and algorithm.

The proposed study should not claim that no previous research has investigated missing values and noise together. Instead, its potential contribution is a focused and controlled comparison of missingness and input-feature noise as two experimental factors. It will compare clean data, missingness alone, feature noise alone, and their combination, while also examining whether the choice of imputation method changes the results.

The precise research gap must be confirmed through a more complete literature search. In particular, the final review should identify studies that have already tested missingness and feature-noise interactions on tabular classification data, and explain exactly how the proposed experimental design differs from them.

## 8. Proposed Methodology

### 8.1 Research design

This study will use a quantitative experimental design. Controlled corruption will be introduced into selected tabular datasets so that model performance can be compared across defined data-quality conditions.

The experiments will include four main conditions:

1. **Clean condition:** the dataset is used without artificially introduced missing values or feature noise.
2. **Missing-only condition:** missing values are introduced, but no additional feature noise is added.
3. **Noise-only condition:** feature noise is introduced, but no additional missing values are added.
4. **Combined condition:** both missing values and feature noise are introduced.

The clean condition will provide a baseline for comparison.

### 8.2 Dataset selection

The study will use publicly available tabular classification datasets. Candidate datasets should have a clearly defined target variable, sufficient observations, and numerical input features suitable for the proposed corruption and imputation methods.

The final dataset selection will consider:

- number of observations and input features;
- class distribution;
- proportion of numerical and categorical variables;
- existing missing values;
- suitability for classification;
- licence and availability for research use.

To keep the first experiment manageable, the initial study may focus on datasets with numerical input features. If mixed numerical and categorical datasets are included, categorical missing-value handling will need to be defined separately. Dataset names and sources will be added after they have been selected and checked.

### 8.3 Introducing missing values

Missing values will initially be introduced using a controlled MCAR procedure. Under this procedure, eligible feature cells will be selected randomly for masking, independently of their observed values.

Candidate missingness levels are 5%, 10%, 20%, and 40% of eligible input-feature cells. The final levels may be adjusted after pilot experiments. The percentage must be reported clearly as the proportion of eligible feature cells made missing, not the proportion of rows removed.

The initial experiment will focus on MCAR to keep the design controlled. MAR and MNAR may be considered as extensions if time and resources permit.

### 8.4 Introducing feature noise

For numerical input features, additive Gaussian noise is a possible controlled method. Noise will be scaled relative to the standard deviation of each feature so that the perturbation is comparable across features with different measurement scales.

One possible definition is:

\[
x'_{ij} = x_{ij} + \epsilon_{ij}, \qquad
\epsilon_{ij} \sim \mathcal{N}(0,(\alpha s_j)^2)
\]

where \(s_j\) is the standard deviation of feature \(j\), and \(\alpha\) controls the noise intensity.

The values of \(\alpha\) will be selected and justified during the pilot stage. The study will specify whether noise is added to training data, test data, or both. These settings should not be mixed within an experiment because training-time and test-time corruption represent different situations.

### 8.5 Imputation methods

The study will compare the following candidate methods for numerical features:

- **Mean imputation:** replaces a missing value with the mean of the observed training values for that feature.
- **Median imputation:** replaces a missing value with the median of the observed training values for that feature.
- **KNN imputation:** estimates missing values using similar observations.

The imputer must be fitted using the training data only and then applied to the relevant validation or test data. This helps prevent information from the test set from influencing preprocessing. Where KNN imputation is used, feature scaling must be handled appropriately within the training pipeline.

### 8.6 Machine learning algorithms

The initial set of classifiers will include:

- **Logistic Regression:** a linear classification model that provides a useful baseline.
- **Decision Tree:** a tree-based model that can learn non-linear decision rules.
- **Random Forest:** an ensemble of decision trees that may behave differently from a single tree under data corruption.

The final selection may be adjusted according to the datasets and available computational resources. Any additional algorithms should be selected for a clear methodological reason rather than simply increasing the number of experiments.

### 8.7 Experimental procedure

The proposed procedure is:

1. Select and inspect the datasets.
2. Separate input features from the target variable.
3. Create consistent training and test partitions, using stratification where appropriate.
4. Establish a clean-data baseline for each selected classifier.
5. Generate missing-only, noise-only, and combined corruption conditions.
6. Apply each imputation method within a leakage-safe preprocessing pipeline.
7. Train each classifier using the same data partitions and comparable settings.
8. Evaluate the models using the selected performance metrics.
9. Repeat corruption generation and/or model evaluation with multiple random seeds.
10. Compare the results across corruption levels, imputation methods, algorithms, and datasets.

The exact placement of corruption in the training and test pipeline will be fixed before the main experiments. The study will report this choice clearly.

### 8.8 Evaluation metrics

The following metrics are candidates for evaluating classification performance:

- **Accuracy:** the proportion of predictions that are correct.
- **Precision:** the proportion of predicted positive cases that are actually positive.
- **Recall:** the proportion of actual positive cases correctly identified.
- **F1-score:** the harmonic mean of precision and recall.
- **Balanced accuracy:** useful when class distributions are uneven.
- **ROC-AUC:** a threshold-independent measure of ranking performance for suitable classification settings.

The final metrics will depend on the class balance and characteristics of the selected datasets. Accuracy alone may be insufficient for imbalanced datasets.

### 8.9 Analysis of the combined effect

The central analysis will examine whether the effect of missingness changes when feature noise is present, and whether the effect of feature noise changes when missingness is present.

The analysis will compare performance across the four main conditions and across the selected levels of missingness and noise. Repeated runs will help estimate variability. Results may be reported using means, standard deviations, confidence intervals, and suitable statistical tests or regression models.

A factorial model with missingness and noise terms, including an interaction term, is one possible way to assess whether their combined effect differs from their separate effects. The final statistical approach will be chosen after the experimental structure and number of datasets are confirmed.

## 9. Expected Contributions

As this is a proposal, the following are expected contributions rather than established findings:

1. A controlled comparison of missing values and feature noise, both separately and in combination.
2. An evaluation of selected traditional classifiers under different data-quality conditions.
3. A comparison of mean, median, and KNN imputation when feature noise is also present.
4. Evidence that may help explain whether the combined effect of these data-quality problems differs from their individual effects.
5. A reproducible experimental workflow that can be used for further study of data quality in tabular machine learning.

## 10. Scope and Limitations

The initial study will focus on tabular classification datasets and selected traditional machine learning algorithms. The proposed missingness experiment will initially use MCAR, and the feature-noise experiment will focus on numerical input features.

Artificially generated corruption provides experimental control, but it may not represent every form of data-quality problem found in real applications. Additive Gaussian noise is only one possible noise model. The findings may also depend on the selected datasets, corruption levels, algorithms, and preprocessing choices.

The study will not initially investigate label noise, image data, text data, or deep learning models. These areas may be considered in future work if the initial research provides a suitable basis.

## 11. Proposed Work Plan

| Stage | Planned activity |
|---|---|
| 1 | Search for and review relevant literature |
| 2 | Refine the research gap, questions, and hypotheses |
| 3 | Select datasets and define corruption procedures |
| 4 | Implement preprocessing and baseline models |
| 5 | Run missing-only, noise-only, and combined experiments |
| 6 | Analyse results and conduct statistical comparisons |
| 7 | Write, revise, and validate the research paper |

The schedule will be updated according to the assignment deadline, supervisor feedback, and experiment requirements.

## 12. Conclusion

This proposal presents a plan to investigate how missing values and feature noise affect traditional machine learning performance on tabular classification data. Rather than studying each problem only in isolation, the research will compare clean data, missingness alone, feature noise alone, and the combination of both. It will also examine whether mean, median, and KNN imputation behave differently when feature noise is present.

The study is intended to provide a controlled and reproducible evaluation of these data-quality conditions. Before the experiments begin, the literature review and research gap will be strengthened, datasets and corruption levels will be finalised, and the experimental pipeline will be specified in detail. Any conclusions about model performance will be based on the results of those experiments rather than assumed in advance.

---

## References — working list to verify

> **Important:** This is a preliminary reference list, not a final verified bibliography. Before submission, check each paper's exact title, author list, publication year, journal or proceedings details, page numbers, and DOI using the publisher's website or a trusted scholarly index. Add only sources that you have read and actually use in the paper.

1. Garciarena, U., & Santana, R. (2017). *An extensive analysis of the interaction between missing data types, imputation methods, and supervised classifiers*. Expert Systems with Applications, 89, 52–65. https://doi.org/10.1016/j.eswa.2017.07.026

2. *Missing value imputation affects the performance of machine learning: A review and analysis of the literature (2010–2021).* (2021). Information in Medicine Unlocked. https://doi.org/10.1016/j.imu.2021.100799

3. Shadbahr, T., et al. (2023). *The impact of imputation quality on machine learning classifiers for datasets with missing values*. Communications Medicine. https://doi.org/10.1038/s43856-023-00356-z

4. *Imputation of data Missing Not At Random: Artificial generation and benchmark analysis.* (2024). Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2024.123654

5. *On the consistency of supervised learning with missing values.* (2024). Statistical Papers. https://doi.org/10.1007/s00362-024-01550-4

6. *The effects of data quality on machine learning performance on tabular data.* (2025). Information Systems. https://doi.org/10.1016/j.is.2025.102549

7. Khan, F. A., et al. (2025). *Still more shades of null: An evaluation suite for responsible missing value imputation*. Proceedings of the VLDB Endowment, 18(9), 2899–2913. https://doi.org/10.14778/3746405.3746416

---

## Research decisions to finalise before implementation

- [ ] Confirm the final research title with the supervisor.
- [ ] Complete and document the literature search.
- [ ] Verify the research gap against closely related studies.
- [ ] Select the public datasets and record their sources and licences.
- [ ] Decide whether the first experiment uses numerical features only.
- [ ] Define whether corruption is applied to training data, test data, or both.
- [ ] Finalise missingness percentages and the exact cell-selection procedure.
- [ ] Finalise the feature-noise model and noise intensities.
- [ ] Specify the train/test split, random seeds, and evaluation metrics.
- [ ] Decide the statistical method for testing the missingness-by-noise interaction.
- [ ] Verify every reference and format citations according to the required style.
- [ ] Revise the document using supervisor feedback and the institution's AI-use policy.
