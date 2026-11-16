# Machine-Learning-and-Data-Quality-Research
Investigating the Impact of Missing Values and Noisy Data on Machine Learning Model Performance

## Research Workflow

This diagram illustrates the proposed experimental workflow.
The study compares clean data, missing values, feature noise,
and their combined effects on traditional machine learning models.


## Research Workflow

```mermaid
flowchart TD
    A["Public Tabular Datasets"] --> B["Data Exploration and Preparation"]
    B --> C["Train / Test Split"]

    C --> D["Experimental Conditions"]

    D --> E["Clean Data"]
    D --> F["Missing Values Only"]
    D --> G["Feature Noise Only"]
    D --> H["Missing Values + Feature Noise"]

    E --> I["Preprocessing Pipeline"]
    F --> J["Imputation"]
    G --> I
    H --> J
    J --> I

    I --> K["Train Machine Learning Models"]

    K --> L["Logistic Regression"]
    K --> M["Decision Tree"]
    K --> N["Random Forest"]

    L --> O["Model Evaluation"]
    M --> O
    N --> O

    O --> P["Compare Performance Metrics"]
    P --> Q["Analyse Missingness and Noise Interaction"]
    Q --> R["Interpret Findings and Document Results"]
```

![alt text](/mermaid-diagram.png)