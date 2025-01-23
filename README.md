# Data Analysis on Diamond Prices

This repository contains a Python-based analysis of a cleaned diamonds dataset. The project involves data manipulation, visualization, and statistical analysis to uncover insights into the properties of diamonds and their pricing factors.

## Project Structure

### File Descriptions:
- **`data_analysis_diamonds.py`**: The main script that performs data processing, analysis, and visualization of the diamonds dataset.
- **`diamonds (cleaned).csv`**: The dataset used for the analysis, containing cleaned and preprocessed data on diamonds' attributes and pricing.

---

## Key Features of the Analysis

### 1. **Data Cleaning and Manipulation**
- Converted string variables (e.g., dimensions) to numeric for accurate calculations.
- Created new variables:
  - **Volume**: Computed from `Length`, `Width`, and `Height`.
  - **LogPrice**: Natural log of the `Price`.
  - **Expensive**: Binary indicator for diamonds priced above the mean.
  - **Standardized Carat Weight**: Z-score of `Carat Weight`.
- Coded categorical variables like `Clarity` into numeric for regression analysis.

### 2. **Data Visualization**
- **Bar Chart**: Distribution of diamond shapes, highlighting the most common shape (Round).
- **Scatter Plot**: Relationship between `Carat Weight` and `Price` to explore trends and outliers.
- **Line Graph**: Comparison of `Depth %` and `Table %` across observations.
- **Box Plot**: Distribution and outliers for the `LogPrice` variable.

### 3. **Statistical Analysis**
- Computed summary statistics and identified missing values for key variables.
- Conducted linear regression analyses:
  - First model treated `Clarity` and `Color` as numeric.
  - Second model improved accuracy by treating these variables as categorical.

---

## Results and Insights

### Visualization Highlights
- **Round diamonds** are the most common, likely due to a combination of cost-effectiveness and superior cut ratings.
- Scatter plots suggest a non-linear relationship between carat weight and price, with other attributes also significantly influencing the price.
- Outliers in various graphs indicate the importance of additional factors beyond the observed features.

### Regression Analysis
- Initial regression underestimated the effect of categorical variables.
- Treating `Clarity` and `Color` as categorical provided more accurate insights into how these factors influence price.

---

## Requirements

To run this project, you need:
- Python 3.7 or above
- Required libraries: `pandas`, `numpy`, `matplotlib`, and `statsmodels`

### Installation:
Use the following command to install the required libraries:
```bash
pip install pandas numpy matplotlib statsmodels

