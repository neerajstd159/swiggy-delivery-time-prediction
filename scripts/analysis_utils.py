import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency, f_oneway, jarque_bera

def numerical_analysis(df: pd.DataFrame, column: str, cat_col=None, bins="auto") -> None:
    # create the figure
    fig = plt.figure(figsize=(15, 10))
    # generate the layout
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    # set the subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    # plot the kde plot
    sns.kdeplot(data=df, x=column, hue=cat_col, ax=ax1)
    # plot the boxplot
    sns.boxplot(data=df, x=column, hue=cat_col, ax=ax2)
    # plot the histogram
    sns.histplot(data=df, x=column, bins=bins, hue=cat_col, ax=ax3, kde=True)

    plt.tight_layout()
    plt.show()


def numerical_categorical_analysis(df: pd.DataFrame, cat_col: str, num_col: str) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    # plot the barplot
    sns.barplot(data=df, x=cat_col, y=num_col, ax=ax[0, 0])
    # plot the boxplot
    sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax[0, 1])
    # plot the violinplot
    sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax[1, 0])
    # plot the stripplot
    sns.stripplot(data=df, x=cat_col, y=num_col, ax=ax[1, 1])

    plt.tight_layout()
    plt.show()


def categorical_analysis(df: pd.DataFrame, cat_col: str) -> None:
    # display the value counts of the categories
    display(
        pd.DataFrame(
            {
                "count": df[cat_col].value_counts(),
                "percentage": (df[cat_col].value_counts(normalize=True) * 100).astype(str).add("%")
            }
        )
    )
    print("*" * 50)
    # get the unique categories
    unique_categories = df[cat_col].unique().tolist()
    number_of_categories = len(unique_categories)
    print(f"unique categories in column {cat_col} are : {unique_categories}")
    print("*" * 50)
    print(f"Number of unique categories in column {cat_col} is : {number_of_categories}")

    # plot the count plot
    sns.countplot(data=df, x=cat_col)
    plt.xticks(rotation=45)
    plt.show()


def multivariate_analysis(df: pd.DataFrame, num_col: str, cat_col1: str, cat_col2: str) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    # plot the barplot
    sns.barplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, ax=ax[0, 0])
    # plot the boxplot
    sns.boxplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, ax=ax[0, 1])
    # plot the violinplot
    sns.violinplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, ax=ax[1, 0])
    # plot the stripplot
    sns.stripplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, ax=ax[1, 1])

    plt.tight_layout()
    plt.show()


def chi2_test(df: pd.DataFrame, cat_col1: str, cat_col2: str) -> None:
    df = df.loc[:, [cat_col1, cat_col2]].dropna()
    # contingency table
    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    # chi2 test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi2 value is : {chi2}")
    print(f"p value is : {p}")
    print(f"Degree of freedom is : {dof}")
    print(f"Expected contingency table is : {expected}")
    print("*" * 50)
    if p < 0.05:
        print("Reject the null hypothesis. There is a significant association between {cat_col1} and {cat_col2}")
    else:
        print("Fail to reject the null hypothesis. There is no significant association between {cat_col1} and {cat_col2}")
    print("*" * 50)


def anova_test(df: pd.DataFrame, num_col: str, cat_col: str) -> None:
    df = df.loc[:, [num_col, cat_col]].dropna()
    # anova test
    cat_groups = df.groupby(cat_col)
    groups = [group[num_col].values for _, group in cat_groups]
    f, p = f_oneway(*groups)
    print(f"F value is : {f}")
    if p < 0.05:
        print(f"p value is : {p}. Reject the null hypothesis. There is a significant association between {num_col} and {cat_col}")
    else:
        print(f"p value is : {p}. Fail to reject the null hypothesis. There is no significant association between {num_col} and {cat_col}")
    print("*" * 50)


def test_for_normality(df: pd.DataFrame, col_name) -> None:
    df = df.loc[:, col_name]
    print("Jarque Bera Test for Normality")
    _, p_val = jarque_bera(df)
    print(p_val)
    if p_val <= 0.5:
        print(f"Reject the null hypothesis. The data is not normally distributed.")
    else:
        print(f"Fail to reject the null hypothesis. The data is normally distributed.",end="\n\n")