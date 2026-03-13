import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Generate exploratory data analysis (EDA) visualizations for key variables
    related to customer churn.

    The function creates three plots:
        1. Customer service calls vs churn distribution.
        2. Daytime usage (minutes) vs churn comparison.
        3. International plan subscription vs churn distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing churn information and customer features.

    Returns
    -------
    None
        Displays the generated plots.
    """

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.countplot(
        data=df,
        x="Customer_service_calls",
        hue="Churn",
        palette="Set2"
    )
    plt.title("Churn vs Customer Service Calls")
    plt.xlabel("Number of Customer Service Calls")

    plt.subplot(1, 3, 2)
    sns.boxplot(
        data=df,
        x="Churn",
        y="Total_day_minutes",
        palette="Set1"
    )
    plt.title("Daytime Usage vs Churn")
    plt.ylabel("Total Day Minutes")

    plt.subplot(1, 3, 3)
    sns.countplot(
        data=df,
        x="International_plan",
        hue="Churn",
        palette="Set2"
    )
    plt.title("International Plan vs Churn")
    plt.xlabel("International Plan (0 = No, 1 = Yes)")

    plt.tight_layout()
    plt.show()