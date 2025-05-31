import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Constants for column names
GRADE_COL = 'pd_decided_class'
CLIENT_ID_COL = 'consolidated_id'
AMOUNT_COL = 'granted_amt_EUR'
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data prep.csv")


# ========== Data Loading & Filtering ==========
def load_data(path: str) -> pd.DataFrame:
    """Load CSV data with header on row 2."""
    return pd.read_csv(path, header=0)


def filter_data_by_year(df: pd.DataFrame, year: str) -> pd.DataFrame:
    """Filter data for a given year, validation sample only, and non-defaulted clients."""
    return df[
        (df['cohort_obs_dt'] == year) &
        (df['Validation_sample'] == 1) &
        (df['default_flg'] == 0)
        ]


# ========== Metrics & Indices ==========
def calculate_metrics(df_filtered: pd.DataFrame):
    """Compute counts and exposures by class."""
    client_counts = df_filtered.groupby(GRADE_COL)[CLIENT_ID_COL].nunique().sort_index()
    exposure_sums = df_filtered.groupby(GRADE_COL)[AMOUNT_COL].sum().sort_index()
    total_clients = client_counts.sum()
    total_exposure = exposure_sums.sum()
    client_freq = client_counts / total_clients
    exposure_freq = exposure_sums / total_exposure
    return client_counts, exposure_sums, client_freq, exposure_freq


def herfindahl_index(cv: float, K: int) -> float:
    """Herfindahl index formula based on coefficient of variation."""
    return 1 + np.log((cv ** 2 + 1) / K) / np.log(K)


def analyze_concentration(client_freq: pd.Series, exposure_freq: pd.Series):
    """Compute Herfindahl indices for clients and exposures."""
    K = len(client_freq)
    cv_clients = np.sqrt(K * np.sum((client_freq - (1 / K)) ** 2))
    cv_exposures = np.sqrt(K * np.sum((exposure_freq - (1 / K)) ** 2))
    hi_clients = herfindahl_index(cv_clients, K)
    hi_exposures = herfindahl_index(cv_exposures, K)
    return hi_clients, hi_exposures


def interpret_concentration_explicit(hhi: float) -> str:
    """Return concentration level label for a given Herfindahl index."""
    if hhi < 0.15:
        return "Low (HHI < 0.15)"
    elif hhi <= 0.25:
        return "Moderate (0.15 ≤ HHI ≤ 0.25)"
    else:
        return "High (HHI > 0.25)"


def concentration_label(hi: float) -> str:
    """Short Bulgarian label for Herfindahl level."""
    if hi < 0.15:
        return "Low"
    elif 0.15 <= hi <= 0.25:
        return "Moderate"
    else:
        return "High"


def compute_hhi_by_class(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """Compute HHI and client counts by risk class."""
    grouped_df = df.groupby([class_col, 'consolidated_id'])[AMOUNT_COL].sum().reset_index()
    results = []
    for class_label, group in grouped_df.groupby(class_col):
        total = group[AMOUNT_COL].sum()
        shares = group[AMOUNT_COL] / total
        hhi = np.sum(shares ** 2)
        label = interpret_concentration_explicit(hhi)
        results.append([
            class_label,
            len(group),
            round(total, 2),
            round(hhi, 2),
            label
        ])
    return pd.DataFrame(results, columns=[
        f'{class_col}', 'Number of customers', 'Total exposure (EUR)', 'Herfindahl Index', 'Concentration level (limits)'
    ])


# ========== Risk Class Labeling & Descriptions ==========
def get_risk_group(cls: int):
    """Returns risk group and description based on PD class."""
    if cls in [1, 2]:
        return "Low risk", "Clients with excellent\ncredit profile"
    elif cls in [3, 4, 5]:
        return "Medium risk", "Financially\nstable clients"
    elif cls in [6, 7]:
        return "Moderate risk", "Clients with some\nfinancial instability"
    else:
        return "High risk", "Clients with a high\nprobability of delinquency"


def get_risk_group_table(cls: int):
    """Risk group/description (single-line for table use)."""
    return get_risk_group(cls)


def get_full_description(cls):
    """Multi-line label for plotting."""
    risk, desc = get_risk_group(cls)
    return f"Class {cls}\n{risk}\n{desc}"


# ========== Plots & Output Tables ==========
def print_interpretation(hi_clients: float, hi_exposures: float):
    """Console summary for concentration indices."""
    print("=" * 50)
    print(f"Herfindahl index (clients): {hi_clients:.4f} - {concentration_label(hi_clients).capitalize()} concentration")
    print(f"Herfindahl index (exposures): {hi_exposures:.4f} - {concentration_label(hi_exposures).capitalize()} concentration")
    print("-" * 50)
    if hi_exposures > hi_clients:
        print("⚠️ Exposures are more concentrated than clients.")
    else:
        print("✅ Concentration is similar between clients and exposures.")
    print("=" * 50)


def generate_detailed_table(client_counts, client_pct, exposure_sums, exposure_pct):
    """Print detailed summary table for PD classes."""
    table_data = []
    for cls in client_counts.index:
        risk_level, description = get_risk_group_table(cls)
        table_data.append([
            f"Class {cls}",
            risk_level,
            description,
            client_counts[cls],
            round(client_pct[cls], 2),
            f"{exposure_sums[cls]:,.2f}",
            round(exposure_pct[cls], 2)
        ])
    headers = ['Class', 'Risk Level', 'Description', 'Number of Clients',
               'Client Share (%)', 'Total Exposure (EUR)', 'Exposure Share (%)']
    print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center', numalign='center'))


def plot_hi_by_class(df: pd.DataFrame, class_col: str = 'pd_decided_class'):
    """
    Plots the Herfindahl Index by classes (exposures and clients) for the specified column.
    """
    grouped = df.groupby([class_col, 'consolidated_id'])['granted_amt_EUR'].sum().reset_index()

    hi_data = []
    for cls in sorted(df[class_col].dropna().unique()):
        subgroup = grouped[grouped[class_col] == cls]
        if subgroup.empty:
            continue

        # HI by exposures
        total_exp = subgroup['granted_amt_EUR'].sum()
        shares_exp = subgroup['granted_amt_EUR'] / total_exp
        hi_exp = np.sum(shares_exp ** 2)

        # HI by clients
        n_clients = subgroup['consolidated_id'].nunique()
        shares_clients = np.ones(n_clients) / n_clients
        hi_clients = np.sum(shares_clients ** 2)

        hi_data.append({
            'Class': int(cls),
            'HI (exposures)': hi_exp,
            'HI (clients)': hi_clients
        })

    hi_df = pd.DataFrame(hi_data).sort_values('Class')
    x = hi_df['Class'].astype(str).tolist()
    indices = np.arange(len(x))
    bar_width = 0.35

    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(indices - bar_width / 2, hi_df['HI (exposures)'], width=bar_width, label='HI (exposures)', color='blue')
    bars2 = plt.bar(indices + bar_width / 2, hi_df['HI (clients)'], width=bar_width, label='HI (clients)', color='green')

    for i, bar in enumerate(bars1):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}", ha='center', fontsize=8)
    for i, bar in enumerate(bars2):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}", ha='center', fontsize=8)

    plt.xticks(indices, x, rotation=90)
    plt.ylabel("HI value")
    plt.xlabel(class_col)
    plt.title(f"Herfindahl Index by classes ({class_col}) – clients and exposures")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def extended_credit_risk_analysis(df_filtered: pd.DataFrame):
    """Scatter plot: PD model % vs granted amount, colored by PD class."""
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_filtered,
        x='pd_model_pct',
        y='granted_amt_EUR',
        hue='pd_decided_class',
        palette='tab10',
        alpha=0.6
    )
    plt.title('Relationship between PD and granted amount (EUR)')
    plt.xlabel('PD (probability of default)')
    plt.ylabel('Granted amount (EUR)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='PD class', loc='upper right')
    plt.tight_layout()
    plt.show()


# ========== Main Analysis Pipeline ==========
def main():
    # --- Data loading and filtering for latest year (2021)
    df = load_data(DATA_PATH)
    df_filtered = filter_data_by_year(df, '12/31/2021').copy()

    client_counts, exposure_sums, client_freq, exposure_freq = calculate_metrics(df_filtered)
    hi_clients, hi_exposures = analyze_concentration(client_freq, exposure_freq)
    print_interpretation(hi_clients, hi_exposures)
    client_pct = client_freq * 100
    exposure_pct = exposure_freq * 100

    plot_hi_by_class(df_filtered, class_col='pd_decided_class')
    plot_hi_by_class(df_filtered, class_col='pd_model_class')

    generate_detailed_table(client_counts, client_pct, exposure_sums, exposure_pct)
    extended_credit_risk_analysis(df_filtered)

    # --- Yearly comparison (2021 vs 2022)
    df_2021 = filter_data_by_year(df, '12/31/2021')
    df_2022 = filter_data_by_year(df, '12/31/2022').copy()
    c_2021, e_2021, f_c_2021, f_e_2021 = calculate_metrics(df_2021)
    c_2022, e_2022, f_c_2022, f_e_2022 = calculate_metrics(df_2022)
    hi_c_2021, hi_e_2021 = analyze_concentration(f_c_2021, f_e_2021)
    hi_c_2022, hi_e_2022 = analyze_concentration(f_c_2022, f_e_2022)

    print("\n📊 Herfindahl Index Comparison:")
    print(f"Herfindahl Index (clients): 2021 = {hi_c_2021:.4f}, 2022 = {hi_c_2022:.4f}")
    print(f"Herfindahl Index (exposures): 2021 = {hi_e_2021:.4f}, 2022 = {hi_e_2022:.4f}")

    hi_data = {
        "2021": analyze_concentration(f_c_2021, f_e_2021),
        "2022": analyze_concentration(f_c_2022, f_e_2022)
    }
    plot_hi_over_years(hi_data)

    hhi_model_df = compute_hhi_by_class(df_filtered, 'pd_model_class')
    hhi_decided_df = compute_hhi_by_class(df_filtered, 'pd_decided_class')
    print("\n📋 Herfindahl Index by PD MODEL CLASS (based on exposures):\n")
    print(tabulate(hhi_model_df, headers='keys', tablefmt='grid', floatfmt=(".0f", ".0f", ",.2f", ".2f", "")))
    print("\n📋 Herfindahl Index by PD DECIDED CLASS (based on exposures):\n")
    print(tabulate(hhi_decided_df, headers='keys', tablefmt='grid', floatfmt=(".0f", ".0f", ",.2f", ".2f", "")))

    df_expo = df[['pd_decided_class', 'consolidated_id', 'granted_amt_EUR', 'RWA_amt_EUR']].copy()
    df_expo = df_expo[df_expo['pd_decided_class'].notna() & df_expo['granted_amt_EUR'].notna()]
    df_expo['RWA_amt_EUR'] = df_expo['RWA_amt_EUR'].fillna(0)
    results = df_expo.groupby('pd_decided_class').agg(
        Брой_клиенти=('consolidated_id', 'count'),
        Обща_експозиция=('granted_amt_EUR', 'sum'),
        Общо_RWA=('RWA_amt_EUR', 'sum')
    ).reset_index()

    results = results.rename(columns={
        'Обща_експозиция': 'Total_Exposure',
        'Общо_RWA': 'Total_RWA',
        'Брой_клиенти': 'Number_of_Clients',
        'pd_decided_class': 'PD_Class'
    })
    print(results.columns)  # За проверка, можеш да го махнеш след това

    results['Total_Exposure'] = (results['Total_Exposure'] / 1_000_000).round(2)
    results['Total_RWA'] = (results['Total_RWA'] / 1_000_000).round(2)
    results['RWA/Exposure (%)'] = ((results['Total_RWA'] / results['Total_Exposure']) * 100).round(2)
    print("\n📋 Analysis by PD DECIDED CLASS – count, exposure, RWA, and risk density (in mln. EUR):\n")
    print(tabulate(
        results,
        headers=['PD Class', 'Number of Clients', 'Total Exposure (mln. EUR)', 'Total RWA (mln. EUR)',
                 'RWA / Exposure (%)'],
        tablefmt='grid',
        floatfmt=(".0f", ".0f", ",.2f", ",.2f", ".2f")
    ))

    # --- Bar plot: Exposure and RWA by class
    plt.figure(figsize=(12, 7))
    x = results['PD_Class'].astype(str)
    bar_width = 0.35
    indices = np.arange(len(x))

    bars1 = plt.bar(indices - bar_width / 2, results['Total_Exposure'], width=bar_width,
                    label='Granted Amount (mln. EUR)')
    bars2 = plt.bar(indices + bar_width / 2, results['Total_RWA'], width=bar_width, label='RWA Amount (mln. EUR)')

    total_exposure = results['Total_Exposure'].sum()
    total_rwa = results['Total_RWA'].sum()

    for bar in bars1:
        percent = (bar.get_height() / total_exposure) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{percent:.1f}%", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        percent = (bar.get_height() / total_rwa) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{percent:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel("PD Decided Class")
    plt.ylabel("Amount (mln. EUR)")
    plt.title("Total Exposure and RWA by PD Decided Class")
    plt.xticks(indices, x)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Bar plot: Mean granted amount by class
    avg_expo = df_filtered.groupby(GRADE_COL)[AMOUNT_COL].mean().reset_index()
    avg_expo.columns = ['Class', 'Average Amount (EUR)']
    avg_expo['Description'] = avg_expo['Class'].apply(get_full_description)
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(
        data=avg_expo,
        x='Description',
        y='Average Amount (EUR)',
        hue='Description',
        palette='viridis',
        legend=False
    )

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:,.0f}", ha='center', va='bottom', fontsize=9)
    plt.title("Average Granted Amount by Class", fontsize=14)
    plt.ylabel("Average Amount (EUR)")
    plt.xlabel("Class by Probability of Default (PD)")
    plt.xticks(rotation=0, fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- Prepare table for Excel export
    table_data = []
    for cls in client_counts.index:
        risk_level, description = get_risk_group_table(cls)
        table_data.append([
            f"Class {cls}",
            risk_level,
            description,
            client_counts[cls],
            round(client_pct[cls], 2),
            exposure_sums[cls],
            round(exposure_pct[cls], 2)
        ])
    df_table_classes = pd.DataFrame(table_data, columns=[
        'Class', 'Risk Level', 'Description', 'Number of Clients',
        'Client Share (%)', 'Total Exposure (EUR)', 'Exposure Share (%)'
    ])

    # --- Export all tables to a single Excel file with multiple sheets ---
    # with pd.ExcelWriter("Credit_Risk_Tables.xlsx") as writer:
    #     df_table_classes.to_excel(writer, sheet_name="Таблица_класове", index=False)
    #     hhi_model_df.to_excel(writer, sheet_name="Herfindahl_MODEL_CLASS", index=False)
    #     hhi_model_df.to_excel(writer, sheet_name="Herfindahl_MODEL_CLASS", index=False)
    #     hhi_decided_df.to_excel(writer, sheet_name="Herfindahl_DECIDED_CLASS", index=False)
    #     results.to_excel(writer, sheet_name="PD_DECIDED_CLASS_RWA", index=False)
    #     avg_expo.to_excel(writer, sheet_name="Средна_сума_по_клас", index=False)


def plot_hi_over_years(hi_data: dict):
    """
    hi_data: речник с ключове "година" и стойности (hi_clients, hi_exposures)
    """
    years = list(hi_data.keys())
    hi_clients = [v[0] for v in hi_data.values()]
    hi_exposures = [v[1] for v in hi_data.values()]
    
    indices = np.arange(len(years))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(indices - bar_width / 2, hi_clients, width=bar_width, label='HI (clients)', color='green')
    plt.bar(indices + bar_width / 2, hi_exposures, width=bar_width, label='HI (exposures)', color='blue')

    for i, val in enumerate(hi_clients):
        plt.text(indices[i] - bar_width / 2, val + 0.005, f"{val:.2f}", ha='center', fontsize=8)
    for i, val in enumerate(hi_exposures):
        plt.text(indices[i] + bar_width / 2, val + 0.005, f"{val:.2f}", ha='center', fontsize=8)

    plt.xticks(indices, years)
    plt.ylim(0, 1)
    plt.ylabel("Aggregate Herfindahl Index")
    plt.title("Aggregate Concentration by Year")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
