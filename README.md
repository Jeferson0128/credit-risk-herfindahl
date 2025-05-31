💳 Credit Risk Herfindahl

**Portfolio concentration analysis by PD (Probability of Default) classes using the Herfindahl Index (HI).**  
Project for Sofia University & KBC Bank Bulgaria – statistical validation of credit risk models, data visualization, and business interpretation.

---

🗂️ Project Structure

### 📦 1. Data Loading & Filtering
**Modules:**  
- `load_data`  
- `filter_data_by_year`  
*Loads the provided CSV portfolio data and filters by reporting year, including only validation and non-defaulted clients for the analysis.*  

---

📊 2. Metrics & Indices
**Modules:**  
- `calculate_metrics`  
- `herfindahl_index`  
- `analyze_concentration`  
- `compute_hhi_by_class`  
- `interpret_concentration_explicit`  
- `concentration_label`  
*Calculates main metrics for the portfolio: unique clients and total exposures by class, their frequencies, and the Herfindahl indices (HI) for both clients and exposures. Provides both calculation and interpretation of concentration levels.*  

---

🏷️ 3. Risk Class Labeling & Descriptions
**Modules:**  
- `get_risk_group`  
- `get_risk_group_table`  
- `get_full_description`  
*Returns meaningful labels and descriptions for each PD class and risk group, to be used in visualizations and output tables.*  

---

📈 4. Plots & Output Tables
**Modules:**  
- `plot_hi_by_class`  
- `print_interpretation`  
- `generate_detailed_table`  
- `extended_credit_risk_analysis`  
- `plot_hi_over_years`  
*Visualizes:*  
- 📊 Herfindahl Index distribution by class and over years  
- 🔍 PD vs. exposure relationship  
- 🧾 Detailed summary tables for business interpretation  

---

🔄 5. Main Analysis Pipeline
**Module:**  
- `main`  
*Central pipeline that runs the full analysis step by step:*  
- 📥 Loads and filters data  
- 📏 Calculates indices and generates visualizations  
- 🗓️ Compares years (2021 vs 2022)  
- 🧾 Generates key tables and summaries  
- 📤 (Prepares Excel export, code included but commented for flexibility)*

---

🛠️ Technologies & Dependencies

- 🐍 Python 3.x
- 🐼 pandas
- 🔢 numpy
- 📊 matplotlib
- 🎨 seaborn
- 🗒️ tabulate

Install dependencies (if needed):
```bash
pip install pandas numpy matplotlib seaborn tabulate
```

🎯 Features
🧮 Calculates Herfindahl Index for both clients and exposures by PD class

📆 Year-over-year concentration analysis (2021 vs 2022)

📊 Visualizes PD class distributions, exposures, and risk

🧾 Generates business-oriented tables and summaries

📤 Supports export to Excel (multi-sheet, code provided)

📊 Example Visualizations
📊 Herfindahl Index by class (clients vs exposures)

💶 Average granted amount by PD class

📈 Concentration evolution over time

All plots are stored in the plots/ folder.

📄 License
📝 MIT License
Project developed for educational and analytical purposes at Sofia University & KBC Bank Bulgaria.

👥 Authors
Sergey Filipov & Team
Sofia University, KBC Bank Bulgaria
