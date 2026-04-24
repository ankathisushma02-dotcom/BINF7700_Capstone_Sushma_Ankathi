import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"

harm_df = pd.read_csv(
    os.path.join(PROJECT_DIR, "drugcomb_harmonised.csv"))

top_drugs = (harm_df['DRUG1']
             .value_counts()
             .head(20).index.tolist())

heat_df = harm_df[
    harm_df['DRUG1'].isin(top_drugs) &
    harm_df['DRUG2'].isin(top_drugs)
]

pivot = (heat_df
         .groupby(['DRUG1', 'DRUG2'])['SYNERGY_LABEL']
         .mean()
         .unstack())

plt.figure(figsize=(14, 11))
sns.heatmap(pivot,
            cmap='seismic',
            center=0.5,
            xticklabels=True,
            yticklabels=True,
            annot=False,
            linewidths=0.3,
            cbar_kws={'label': 'Synergy Rate'})
plt.title('Drug Pair Synergy Heatmap\n'
          'Top 20 drugs — mean synergy rate',
          fontsize=13)
plt.xlabel('Drug 2', fontsize=11)
plt.ylabel('Drug 1', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         'synergy_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> synergy_heatmap.png")