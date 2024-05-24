import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2. Create the "overweight" column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normalize cholesterol and glucose data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # 6. Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name="total")

    # 7. Create the catplot
    cat_plot = sns.catplot(x="variable", y="total", hue="value", col="cardio",
                           data=df_cat, kind="bar", height=5, aspect=1)

    # 8. Get the figure for the output
    fig = cat_plot.fig

    # 9. Save the figure
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15. Plot the correlation matrix using seaborn's heatmap
    sns.heatmap(corr.round(1), mask=mask, annot=True, fmt='.1f', cmap='coolwarm', ax=ax, square=True,
                cbar_kws={'shrink': .5}, linewidths=0.5)

    # 16. Save the figure
    fig.savefig('heatmap.png')
    return fig