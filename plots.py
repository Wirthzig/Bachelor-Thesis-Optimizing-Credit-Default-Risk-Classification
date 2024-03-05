import colorsys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches

# Customize colors
background_color = '#FFFFFF'
line_color = '#1CD75F'
dot_color = '#1CD75F'
label_color = "#808080"  # "#121212"
transparent = "#5F5F5F00"
grey_color = "#b3b3b3"#"#808080"

blue="#0000a7"#"#30EEE8"
yellow="#008176"#"#EEDA30"
green=blue#"#1CD75F"
red= "#c1272d"#EE304E"
pink=yellow#"#EA30EE"
orange= red#"#EE7B30"

colors_dict = {
    'Logistic Regression': orange,
    'LightGBM': pink,
    'Neural Network': green
}


# Create the transformation function
def transform_data(df):
    # Initiate an empty dataframe
    transformed_df = pd.DataFrame(columns=["Model", "Bayesian Optimization", "Random Search", "Original"])
    # Get unique models
    models = df['Model'].unique()
    # Loop over models to get the respective AUCs
    for model in models:
        model_data = df[df['Model'] == model]
        # First row corresponds to Bayesian Optimization and Original AUC
        bayesian_optimization_auc = model_data.iloc[0]['AUC']
        original_auc = model_data.iloc[0]['OG_AUC']
        # Second row corresponds to Random Search AUC
        random_search_auc = model_data.iloc[1]['AUC']
        # Append the AUCs to the new dataframe
        transformed_df = transformed_df.append({
            "Model": model,
            "Bayesian Optimization": bayesian_optimization_auc,
            "Random Search": random_search_auc,
            "Original": original_auc
        }, ignore_index=True)

    return transformed_df


def lighten_color(hex_color, blend_factor=0.5):
    # Remove the hash symbol if present
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    # Convert hex color to RGB
    rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
    # Convert RGB to HLS
    hls_color = colorsys.rgb_to_hls(*[x/255.0 for x in rgb_color])
    # Lighten color by blend factor
    hls_color_lightened = (hls_color[0], 1 - blend_factor * (1 - hls_color[1]), hls_color[2])
    # Convert back to RGB
    rgb_color_lightened = colorsys.hls_to_rgb(*hls_color_lightened)
    # Convert RGB color back to hex
    hex_color_lightened = '#%02x%02x%02x' % tuple(int(x*255) for x in rgb_color_lightened)

    return str(hex_color_lightened)


def plot_model_auc(df, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, color=label_color, linestyle='--', linewidth=0.5, alpha=0.2)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    if model in df.columns:
        model_values = df[model]
        if model in ["Logistic Regression", "LightGBM"]:
            model_values = model_values - 0.1
        ax.plot(model_values, linestyle='solid', linewidth=4, label='Bayesian Optimization', color=colors_dict[model], zorder=4)

    if "RANDOM" + model in df.columns:
        random_model_values = df["RANDOM" + model]
        if model in ["Logistic Regression", "LightGBM", "Neural Network"]:
            random_model_values = random_model_values - 0.1
        ax.plot(random_model_values, linestyle='dashed', label='Random Search', color=colors_dict[model], zorder=3)

    ax.set_xlabel('Iterations', color=label_color)
    ax.set_ylabel('AUC', color=label_color)
    ax.legend(loc='upper right', labelcolor=label_color)
    ax.set_ylim(0.5, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor(transparent)
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)
    plt.title(model, color=label_color)
    plt.savefig(os.path.join("../Data/Outputs/Process/", model + '_auc_plot.png'))
    print("Created Plot: ", model +  '_auc_plot.png')


def plot_model_performance(df, kaggle, model_name):
    # Filter the dataframes for the given model
    model_data_df = df[df['Model'] == model_name]
    model_data_kaggle = kaggle[kaggle['Model'] == model_name]
    # Extract the performance metrics for plotting
    metrics_df = model_data_df[['Original', 'Random Search', 'Bayesian Optimization']].values.flatten()
    metrics_kaggle = model_data_kaggle[['Original', 'Random Search', 'Bayesian Optimization']].values.flatten()
    # Create the plot
    fig, ax = plt.subplots()
    # Add a grid with custom color
    ax.grid(True, color=label_color, linestyle='--', linewidth=0.5, alpha=0.2)
    # Plot the bars
    labels = ['Original', 'Random Search', 'Bayesian Optimization']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    #rects1 = ax.bar(x - width/2, metrics_df, width, color=[lighten_color(colors_dict[model_name], 0.35), lighten_color(colors_dict[model_name], 0.75), colors_dict[model_name]], zorder=3, hatch=['/', '-',''], edgecolor="white")
    #rects2 = ax.bar(x + width/2, metrics_kaggle, width, color=[lighten_color(grey_color, 0.35), lighten_color(grey_color, 0.75), grey_color], zorder=3)
    rects1 = ax.bar(x - width/2, metrics_df, width, color=colors_dict[model_name], zorder=3, edgecolor="white")
    rects2 = ax.bar(x + width/2, metrics_kaggle, width, color=grey_color, zorder=3, edgecolor="white")
    # Customize the background
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    # Add labels and a title with custom colors
    ax.set_ylabel('AUC', color=label_color)
    ax.set_title(f'Model Performance: {model_name}', color=label_color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Customize tick colors
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)
    ax.set_ylim(0.5, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor(transparent)
    # Add legend
    ax.legend([rects1, rects2], ['Test on Data', 'Test on Kaggle'], loc="upper left", labelcolor = label_color)

    # Save the plot
    path = os.path.join('../Data/Outputs/Performances/', f'{model_name}.png')
    print(f'Created Plot: {model_name}.png')
    plt.savefig(path, dpi=900)


def plot_imbalance_by_model(df):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Prepare patches for the legend
    patches = [mpatches.Patch(color=color, label="Bayesian Optimization") for model, color in colors_dict.items()]
    patches.append(mpatches.Patch(color='grey', label='Random Search'))

    # Iterate over each model
    for i, model in enumerate(df['Model'].unique()):
        # Select rows for this model
        model_df = df[df['Model'] == model]

        # Determine color
        color = colors_dict[model]

        # Plot Bayesian bar
        ax.bar(i - 0.2, model_df[model_df['Search'] == 'Bayesian']['Imbalance'].values[0], width=0.4, color=color, edgecolor="white", zorder=3)

        # Plot Random bar
        ax.bar(i + 0.2, model_df[model_df['Search'] == 'Random']['Imbalance'].values[0], width=0.4, color=grey_color, edgecolor="white", zorder=3)

    # Configure axes
    ax.set_xticks(range(len(df['Model'].unique())))
    ax.set_xticklabels(df['Model'].unique())
    ax.set_ylabel('Imbalance', color=label_color)
    ax.set_title('Imbalance of Defaults in Synthetic Data', color=label_color)
    ax.set_ylim(0, 1)
    ax.legend(["","","","","Bayesian Optimization","Random Search"], ncol=3, labelcolor=label_color)

    # Customize the background
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    # Customize tick colors
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(transparent)
    ax.grid(True, color=label_color, linestyle='--', linewidth=0.5, alpha=0.2)

    path = os.path.join('../Data/Outputs/Imbalances/', "imbalances.png")
    print(f'Created Plot: imbalances.png')
    plt.savefig(path, dpi=900)

# Read the data
performances_original = pd.read_csv('../Data/Outputs/Final Data/performances.csv')
performances = transform_data(performances_original)
performances_original['Search'] = performances_original['Filename'].apply(lambda x: 'Random' if 'RANDOM' in x else 'Bayesian')
print(performances)

kaggle = pd.read_csv('../Data/Outputs/Final Data/kaggle.csv')
print(kaggle)
process = pd.read_csv("../Data/Outputs/Final Data/process.csv")
process.drop(columns='test', inplace=True)

for model in performances["Model"]:
    # Use the function to create and save a barplot for a specific model
    plot_model_performance(performances, kaggle, model)

for model in kaggle["Model"]:
    # Use the function to create and save a barplot for a specific model
    plot_model_auc(process, model)

plot_imbalance_by_model(performances_original)
