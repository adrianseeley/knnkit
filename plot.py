import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_knn_heatmaps(csv_file='log.csv', output_file='plot.png'):
    # Read the CSV
    df = pd.read_csv(csv_file, keep_default_na=False)
    
    # Ensure threshold and exponent are treated as numeric
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    df['exponent']  = pd.to_numeric(df['exponent'],  errors='coerce')

    # Determine global min/avg fitness
    fitness_min = df['fitness'].min()
    fitness_avg = df['fitness'].mean()
    fitness_05  = df['fitness'].quantile(0.05)
    
    # Extract unique values
    unique_k = sorted(df['k'].unique())
    # Build unique approach tuples: (normalizeFunction, aggregateFunction)
    df['approach'] = list(zip(df['normalizeFunction'], df['aggregateFunction']))
    unique_approaches = df['approach'].drop_duplicates().tolist()
    
    # Dimensions of the subplot grid
    nrows = len(unique_k)
    ncols = len(unique_approaches)
    
    # Each subplot ~5 inches square (=> ~500 px if DPI ~100)
    fig, axes = plt.subplots(nrows, 
                             ncols, 
                             figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)  # Keep axes 2D even if nrows/ncols=1

    for i, k_val in enumerate(unique_k):
        for j, (norm_func, agg_func) in enumerate(unique_approaches):
            # print progress
            print(f'Processing k={k_val}, {norm_func}, {agg_func}')
        
            ax = axes[i, j]
            # Subset data for this k and approach
            sub = df[(df['k'] == k_val) & 
                     (df['normalizeFunction'] == norm_func) & 
                     (df['aggregateFunction'] == agg_func)]
            
            # Pivot so exponent is rows, threshold is columns, fitness is the heat value
            pivoted = sub.pivot(index='exponent', columns='threshold', values='fitness')
            
            # Create heatmap
            # vmin=0, vmax=1 will map fitness=0 as the "best" color and 1 as "worst" color
            sns.heatmap(
                pivoted, 
                ax=ax, 
                cmap='viridis_r', 
                vmin=fitness_min, 
                vmax=fitness_05, 
                cbar=True
            )
            
            # Title for each subplot
            ax.set_title(f'k={k_val}\n{norm_func}, {agg_func}', fontsize=10)
            ax.set_xlabel('threshold')
            ax.set_ylabel('exponent')
    
    # Tight layout for better spacing
    plt.tight_layout()
    # Save figure to file
    plt.savefig(output_file, dpi=100)
    plt.close(fig)

if __name__ == '__main__':
    plot_knn_heatmaps('log.csv', 'plot.png')
