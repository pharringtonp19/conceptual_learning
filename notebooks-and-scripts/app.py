import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import streamlit as st

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
plt.style.use('seaborn-v0_8-dark-palette')

from matplotlib import font_manager 
locations = './styles/Newsreader'
font_files = font_manager.findSystemFonts(fontpaths=locations)
print(locations)
print(font_files[0])
for f in font_files: 
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Newsreader"


from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mtick 

def format_date(x, pos=None):
    month_year = mdates.num2date(x).strftime('%b\n%Y')
    if  mdates.num2date(x).month == 1:
        return month_year
    else:
        return  mdates.num2date(x).strftime('%b')
    

def random_step_function(scale, key):
    # Domain [0, 1]
    x = jnp.linspace(0, 1, 1000)
    # Number of partitions
    num_partitions = scale
    # Create partition edges
    edges = jnp.linspace(0, 1, num_partitions + 1)
    # Generate random heights for each partition
    heights = random.uniform(key, shape=(num_partitions,))
    
    # Create the step function values
    y = jnp.zeros_like(x)
    for i in range(num_partitions):
        if i < num_partitions - 1:
            # Apply height to the range of the current partition, excluding the endpoint
            y = jnp.where((x >= edges[i]) & (x < edges[i+1]), heights[i], y)
        else:
            # For the last partition, include the endpoint
            y = jnp.where((x >= edges[i]) & (x <= edges[i+1]), heights[i], y)
        
    return x, y


def plot_step_function(scale):
    key = random.PRNGKey(0)
    x, y = random_step_function(scale, key)
    
    plt.figure(figsize=(10, 4))
    plt.step(x, y, where="post")
    plt.title(f"Random Step Function with {scale} Partitions")
    plt.xlabel("x")
    plt.ylabel("Height")
    plt.grid(True)
    plt.show()

def main():
    st.title("The Conditional Expectation Function (CEF)")

    st.write("Conditioning on additional variables can be understood as allowing for finer variation of the CEF")

    # Slider for adjusting the scale (number of partitions)
    scale = st.slider("Select the number of partitions", min_value=2, max_value=50, value=10)

    # Display the plot
    fig = plt.figure(dpi=300, tight_layout=True, figsize=(7, 4.5))
    ax = plt.axes(facecolor=(.95, .96, .97))
    ax.xaxis.set_tick_params(length=0, labeltop=False, labelbottom=True)
    for key in 'left', 'right', 'top':
        ax.spines[key].set_visible(False)
    ax.set_title('Conditional Expectation Function', size=16, loc='center', pad=20)
    ax.text(0., 1.02, s='Value', transform=ax.transAxes, size=14)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.grid(True, color='white', linewidth=2)
    key = random.PRNGKey(0)
    x, y = random_step_function(scale, key)
    ax.step(x, y, where="post")
    ax.set_xlabel("Feature Space", size=14)
    plt.ylim(0, 1)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
