# machinelearning-housingprices-python

JupyterLab

In this model, checkpoint "housing.ckpt" is evaluated and processed.
The housing data evaluated is taken from the 1990 California census.
First, a correlation between median_housing_price and other variables is established, allowing us to determine which variable correlates most with median_housing_price.

The variables are per block, so from this data, we can determine that on average, there are approximately 34 bedrooms in a house.

![JupyterLab](train_data_histogram.png)

Since there are negative values present in the data, we perform data.dropna(inplace=True).

Looking at a sns.heatmap, we can better visualize correaltions between variabls.


![JupyterLab](train_data_heatmap.png)
