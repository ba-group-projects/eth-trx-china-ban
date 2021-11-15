# SMM638 Network Analytics MTP
## How to start
1. install packages
```bash
pip install -r requirements.txt # install packages
```
2. run the code in the smm638.ipynb

## The structure of the code
The structure of the code is as follows:
- data
  - address_type.json
  - cleaned_data.csv
  - cleaned_data_after_ban.csv
  - cleaned_data_before_ban.csv
  - id_node.csv
  - modularity_score.csv
- dataCleaned
  - cleaned_data.csv
- docs
  - quick_start.md(doc for cooperation)
- utils
  - \_\_init\_\_.py
  - analyze.py(main analysis methods are encapsulated in this file)
  - identify_contract.py(function to identify contract address)
  - preprocess.py(function to preprocess data, since we already have the processed data in the data file, we do not need to apply functions in this file in our main code.)
- figure(We save all of our figure in this file)
  - Bipartite_Degree_Distribution.jpg
  - Monopartite_Degree_Distribution.jpg
  - bipartite_full_network_comparison.jpg
  - bipartite_subset_network_comparison.jpg
  - centrality_plot.png
  - modularity_scores.png
  - monopartite_full_network_comparison.jpg
  - monopartite_subset_network_comparison.jpg
  - plot_degree_dist.jpg
  - plot_power_law.png
- smm638.ipynb # main code
- requirements.txt
- README.md
- config.json(the config file we used to get the data)
