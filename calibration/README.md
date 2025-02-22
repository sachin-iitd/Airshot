## AirShot: Pollution Sensor Recommendation

Run as

``python -m src.main``

The important parameters are under args.debug flag in main.py

### Prerequisites

pip install numpy pandas torch scikit-learn 

Create the folders _**out**_ and **_models_** in the base directory.

## Kolkata Dataset

The calibrated Kolkata Dataset is available at 

[Kolkata Pollution Website](http://cse.iitd.ac.in/pollutiondata/kolkata/details)

[Direct Link](https://github.com/sachin-iitd/KolkataDataset)

The processed init-train-validate-test splitted data is provided in _**data**_ folder.

### Sample Data

|    |    lat |  long |      time      |     pm      |
|---:|:------:|------:|---------------:|------------:|
|  1 |  22.46	| 88.37	| 3/28/2023 0:00 | 53.37005078 |
|  2 |  22.50 |	88.35 |	3/28/2023 0:00 | 49.05392615 |
|  3 |  22.52 | 88.34 | 3/28/2023 0:00 | 54.73606112 |
|  4 |  22.54 | 88.34 | 3/28/2023 0:00 | 51.97899615 |
|  5 |  22.54 | 88.35 | 3/28/2023 0:00 | 60.40570953 |


### Details of Kolkata (India), Delhi-NCR (India) and Hamilton, Ontario (Canada) datasets


|                    | Kolkata                                  | Delhi-NCR                                |               Canada |
|-------------------:|:-----------------------------------------|:-----------------------------------------|:---------------------|
| Total area         | 160 km<sup>2</sup>                       | 559 km<sup>2</sup>                       | 1138 km<sup>2</sup>  |
| Total samples      | 104,447                                  | 12,542,183                               | 46,080               |
| PM2.5 Samples      | 104,447                                  | 12,542,183                               | 12,154               |
| Pollutants covered | PM2.5                                    | PM1, PM2.5 and PM10                      | PM1, PM2.5, PM10, CO, NO, NO2, SO2, O3|
| Meteorological     | Temp, RH, Pressure, Wind Speed, Rainfall | Temp, RH, Pressure, Wind Speed, Rainfall | -                    |
| Sensor source      | Colleges / Bus                           | Public bus                               | Commercial van       |
| Monitoring days    | 142                                      | 91                                       | 114                  |

### Statistical comparison of PM2.5 values in Kolkata, Delhi-NCR and Canada datasets

|           | Kolkata |  Delhi    |  Canada |
|----------:|:--------|:----------|:--------|
| Mean      |  75.82  |   207.92  |  15.08  |
| Std-dev   |  35.49  |   114.36  |  12.87  |
| Missing%  |   0     |     0     |  73.62  |

## Benchmarking Algorithms

* RLselect: We designed a novel active learning based algorithm to utilize Reinforcement Learning (RL) in
optimal sensor selection.
* Mutual Information (MI) [Guestrin et al., 2005; Krause et al., 2008]: This algorithm considers mutual
information as a function of entropy between selected and remaining locations. It aims to recommend
locations which maximizes the overall mutual information. It uses covariance between any pair of
location to find entropy, which limits the usage of this algorithm over a single temporal dimension to
generate data series for each location.
* Lerner’s [Lerner et al., 2019]: This algorithm was originally devised for heterogeneous sensors and
recommends the central locations with more selection pool locations in the adjacent neighbourhood.
* Coverage [Agarwal et al., 2020]: This method selects the sensor locations such that maximum area
of the city is covered. The coverage can be maximized in spatial or spatiotemporal settings. As a
functional low-cost static sensor is expected to provide PM data continuously, we focus on the spatial
coverage in our experiments.
* MaxError [Patel, 2021]: This method utilizes an active-learning framework wherein the sensor
with the highest likelihood of minimizing the overall prediction error, measured through Root Mean
Squared Error (RMSE), is chosen iteratively.
* Centrality [Freeman, 1977]: The betweenness centrality of a node v is defined as the fraction of
shortest paths between all pairs of nodes in the graph that pass through v.
* PageRank [Brin and Page, 1998]: In PageRank, nodes that are linked to by other important nodes
receive higher scores, reflecting their centrality in the network. This concept has been widely adopted
beyond web page ranking and is applied in various fields to measure the importance of nodes in
different types of networks.
* Random: We randomly select location nodes from the selection pool. It represents the most naïve yet
a powerful selection policy.

### Benchmarking Performance

<img src="img/legends.png">
<img src="img/metrics.jpg">
