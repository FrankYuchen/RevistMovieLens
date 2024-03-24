# Codes for the paper “Our Model Achieves Excellent Performance on MovieLens: What Does it Mean?”
Our paper focuses on the **MovieLens** dataset, one of the benchmark datasets in the field of recommender systems.
Our study attempts to explore a different perspective: **to what extent do we understand the user-item interaction generation mechanisms of a dataset?**
## MovieLens1518 Dataset
To minimize the potential impact of different versions, we extract user interaction data from the last four years  (2015-2018) in the MovieLens-25M dataset. <br>
In the curated dataset, we only retain ratings from the users whose entire rating history falls within the 4-year period.<br>
In addition, we remove the less active users who have fewer than 35 interactions with MovieLens, from the collected data.<br>
The dataset contains about 4.2 million user-item interactions (see Table below).
| \#Users      | \#Items  | Avg \#Ratings per user | Avg \#Ratings per item| \#Interactions|
| :---        |    :----:   |     :----:    |   :----:    | :----:    |
|   24,812    | 36,378      | 170.3   | 116.2   | 4.2M   |


The internal recommendation considers the popularity score of movies in the past year. We follow the same time scale and set the evaluation time window as one year and conduct independent comparative experiments **year by year**, from 2015 to 2018. <br>

## Baseline and Evaluation Metric
We select seven widely used baselines from four categories: (1) memory-based methods: **MostPop** and **ItemKNN**; (2) latent factor method: **PureSVD**; (3) non-sampling deep learning method: **Multi-VAE**; and (4) sequence-aware deep learning method: **SASRec**, **TiSASRec**, **Caser**.<be> 

## Experiments
Our experiments are divided into two main parts. All of the experiments follow the **leave-last-one-out** scheme. 
### Impact of Interaction Context at Different Stages
We conduct an ablation experiment with the removal of the first $15$ ratings, randomly sampled $15$ ratings, and the last $15$ ratings of each user's training instances. For the experiments that randomly remove $15$ ratings, we repeat the experiments three times with different seeds and get the average recommendation performance to reduce random error.<br>
### Impact of Interaction Sequence
Our final experiment changes the order in the original sequence by data shuffling. We keep the validation set and test set unchanged, get new pseudo-sequences by disrupting the order of user interaction sequences in the training set, and observe the performance changes of the sequence recommendation algorithm.<br>

## Acknowledgements
We build on the following repositories to improve our codes for customized experiments, which also ensures the reproducibility and reliability of our results.
- [DaisyRec 2.0](https://github.com/recsys-benchmark/DaisyRec-v2.0)
- [SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)
- [TiSASRec.pytorch](https://github.com/pmixer/TiSASRec.pytorch)
- [Caser.pytorch](https://github.com/graytowne/caser_pytorch)
