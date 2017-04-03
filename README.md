# TLGProb

There is a growing interest in applying machine learning algorithms to real-world examples by explicitly deriving models based on statistical reasoning. Sports analytics, being favoured mostly by the statistics community and less discussed in the machine learning community, becomes our focus in this paper. Specifically, we model two-team sports for the sake of one-match-ahead forecasting. We present a pioneering approach based on stacked Bayesian regressions. Benefiting from regression flexibility and high standard of performance, Sparse Spectrum Gaussian Process Regression (SSGPR), which improves the standard Gaussian Process Regression (GPR), was chosen to carry out Bayesian regression, resulting in a novel predictive model called TLGProb. For evaluation, the models were applied on a popular sports event -- National Basketball Association (NBA). Finally, with TLGProb, 85.28\% of the matches in NBA 2014/2015 season were correctly predicted, surpassing other prediction methods.

For any enquiries, please email me at maxingaussian@gmail.com

## Highlight: Player's Ability Inferred From Player's Performance
![lebron](experiment-2014-2015/lebron_james_3p_fg.png?raw=true "LeBron James")

## Highlight: Two-Layer Gaussian Process Regression Model
![TLGstructure](experiment-2014-2015/TLGProb.png?raw=true "TLG structure")


## Experimental Results
![AccuracyVsRejection](experiment-2014-2015/accuracy_vs_rejection.png?raw=true "Accuracy vs Rejection")