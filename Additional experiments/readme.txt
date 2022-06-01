We present the process of evaluating different models in this folder:

1.The results of Simulation study is reported in ``Simulation study on dsitribution-uncertain...''

2. The comparisons of distribution-uncertain methods and SOTA methods in terms of time-cost and accuracy are presented in ``comparisons of training time...''. These comparisons are conducted on three CelebA datasets.

3.  The comprehensive comparisons of distribution-uncertain methods and all of the SOTA methods can be found in other ``.ipnb'' files, each of which corresponds to a dataset. Not all the results are uploaded here, because the process for different datasets are the same. You can contact the writer to request a detailed process on any dataset.

To make it more clear, we have packaged the related codes of ``Mean-uncertain method'' and ``Volatility-uncertain method'' in folder ``Distribution-uncertain methods'', including both the process of estimation and cross-validation. An example is given thereafter to show how the codes work in practice.

To compare all the methods in the same platform, we provide the codes of DRM and max-mean loss methods, which are not available in ``python'' environment at least, in the folder ``Supplimentary codes''. Also, an example is given thereafter to show how the codes work in practice.

The folder ``Fivefold cross validation'' summarizes the performances of classifiers, depend on which we select parameters. The files in that folder mainly contain the cross-validation results of DRM, and mean-uncertain LR on simulated data as well as the CelebA datasets. As for parameters selected for distribution-uncertain methods on other datasets, they are decided automaticly in our codes. See ``mean-uncertain method'' file in the folder ``Distribution-uncertain methods'' for detailed codes.

Finally, all codes can be run in jupyter.
