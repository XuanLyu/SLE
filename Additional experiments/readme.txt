We present the process of evaluating different models in this folder:

1.The results of Simulation study is reported in ``Simulation study on dsitribution-uncertain...''

2. The comparisons of distribution-uncertain methods and SOTA methods in terms of time-cost and accuracy are presented in ``comparisons of training time...''. These comparisons are conducted on three CelebA datasets.

3.  The comprehensive comparisons of distribution-uncertain methods and all of the SOTA methods can be found in other ``.ipnb'' files, each of which corresponds to a dataset.

To make it more clear, we have packaged the related codes of ``Mean-uncertain method'' and ``Volatility-uncertain method'' in folder ``Distribution-uncertain methods'', including both the process of estimation and cross-validation. An example is given thereafter to show how the codes work in practice.

To compare all the methods in the same platform, we provide the codes of DRM and max-mean loss methods, which are not available in ``python'' environment at least, in the folder ``Supplimentary codes''. Also, an example is given thereafter to show how the codes work in practice.



Finally, all codes can be run in jupyter.
