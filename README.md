# Gaussian process regression for EM duct height estimation

Accompanying paper: Gaussian Process Regression for Estimating EM Ducting Within the Marine Atmospheric Boundary Layer. Hilarie Sit and Christopher J. Earls.

Gaussian process regression (GPR) using sci-kit learn for predicting evaporation duct height from EM propagation factor data. For noise-contaminated test inputs, the ground truth posterior predictive distribution by using 1000 Monte-Carlo (MC) samples with equal mixing proportions. Inverse-variance weighted proportions with a specified number of samples (aug_num) to calculate predictions and uncertainty is available. Dataset and results are located in their respective folders. 

## Run gaussian process regression
Specify name of csv data file and the testing ratio. If noise is applied to the propagation factor measurement, you can use --MC tag to calculate the ground truth and specify the --aug_num tag to use inverse-variance weighted proportions. Automatically performs GPR naively on clean and noise-contaminated test inputs. To exclude either, include --clean or --noise tags. 

```bash
python gpr.py --csv case1 --ratio 20 --aug_num (5, 10)
```
The attached bash script loops over different testing ratios.
```bash
bash run_gpr.sh
```
