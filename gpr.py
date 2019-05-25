import numpy as np
import os
import csv
import time
import pandas
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import scipy
from scipy.spatial.distance import pdist, squareform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
parser.add_argument('--folder', type=str, default=time.strftime("%Y%m%d-%H%M%S") , help='filename for results')
parser.add_argument('--ratio', type=int, default=20 , help='ratio in testing set')
parser.add_argument('--end', action='store_false', help='include endpoint in training')
parser.add_argument('--clean', action='store_false', help='skip over clean data')
parser.add_argument('--noise', action='store_false', help='skip over noisy data')
parser.add_argument('--MC', action='store_true', help='perform montecarlo sampling (only if --noise=True)')
parser.add_argument('--aug_num', type=int, default=1 , help='number to augment test input for inverse-variance weighting (only if --noise=True)')
args = parser.parse_args()

if (args.MC == True or args.aug_num > 1) and args.noise == False:
    raise Exception('cannot augment data without noise')
if args.clean == False and args.noise == False:
    raise Exception('must have at least clean or noise == True')

""" Code for Gaussian Process Regression for EM duct characterization
    If MC is True, perform Monte carlo sampling to get ground truth
    If aug_num > 1, also perform inverse-variance weighting
 """

# calculate case number
case_str = args.csv
case_title = 'case '+str(int(case_str[4]))

# create directory if it does not exist
if not os.path.exists(args.folder):
    os.makedirs(args.folder)

def shuffle(x,y):
    """ Returns shuffled x and y arrays
        [x],[y]         list or 1D array
    """
    x, y = np.array(x), np.array(y)
    n_y = len(y)
    index_array = np.arange(n_y)
    np.random.shuffle(index_array)
    sx, sy = [], []
    for idx, val in enumerate(index_array):
        sx.append(x[val])
        sy.append(y[val])
    sx, sy = np.array(sx), np.array(sy)
    return sx, sy

class dataset:
    def __init__(self, args):
        self.x_train, self.y_train = None, None # stores x and y for training
        self.y_test_no_noise = None # stores unaugmented y
        self.x_test_no_noise, self.x_test_noise = None, None # stores a clean and noisy version of x
        self.x_test_MC, self.y_test_MC = None, None # x and y augmented 1000 times for MC
        self.x_test_aug, self.y_test_aug = None, None # x and y augmented aug_num times for variance-weighting
        self.process_data(args)

    def process_data(self, args):
        """ Import dataset from csv and format into feature and labels np float arrays
            [args]          arguments from parser
        """
        df = pandas.read_csv('data/'+case_str+'.csv')
        data = df.values
        m,n = data.shape # m - input dimension, # n - examples
        inputs, labels = [], []
        header = list(df)

        # seperate dataframe into inputs and labels
        for ind, val in enumerate(header):
            inp = np.float32(data[:,ind])
            label = np.float32(val[1:].replace('_','.'))
            inputs.append(inp)
            labels.append(label)

        # calculate split index
        ind = np.int64(np.floor(len(labels)*(args.ratio/100)))

        # if args.end == True (put endpoints into the training set), take out the endpoints and save them
        if args.end == True:
            # endpoints
            inputs_end = np.concatenate([np.expand_dims(inputs[0], axis=0), np.expand_dims(inputs[n-1], axis=0)], axis=0)
            labels_end = np.array([labels[0], labels[n-1]])
            # dataset without endpoints
            inputs = inputs[1:n-1]
            labels = labels[1:n-1]

        inputs = np.array(inputs)
        labels = np.array(labels)
        inputs, labels = shuffle(inputs, labels)

        # TRAINING DATA
        self.x_train, self.y_train = inputs[ind:,:], labels[ind:]
        # put back the endpoints into the training set
        if args.end == True:
            self.x_train = np.concatenate([self.x_train, inputs_end])
            self.y_train = np.concatenate([self.y_train, labels_end])

        # TESTING DATA (CLEAN)
        self.x_test_no_noise, self.y_test_no_noise = inputs[0:ind,:], labels[0:ind]
        #self.y_test_no_noise, self.x_test_no_noise = zip(*sorted(zip(self.y_test_no_noise, self.x_test_no_noise)))
        #self.y_test_no_noise, self.x_test_no_noise = np.array(self.y_test_no_noise), np.array(self.x_test_no_noise)

        # TESTING DATA (NOISY)
        if args.noise: # naive
            self.x_test_noise, _ = self.add_noise(1, self.x_test_no_noise, self.y_test_no_noise)
        if args.MC: # augment 1000 times for MC
            self.x_test_MC, self.y_test_MC = self.add_noise(1000, self.x_test_no_noise, self.y_test_no_noise)
        if args.aug_num > 1: # augment aug_num times for variance-weighting
            self.x_test_aug, self.y_test_aug = self.add_noise(args.aug_num, self.x_test_no_noise, self.y_test_no_noise)

    def add_noise(self, aug_num, inp, labels):
        """ Create and then add gaussian noise - zero mean and std of 0.1*||inp||_infty
            [inp]           array that needs noise
        """
        inf_norm = (abs(inp)).max(axis=1)
        _, d = inp.shape
        std_v = np.transpose(np.tile(0.1*inf_norm, (d,1)))
        inp = np.repeat(inp, aug_num, axis=0)
        std_v = np.repeat(std_v, aug_num, axis=0)
        labels = np.repeat(labels, aug_num, axis=0)
        noise = np.random.normal(loc=0,scale=std_v,size=inp.shape)
        return inp+noise, labels

### DATASET
data = dataset(args)

### TRAINING/FITTING
# training points
X_tr = data.x_train
y_tr = data.y_train
print('Training pts: ', len(y_tr))

# fit GP to training points
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
start_fit = time.clock()
model.fit(X_tr, y_tr)
fit_time = time.clock() - start_fit
params = model.kernel_.get_params()

### INFERENCE
y_ts_no = data.y_test_no_noise
num = len(y_ts_no)

# CLEAN DATA
if args.clean:
    # combine test and training points
    X_ts_no = data.x_test_no_noise
    X_no = np.concatenate([X_ts_no, X_tr])
    y_no = np.concatenate([y_ts_no, y_tr])

    # calculate distances
    pairwise_dists = squareform(pdist(X_no, 'euclidean'))
    max_dis = np.min(pairwise_dists[0:num, num:], axis=1)
    av_dis_clean = np.mean(max_dis)

    # predict observations
    y_pred_no, std_no = model.predict(X_no, return_std=True)

    # decompose predictions to only calculate metrics on test observations
    y_pred_no_ts = y_pred_no[0:num]
    std_no_ts = std_no[0:num]
    mse_loss_no = ((y_pred_no_ts-y_ts_no)**2).mean()
    print('Clean MSE loss (to true label):', mse_loss_no)

    # reorder predictions for plotting
    y, y_pred_no, std_no = zip(*sorted(zip(y_no, y_pred_no, std_no)))
    std_no = np.array(std_no)

    # ground-truth function (only for plotting)
    f = np.arange(2,41)

    # plot (adapted from scikit-learn)
    plt.figure()
    plt.plot(f, f, 'r:', label=u'Ground Truth')
    plt.plot(y_tr, y_tr, 'r.', markersize=7, label=u'Observations')
    plt.plot(y, y_pred_no, 'b-', label=u'GP Mean')
    plt.fill(np.concatenate([y, y[::-1]]),
         np.concatenate([y_pred_no - 1.9600 * std_no,
                            (y_pred_no + 1.9600 * std_no)[::-1]]),
                            alpha=.5, fc='#B4CFEC', ec='None', label='95% confidence interval')
    plt.title('Noise-less, '+case_title)
    plt.xlabel('Actual duct height')
    plt.ylabel('Predicted duct height')
    plt.ylim(2, 40)
    plt.xlim(2, 40)
    plt.legend(loc='upper left')
    plt.savefig(args.folder+'/'+case_str+'_'+str(args.ratio)+'_'+str(args.end)+'clean.pdf')
    #plt.show()

if args.MC:
    # combined points with noise and augmentation on test
    X_ts_MC = data.x_test_MC
    y_ts_MC = data.y_test_MC
    X_MC = np.concatenate([X_ts_MC, X_tr])
    y_MC = np.concatenate([y_ts_MC, y_tr])

    # predict all observations
    y_pred_MC, std_MC = model.predict(X_MC, return_std=True)

    # decompose predictions back into training and testing predictions
    n, _ = X_ts_MC.shape
    y_pred_block_ts_MC = y_pred_MC[0:n].reshape(-1, 1000)
    std_block_ts_MC = std_MC[0:n].reshape(-1, 1000)
    y_pred_tr_MC = y_pred_MC[n:]
    std_tr_MC = std_MC[n:]

    # calculate prediction and variance for MC
    y_pred_ts_MC = np.mean(y_pred_block_ts_MC, axis=1)
    var_ts_MC = np.mean(np.square(std_block_ts_MC), axis=1) + np.mean(np.square(y_pred_block_ts_MC), axis=1) - np.square(y_pred_ts_MC)
    std_ts_MC = np.sqrt(var_ts_MC)

    #calculate MSE (to true label)
    mse_loss_MC = ((y_pred_ts_MC-y_ts_no)**2).mean()
    print('MC MSE loss (to true label):', mse_loss_MC)

    # recombine for plotting
    y_pred_comb_MC = np.concatenate([y_pred_ts_MC, y_pred_tr_MC])
    std_comb_MC = np.concatenate([std_ts_MC, std_tr_MC])

    # for plotting, reorder observations:
    y, y_pred_comb_MC, std_comb_MC = zip(*sorted(zip(y_no, y_pred_comb_MC, std_comb_MC)))
    std_comb_MC = np.array(std_comb_MC)

    # plot (adapted from scikit-learn)
    plt.figure()
    plt.plot(y_tr, y_tr, 'r.', markersize=7, label=u'Observations')
    plt.plot(y, y_pred_comb_MC, 'b-', label=u'GP Mean')
    plt.fill(np.concatenate([y, y[::-1]]),
         np.concatenate([y_pred_comb_MC - 1.9600 * std_comb_MC,
                            (y_pred_comb_MC + 1.9600 * std_comb_MC)[::-1]]),
                            alpha=.5, fc='#B4CFEC', ec='None', label='95% confidence interval')
    plt.title('Monte Carlo, "Ground-Truth", '+case_title)
    plt.xlabel('Actual duct height')
    plt.ylabel('Predicted duct height')
    plt.ylim(2, 40)
    plt.xlim(2, 40)
    plt.legend(loc='upper left')
    plt.savefig(args.folder+'/'+case_str+'_'+str(args.ratio)+'_'+str(args.aug_num)+'_'+str(args.end)+'MC.pdf')
    #plt.show()

# NOISY DATA
if args.noise:
    # combine test and training points (labels same as no noise)
    X_ts_1 = data.x_test_noise
    X_1 = np.concatenate([X_ts_1, X_tr])

    # calculate distances
    pairwise_dists_1 = squareform(pdist(X_1, 'euclidean'))
    max_dis_1 = np.min(pairwise_dists_1[0:num, num:], axis=1)
    av_dis_noise = np.mean(max_dis_1)

    # predict observations
    y_pred_1, std_1 = model.predict(X_1, return_std=True)

    # decompose predictions to only calculate metrics on test observations
    y_pred_1_ts = y_pred_1[0:num]
    std_1_ts = std_1[0:num]
    mse_loss_1 = ((y_pred_1_ts-y_ts_no)**2).mean()
    print('Noisy MSE loss (to true label):', mse_loss_1)
    if args.MC:
        mse_loss_1_gt = ((y_pred_1_ts-y_pred_ts_MC)**2).mean()
        print('Noisy MSE loss (to ground truth label):', mse_loss_1_gt)

    # reorder predictions for plotting
    y, y_pred_1, std_1 = zip(*sorted(zip(y_no, y_pred_1, std_1)))
    std_1 = np.array(std_1)

    # plot (adapted from scikit-learn)
    plt.figure()
    plt.plot(y_tr, y_tr, 'r.', markersize=7, label=u'Observations')
    plt.plot(y, y_pred_1, 'b-', label=u'GP Mean')
    plt.fill(np.concatenate([y, y[::-1]]),
         np.concatenate([y_pred_1 - 1.9600 * std_1,
                            (y_pred_1 + 1.9600 * std_1)[::-1]]),
                            alpha=.5, fc='#B4CFEC', ec='None', label='95% confidence interval')
    plt.title('Na$\ddot{\imath}$ve Approach, '+case_title)
    plt.xlabel('Actual duct height')
    plt.ylabel('Predicted duct height')
    plt.ylim(2, 40)
    plt.xlim(2, 40)
    plt.legend(loc='upper left')
    plt.savefig(args.folder+'/'+case_str+'_'+str(args.ratio)+'_'+str(args.aug_num)+'_'+str(args.end)+'noise.pdf')
    #plt.show()

if args.aug_num > 1:
    # combined points with noise and augmentation on test
    X_ts_aug = data.x_test_aug
    y_ts_aug = data.y_test_aug
    X_aug = np.concatenate([X_ts_aug, X_tr])
    y_aug = np.concatenate([y_ts_aug, y_tr])

    # predict all observations
    y_pred_aug, std_aug = model.predict(X_aug, return_std=True)

    # decompose predictions back into training and testing predictions
    n, _ = X_ts_aug.shape
    y_pred_block_ts_aug = y_pred_aug[0:n].reshape(-1, args.aug_num)
    std_block_ts_aug = std_aug[0:n].reshape(-1, args.aug_num)
    var_block_ts_aug = np.square(std_block_ts_aug)
    y_pred_tr_aug = y_pred_aug[n:]
    std_tr_aug = std_aug[n:]

    # calculate scaled variance
    var_ts_aug_scaled = np.array(1/var_block_ts_aug)
    numerator = var_ts_aug_scaled
    denominator = np.repeat(np.expand_dims(np.sum(var_ts_aug_scaled, 1), axis=1), args.aug_num, axis=1)
    f_var_ratio = numerator/denominator

    y_pred_ts_aug = np.sum(np.multiply(f_var_ratio, y_pred_block_ts_aug), axis=1)
    m_w_sq = np.sum(np.multiply(f_var_ratio, np.square(y_pred_block_ts_aug)), axis=1)
    var_w = np.sum(np.multiply(f_var_ratio, var_block_ts_aug), axis=1)
    var_ts_aug = var_w + m_w_sq - np.square(y_pred_ts_aug)
    mse_loss_aug = ((y_pred_ts_aug-y_ts_no)**2).mean()
    print('IVW MSE loss (to true label):', mse_loss_aug)
    if args.MC:
        mse_loss_aug_gt = ((y_pred_ts_aug-y_pred_ts_MC)**2).mean()
        print('IVW MSE loss (to ground truth label):', mse_loss_aug_gt)

    std_ts_aug = np.sqrt(var_ts_aug)

    y_pred_comb_aug = np.concatenate([y_pred_ts_aug, y_pred_tr_aug])
    std_comb_aug = np.concatenate([std_ts_aug, std_tr_aug])

    # for plotting, reorder observations:
    y, y_pred_comb_aug, std_comb_aug = zip(*sorted(zip(y_no, y_pred_comb_aug, std_comb_aug)))
    std_comb_aug = np.array(std_comb_aug)

    # plot (adapted from scikit-learn)
    plt.figure()
    plt.plot(y_tr, y_tr, 'r.', markersize=7, label=u'Observations')
    plt.plot(y, y_pred_comb_aug, 'b-', label=u'GP Mean')
    plt.fill(np.concatenate([y, y[::-1]]),
         np.concatenate([y_pred_comb_aug - 1.9600 * std_comb_aug,
                            (y_pred_comb_aug + 1.9600 * std_comb_aug)[::-1]]),
                            alpha=.5, fc='#B4CFEC', ec='None', label='95% confidence interval')
    plt.title('Inverse-Variance Weighting Approach, '+case_title)
    plt.xlabel('Actual duct height')
    plt.ylabel('Predicted duct height')
    plt.ylim(2, 40)
    plt.xlim(2, 40)
    plt.legend(loc='upper left')
    plt.savefig(args.folder+'/'+case_str+'_'+str(args.ratio)+'_'+str(args.aug_num)+'_'+str(args.end)+'weighted.pdf')
    #plt.show()

## TIMING CALCULATIONS (ONLY INTERESTED IN THE TIMING THE PREDICTIONS & STD FROM TEST SET)
#calculate timing info for 1 example
start_inf = time.clock()
_, _ = model.predict(X_ts_no, return_std=True)
inf_time_1 = time.clock() - start_inf

#calculate timing info for MCMC
if args.MC:
    n, _ = X_ts_MC.shape
    start_inf = time.clock()
    pred_MC_time, std_MC_time = model.predict(X_ts_MC, return_std=True)
    y_pred_block_ts_MC_time = pred_MC_time[0:n].reshape(-1, 1000)
    std_block_ts_MC_time = std_MC_time[0:n].reshape(-1, 1000)
    # calculate prediction and variance for MC
    y_pred_ts_MC_time = np.mean(y_pred_block_ts_MC_time, axis=1)
    var_ts_MC_time = np.mean(np.square(std_block_ts_MC_time), axis=1) + np.mean(np.square(y_pred_block_ts_MC_time), axis=1) - np.square(y_pred_ts_MC_time)
    std_ts_MC_time = np.sqrt(var_ts_MC_time)
    inf_time_MC = time.clock() - start_inf

# calculate timing info for inverse-variance weighting
if args.aug_num > 1:
    n, _ = X_ts_aug.shape
    start_inf = time.clock()
    pred_aug_time, std_aug_time = model.predict(X_ts_aug, return_std=True)
    y_pred_block_ts_aug_time = pred_aug_time[0:n].reshape(-1, args.aug_num)
    std_block_ts_aug_time = std_aug_time[0:n].reshape(-1, args.aug_num)
    var_block_ts_aug_time = np.square(std_block_ts_aug_time)
    var_ts_aug_scaled_time = np.array(1/var_block_ts_aug_time)
    numerator_time = var_ts_aug_scaled_time
    denominator_time = np.repeat(np.expand_dims(np.sum(var_ts_aug_scaled_time, 1), axis=1), args.aug_num, axis=1)
    f_var_ratio_time = numerator_time/denominator_time
    y_pred_ts_aug_time = np.sum(np.multiply(f_var_ratio_time, y_pred_block_ts_aug_time), axis=1)
    m_w_sq_time = np.sum(np.multiply(f_var_ratio_time, np.square(y_pred_block_ts_aug_time)), axis=1)
    var_w_time = np.sum(np.multiply(f_var_ratio_time, var_block_ts_aug_time), axis=1)
    var_ts_aug_time = var_w_time + m_w_sq_time - np.square(y_pred_ts_aug_time)
    std_ts_aug_time = np.sqrt(var_ts_aug_time)
    inf_time_aug = time.clock() - start_inf

#open csv file to store information
csv_filename = args.folder+'/results'+case_str+'.csv'
with open(csv_filename, 'a') as csv_file:
    writer = csv.writer(csv_file)
    # background info
    writer.writerow(['ratio', args.ratio])
    writer.writerow(['training pts', len(y_tr)])
    writer.writerow(['end', args.end])
    writer.writerow(['params', params])
    writer.writerow(['av_dis_clean', av_dis_clean])
    writer.writerow(['av_dis_noise', av_dis_noise])
    # predictions & MSE losses
    writer.writerow(['y_test_original', data.y_test_no_noise])
    if args.clean:
        writer.writerow(['y_pred_no', y_pred_no_ts])
        writer.writerow(['mse_loss_clean', mse_loss_no])
    if args.MC:
        writer.writerow(['y_pred_MC', y_pred_ts_MC])
        writer.writerow(['mse_loss_MC', mse_loss_MC])
    if args.noise:
        writer.writerow(['y_pred_noise', y_pred_1_ts])
        writer.writerow(['mse_loss_noise', mse_loss_1])
        if args.MC:
            writer.writerow(['mse_loss_noise_gt', mse_loss_1_gt])
    if args.aug_num > 1:
        writer.writerow(['y_pred_aug', y_pred_ts_aug])
        writer.writerow(['mse_loss_aug', mse_loss_aug])
        if args.MC:
            writer.writerow(['mse_loss_aug_gt', mse_loss_aug_gt])
    # timing info
    writer.writerow(['train_time', fit_time])
    writer.writerow(['test_time', inf_time_1])
    if args.MC:
        writer.writerow(['test_time_MC', inf_time_MC])
    if args.aug_num > 1:
        writer.writerow(['test_time_aug', inf_time_aug])

# better format csv for paper
csv_filename = args.folder+'/results.csv'
with open(csv_filename, 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['case', case_str])
    writer.writerow(['ratio', args.ratio])
    writer.writerow(['training pts', len(y_tr)])
    writer.writerow(['end', args.end])
    writer.writerow(['av_dis_clean', av_dis_clean])
    writer.writerow(['av_dis_noise', av_dis_noise])
    if args.clean:
        writer.writerow(['mse_loss_no', mse_loss_no])
    if args.MC:
        writer.writerow(['mse_loss_MC', mse_loss_MC])
    if args.noise:
        writer.writerow(['mse_loss_noise', mse_loss_1])
        if args.MC:
            writer.writerow(['mse_loss_noise_gt', mse_loss_1_gt])
    if args.aug_num > 1:
        writer.writerow(['mse_loss_aug', mse_loss_aug])
        if args.MC:
            writer.writerow(['mse_loss_aug_gt', mse_loss_aug_gt])
    writer.writerow(['train_time', fit_time])
    writer.writerow(['test_time', inf_time_1])
    if args.MC:
        writer.writerow(['test_time_MC', inf_time_MC])
    if args.aug_num > 1:
        writer.writerow(['test_time_aug', inf_time_aug])
