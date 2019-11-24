import numpy as np
import os
import csv
import time
import argparse
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import scipy
from scipy.spatial.distance import pdist, squareform
import dataset
from dataset import *

""" Code for Gaussian Process Regression for EM duct characterization """

np.random.seed(0)

class fitted_gpr_model:
    def __init__(self, args, data):
        """ Initialize class atributes and call process_data method
            [self]              class instance
            [args]              arguments from parser
            [data]              data object
        """
        self.data = data
        self.fit_time, self.pred_MC = None, []
        # get trained model by calling perform_fitting method
        if args.csv == 'case1':
            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
        else:
            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 5e1)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
        self.model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
        self.perform_fitting()
        # get predictions and std on training data by calling perform_inference method
        train_results = self.perform_inference(args, 'x_train', None)
        self.pred_tr, self.std_tr = train_results['pred'], train_results['std']

    def perform_fitting(self):
        """ Fit GP to training points and set model and time attributes
            [self]              class instance
        """
        start_time = time.clock()
        self.model.fit(self.data.x_train, self.data.y_train)
        self.fit_time = time.clock() - start_time

    def perform_inference(self, args, varname, beta):
        """ Returns dictionary with predictions, std, MSE
            [self]              class instance
            [args]              arguments from parser
            [varname]           variable name (corresponding to data)
            [beta]              location in data list
        """
        X = getattr(self.data, varname)                                         # get appropriate X using variable name and beta
        if beta != None:
            X = X[beta]
            beta2 = int(np.floor(beta/len(args.aug_num)))                       # calculate beta for IVW
            aug_num = args.aug_num[beta % len(args.aug_num)]                    # get appropriate aug num from list
        color = ['white', 'pink']                                               # list of colors
        results = {}                                                            # dictionary to store results
        start_time = time.clock()                                               # calculate inference time
        pp_pred, pp_std = self.model.predict(X, return_std=True)                # make predictions

        # perform calculations to update predictions and standard dev to account for noise
        if varname == 'x_test_MC':
            pp_pred = pp_pred.reshape(-1, 1000)     # reshape preprocessed pred to block
            pp_std = pp_std.reshape(-1, 1000)       # reshape preprocessed std to block
            pred = np.mean(pp_pred, axis=1)         # calculate pred by averaging pp_pred
            # calculate var using Eq. 20
            var = np.mean(np.square(pp_std), axis=1) + np.mean(np.square(pp_pred), axis=1) - np.square(pred)
            std = np.sqrt(var)
        elif varname == 'x_test_IVW':
            pp_pred = pp_pred.reshape(-1,aug_num)         # reshape preprocessed pred to block
            pp_std = pp_std.reshape(-1,aug_num)           # reshape preprocessed std to block
            # calculate the mixing proportion using Eq. 21
            pp_var = np.square(pp_std)
            pp_var_inv = np.array(1/pp_var)
            numerator = pp_var_inv
            denominator = np.repeat(np.expand_dims(np.sum(pp_var_inv, axis=1), axis=1), aug_num, axis=1)
            f_var_ratio = numerator/denominator
            # calulate predictions and var using Eq. 19 & 20
            pred = np.sum(np.multiply(f_var_ratio, pp_pred), axis=1)
            m_w_sq = np.sum(np.multiply(f_var_ratio, np.square(pp_pred)), axis=1)
            var_w = np.sum(np.multiply(f_var_ratio, pp_var), axis=1)
            var = var_w + m_w_sq - np.square(pred)
            std = np.sqrt(var)
        else:
            pred = pp_pred
            std = pp_std

        # stop timer then make plots
        if varname != 'x_train':
            results['inf_time'] = time.clock() - start_time
            results['mse'] = self.calculate_mse(pred, None)
            if varname == 'x_test_clean':
                self.plot(args, pred, std, 'Noiseless', 'clean'+str(args.ratio))
            else:
                if varname == 'x_test_MC':
                    self.pred_MC.append(pred)
                    self.plot(args, pred, std, 'Monte Carlo Ground Truth ('+color[beta]+')', 'MC'+color[beta]+str(args.ratio))
                elif varname == 'x_test_IVW':
                    results['mse_MC'] = self.calculate_mse(pred, self.pred_MC[beta2])
                    self.plot(args, pred, std, 'IVW'+str(aug_num)+' Approach ('+color[beta2]+')', 'IVW'+str(aug_num)+color[beta2]+str(args.ratio))
                else:
                    results['mse_MC'] = self.calculate_mse(pred, self.pred_MC[beta])
                    self.plot(args, pred, std, 'Naive Approach ('+color[beta]+')', 'naive'+color[beta]+str(args.ratio))

        results['pred'], results['std'] = pred, std
        return results

    def plot(self, args, pred_te, std_te, ptitle, pfile):
        """ Create plots for paper
            [self]              class instance
            [args]              arguments from parser
            [pred_te]           predictions for test points
            [std_te]            standard deviation for test points
            [ptitle]            title for plot
            [pfile]             filename for saved plot
        """
        # recombine training and calculated test predictions to plot
        pred = np.concatenate([self.pred_tr, pred_te])
        std = np.concatenate([self.std_tr, std_te])
        y = np.concatenate([self.data.y_train, self.data.y_test])

        # reorder predictions for plotting
        y, pred, std = zip(*sorted(zip(y, pred, std)))
        std = np.array(std)

        # ground-truth function (only for plotting)
        f = np.arange(2,41)

        # plot (adapted from scikit-learn)
        plt.figure()
        plt.plot(f, f, 'r:', label=u'Ground Truth')
        plt.plot(self.data.y_train, self.data.y_train, 'r.', markersize=7, label=u'Observations')
        plt.plot(y, pred, 'b-', label=u'GP Mean')
        plt.fill(np.concatenate([y, y[::-1]]),
             np.concatenate([pred - 1.9600 * std,
                                (pred + 1.9600 * std)[::-1]]),
                                alpha=.5, fc='#B4CFEC', ec='None', label='95% confidence interval')
        plt.title(ptitle)
        plt.xlabel('Actual duct height')
        plt.ylabel('Predicted duct height')
        plt.ylim(2, 40)
        plt.xlim(2, 40)
        plt.legend(loc='upper left')
        plt.savefig(args.folder+'/'+args.csv+'/'+pfile+'.pdf')

    def calculate_mse(self, pred, ref):
        """ Create plots for paper
            [self]              class instance
            [pred]              predictions to calculate mse
            [ref]               reference to calculate mse against
        """
        # if no reference is provided, use true labels to calculate mse
        pred = np.squeeze(pred)
        if ref is None:
            ref = self.data.y_test
        return np.mean(np.subtract(pred, ref)**2)

    def calculate_pairwise_dist(self, varname, beta):
        """ Create plots for paper
            [self]              class instance
            [varname]           variable name (corresponding to data)
            [beta]              0 for white noise, 1 for pink noise
        """
        # calculate distances
        X_te = getattr(self.data, varname)
        if beta != None:
            X_te = X_te[beta]
        n = len(self.data.y_test)
        X = np.concatenate([X_te, self.data.x_train])
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        # calculate distance to closest training point
        min_dis = np.min(pairwise_dists[0:n, n:], axis=1)
        # return average distance
        return np.mean(min_dis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
    parser.add_argument('--folder', type=str, default=time.strftime("%Y%m%d%H%M%S") , help='filename for results')
    parser.add_argument('--ratio', type=int, default=20 , help='ratio in testing set')
    parser.add_argument('--clean', action='store_false', help='skip over clean data')
    parser.add_argument('--noise', action='store_false', help='skip over noisy data')
    parser.add_argument('--aug_num', type=int, default=(5, 10) , help='list of numbers to augment test input for inverse-variance weighting (only if --noise=True)')
    parser.add_argument('--paper', action='store_false', help='if all specifications are same as paper, set true')
    args = parser.parse_args()

    # create directory if it does not exist
    if not os.path.exists(args.folder+'/'+args.csv):
        os.makedirs(args.folder+'/'+args.csv)

    if args.clean == False and args.noise == False:
        raise Exception('must have at least clean or noise == True')

    ### GET DATA, FIT MODEL
    data = dataset(args)
    gpr = fitted_gpr_model(args, data)

    ### INFERENCE
    if args.clean:
        results_clean = gpr.perform_inference(args, 'x_test_clean', None)
        av_dis_clean = gpr.calculate_pairwise_dist('x_test_clean', None)

    if args.noise:
        MC, naive, IVW, av_dis_noise = [], [], [], []
        for beta in range(2):
            MC.append(gpr.perform_inference(args, 'x_test_MC', beta))
            naive.append(gpr.perform_inference(args, 'x_test_naive', beta))
            av_dis_noise.append(gpr.calculate_pairwise_dist('x_test_naive', beta))
        for beta in range(2*len(args.aug_num)):
            IVW.append(gpr.perform_inference(args, 'x_test_IVW', beta))
            # e.g. white 5, white 10, pink 5, pink 10

    # store information to correspond to table format in paper (write your own if not true)
    if args.paper:
        MC_white, MC_pink = MC[0], MC[1]
        naive_white, naive_pink = naive[0], naive[1]
        IVW5_white, IVW10_white, IVW5_pink, IVW10_pink = IVW[0], IVW[1], IVW[2], IVW[3]
        white_mse_row = ('white', MC_white['mse'], naive_white['mse'], IVW5_white['mse'], IVW10_white['mse'], 100*(naive_white['mse']-IVW5_white['mse'])/naive_white['mse'], 100*(naive_white['mse']-IVW10_white['mse'])/naive_white['mse'])
        pink_mse_row  = ('pink', MC_pink['mse'], naive_pink['mse'], IVW5_pink['mse'], IVW10_pink['mse'], 100*(naive_pink['mse']-IVW5_pink['mse'])/naive_pink['mse'], 100*(naive_pink['mse']-IVW10_pink['mse'])/naive_pink['mse'])
        white_MC_mse_row = (' ', naive_white['mse_MC'], IVW5_white['mse_MC'], IVW10_white['mse_MC'], 100*(naive_white['mse_MC']-IVW5_white['mse_MC'])/naive_white['mse_MC'], 100*(naive_white['mse_MC']-IVW10_white['mse_MC'])/naive_white['mse_MC'])
        pink_MC_mse_row = (' ', naive_pink['mse_MC'], IVW5_pink['mse_MC'], IVW10_pink['mse_MC'], 100*(naive_pink['mse_MC']-IVW5_pink['mse_MC'])/naive_pink['mse_MC'], 100*(naive_pink['mse_MC']-IVW10_pink['mse_MC'])/naive_pink['mse_MC'])

        av_dis_row = (' ', av_dis_clean, av_dis_noise[0], av_dis_noise[0]/av_dis_clean, av_dis_noise[1], av_dis_noise[1]/av_dis_clean)
        av_time_MC = (MC_white['inf_time']+MC_pink['inf_time'])/2
        av_time_naive = (naive_white['inf_time']+naive_pink['inf_time'])/2
        av_time_IVW5 = (IVW5_white['inf_time']+IVW5_pink['inf_time'])/2
        av_time_IVW10 = (IVW10_white['inf_time']+IVW10_pink['inf_time'])/2
        time_row = (' ', gpr.fit_time, av_time_MC, av_time_naive, av_time_IVW5, av_time_IVW10)

        csv_filename = args.folder+'/'+args.csv+'/results.csv'
        with open(csv_filename, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # paper info:
            writer.writerow(['Ratio '+str(args.ratio), 'Training pts '+str(len(data.y_train))])
            writer.writerow(['MSE loss to label: Clean', results_clean['mse']])
            writer.writerow(['MSE loss to label:', 'MC', 'Naive', 'IVW5', 'IVW10', 'IVW5Imp', 'IVW10Imp'])
            writer.writerow(white_mse_row)
            writer.writerow(pink_mse_row)
            writer.writerow(['MSE loss to ground truth', 'Naive', 'IVW5', 'IVW10', 'IVW5Imp', 'IVW10Imp'])
            writer.writerow(white_MC_mse_row)
            writer.writerow(pink_MC_mse_row)
            writer.writerow(['Average distance between train and test', 'Clean', 'Naive (white)', 'Times', 'Naive (pink)', 'Times'])
            writer.writerow(av_dis_row)
            writer.writerow(['Time', 'Training', 'MC', 'Naive', 'IVW5', 'IVW10'])
            writer.writerow(time_row)
