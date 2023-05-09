"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import recall_score, precision_score
import csv

import sys
sys.path.append('../..')
from CLIMAX.slime_lm._least_angle import lars_path

from CLIMAX.grad_utils import grad_logloss_theta_lr
from CLIMAX.grad_utils import batch_grad_logloss_lr
from CLIMAX.inverse_hvp import inverse_hvp_lr_newtonCG
from CLIMAX.dataset import select_from_one_class
from collections import Counter
from sklearn.mixture import GaussianMixture

import jsonschema
import tensorflow as tf
import h5py
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import warnings
from pyDOE2 import lhs
from scipy.stats.distributions import norm


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None,
                 text_mode = False,
                 influence_mode = False,
                 image_mode = False):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.text_mode = text_mode
        self.influence_mode = influence_mode
        self.image_mode = image_mode

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels, testing=False, alpha=0.05):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        if not testing:
            alphas, _, coefs = lars_path(x_vector,
                                         weighted_labels,
                                         method='lasso',
                                         verbose=False,
                                         alpha=alpha)
            return alphas, coefs
        else:
            alphas, _, coefs, test_result = lars_path(x_vector,
                                                           weighted_labels,
                                                           method='lasso',
                                                           verbose=False,
                                                           testing=testing)
            return alphas, coefs, test_result   

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                 labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method, testing=False, alpha=0.05):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            if not testing:
                weighted_data = ((data - np.average(data, axis=0, weights=weights))
                                 * np.sqrt(weights[:, np.newaxis]))
                weighted_labels = ((labels - np.average(labels, weights=weights))
                                   * np.sqrt(weights))

                nonzero = range(weighted_data.shape[1])
                _, coefs = self.generate_lars_path(weighted_data,
                                                   weighted_labels)
                for i in range(len(coefs.T) - 1, 0, -1):
                    nonzero = coefs.T[i].nonzero()[0]
                    if len(nonzero) <= num_features:
                        break
                used_features = nonzero
                return used_features
            else:
                weighted_data = ((data - np.average(data, axis=0, weights=weights))
                                 * np.sqrt(weights[:, np.newaxis]))
                weighted_labels = ((labels - np.average(labels, weights=weights))
                                   * np.sqrt(weights))

                nonzero = range(weighted_data.shape[1])
                alphas, coefs, test_result = self.generate_lars_path(weighted_data,
                                                                     weighted_labels, 
                                                                     testing=True,
                                                                     alpha=alpha)
                for i in range(len(coefs.T) - 1, 0, -1):
                    nonzero = coefs.T[i].nonzero()[0]
                    if len(nonzero) <= num_features:
                        break
                used_features = nonzero
                return used_features, test_result
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        
        if model_regressor == 'Bay_non_info_prior':
            #all default args
            model_regressor=BayesianRidge(fit_intercept=True,
                                         n_iter=1000, tol=0.0001,
                                         verbose=True,
                                         alpha_1=1e-06, alpha_2=1e-06, 
                                         lambda_1=1e-06, lambda_2=1e-06, 
                                         alpha_init=None, lambda_init=None)
            print('using Bay_non_info_prior option for model regressor')       

        print(used_features)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))        
        if self.verbose:
            print('Intercept: ', easy_model.intercept_)
            print('Prediction_local: ', local_pred,)
            print('Right: ', neighborhood_labels[0, label])
        print(sorted(zip(used_features, easy_model.coef_)))
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
                        
    def testing_explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='lasso_path',
                                   model_regressor=None,
                                   alpha=0.05):
        """Takes perturbed data, labels and distances, returns explanation. 
            This is a helper function for slime.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()
            alpha: significance level of hypothesis testing.

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features, test_result = self.feature_selection(neighborhood_data,
                                                                    labels_column,
                                                                    weights,
                                                                    num_features,
                                                                    feature_selection,
                                                                    testing=True,
                                                                    alpha=alpha)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred, used_features, test_result)

    def if_explain_instance_with_data(self,
                                   bbox_model,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   num_samples,
                                   feature_selection='auto',
                                   model_regressor=None):

        self.text_mode = False
        self.influence_mode = False
        self.image_mode = False

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        
        if self.text_mode:
            y = []
            for probs in neighborhood_labels:
                if probs[0] > probs[1]:
                    y.append(0)
                else:
                    y.append(1)
        elif self.image_mode:
            y = []
            for row in neighborhood_labels:
                index_max = max(range(len(row)), key=row.__getitem__)
                y.append(index_max)
        else:
            y = bbox_model.predict(neighborhood_data)            
        cols = []
        for i in range(neighborhood_data.shape[1]):
            cols.append(i)
        df = pd.DataFrame(neighborhood_data)
        df1 = df.copy()
        df['target'] = y

        X_train, X_va, y_train, y_va = train_test_split(neighborhood_data, y, random_state=104, test_size=0.20, shuffle=True)
        sigmoid_k = 10
        C = 0.1
        sample_ratio = 0.6        
        num_tr_sample = X_train.shape[0]

        clf = LogisticRegression()
        model = clf.fit(X_train,y_train)
        y_pred = model.predict(X_va)

        GMM = True
        RND = False

        if GMM:
            no_clusters = len(neighborhood_labels[0])
            gmm = GaussianMixture( no_clusters, covariance_type='tied', max_iter=10000)
            gmm.fit(X_train)

            cluster_mean = pd.DataFrame(data={"Cluster": gmm.predict(X_train), 
                                                "Mean Target Variable (y)": y_train 
                                            }).groupby("Cluster").mean().sort_values(by='Mean Target Variable (y)', ascending=False)
            bottom_clusters = cluster_mean.sort_values(by='Mean Target Variable (y)', ascending=False).index[:no_clusters]
            samples, clusters = gmm.sample(num_samples)
            bottom_clusters_filter = [np.any([cluster == x for x in bottom_clusters[0:-1]]) for cluster in clusters]
            samples_to_keep = samples[bottom_clusters_filter]
            clusters_to_keep = clusters[bottom_clusters_filter]
            neighborhood_data_resampled = np.concatenate((neighborhood_data, samples_to_keep), axis=0)
            y_resampled = np.append(y, clusters_to_keep)
            print("Saving x and y values for tSNE plots.")
            np.save('x_data_bc.npy',neighborhood_data_resampled)
            np.save('y_data_bc.npy', y_resampled)

        if RND:
            ros = RandomOverSampler(random_state=0)
            neighborhood_data_resampled, y_resampled = ros.fit_resample(neighborhood_data, y)
            y_resampled = np.array(y_resampled)

        X_train, X_va, y_train, y_va = train_test_split(neighborhood_data_resampled, y_resampled, random_state=104, test_size=0.20, shuffle=True)
        oversample_model = clf.fit(X_train,y_train)
        y_pred = oversample_model.predict(X_va)

        if self.influence_mode:
            y_va_pred = clf.predict_proba(X_va)[:,1]
            weight_ar = clf.coef_.flatten()
            test_grad_loss_val = grad_logloss_theta_lr(y_va,y_va_pred,X_va,weight_ar,C,False,0.1/(num_tr_sample*C))
            tr_pred = clf.predict_proba(X_train)[:,1]
            batch_size = 64

            M = None
            total_batch = int(np.ceil(num_tr_sample / float(batch_size)))
            for idx in range(total_batch):
                batch_tr_grad = batch_grad_logloss_lr(y_train[idx*batch_size:(idx+1)*batch_size],
                    tr_pred[idx*batch_size:(idx+1)*batch_size],
                    X_train[idx*batch_size:(idx+1)*batch_size],
                    weight_ar,
                    C,
                    False,
                    1.0)

                sum_grad = batch_tr_grad.multiply(X_train[idx*batch_size:(idx+1)*batch_size]).sum(0)
                if M is None:
                    M = sum_grad
                else:
                    M = M + sum_grad

            M = M + 0.1/(num_tr_sample*C) * np.ones(X_train.shape[1])
            M = np.array(M).flatten()
            iv_hvp = inverse_hvp_lr_newtonCG(X_train,y_train,tr_pred,test_grad_loss_val,C,True,1e-5,True,M,0.1/(num_tr_sample*C))
            total_batch = int(np.ceil(X_train.shape[0] / float(batch_size)))
            predicted_loss_diff = []

            for idx in range(total_batch):
                train_grad_loss_val = batch_grad_logloss_lr(y_train[idx*batch_size:(idx+1)*batch_size],
                    tr_pred[idx*batch_size:(idx+1)*batch_size],
                    X_train[idx*batch_size:(idx+1)*batch_size],
                    weight_ar,
                    C,
                    False,
                    1.0)
                predicted_loss_diff.extend(np.array(train_grad_loss_val.dot(iv_hvp)).flatten())

            predicted_loss_diffs = np.asarray(predicted_loss_diff)        
            phi_ar = - predicted_loss_diffs
            IF_interval = phi_ar.max() - phi_ar.min()
            a_param = sigmoid_k / IF_interval
            prob_pi = 1 / (1 + np.exp(a_param * phi_ar))
            pos_idx = select_from_one_class(y_train,prob_pi,1,sample_ratio)
            neg_idx = select_from_one_class(y_train,prob_pi,0,sample_ratio)
            sb_idx = np.union1d(pos_idx,neg_idx)
            sb_x_train = X_train[sb_idx]
            sb_y_train = y_train[sb_idx]

            if_model = clf.fit(sb_x_train,sb_y_train)
            y_va_pred = if_model.predict(X_va)
            
        distances_new = sklearn.metrics.pairwise_distances(
                neighborhood_data_resampled,
                neighborhood_data[0].reshape(1, -1),
                metric='euclidean'
        ).ravel()
        weights_resampled = self.kernel_fn(distances_new/1000)
        if self.influence_mode:
            count = 0
            for idx in sb_idx:
                if(idx < num_samples):
                    count += 1
            neighborhood_data_resampled = neighborhood_data_resampled[sb_idx]
            y_resampled = y_resampled[sb_idx]
            weights_resampled = weights_resampled[sb_idx]
            fin_model = if_model

        else:
            fin_model = oversample_model

        if self.text_mode:
            surrogate_model = LogisticRegression()
            surrogate_model.fit(neighborhood_data, y)
            yss = surrogate_model.predict_proba(neighborhood_data_resampled)
            labels_column_resampled = yss[:, label]
            fin_model = surrogate_model

        elif self.image_mode:
            surrogate_model = LogisticRegression()
            surrogate_model.fit(neighborhood_data, y)
            yss = surrogate_model.predict_proba(neighborhood_data_resampled)
            y_ = yss
            if label in y:
                labels_column_resampled = yss[:, 0]
                y_ = np.delete(y_, 0, 1)
                print(y_.shape)
            else:
                labels_column_resampled = np.zeros(yss.shape[0])
            fin_model = surrogate_model

        else:
            yss = bbox_model.predict_proba(neighborhood_data_resampled)
            labels_column_resampled = yss[:, label]

        used_features = self.feature_selection(neighborhood_data_resampled,
                                            labels_column_resampled,
                                            weights_resampled,
                                            num_features,
                                            feature_selection)

        if self.image_mode:
            fin_model.fit(neighborhood_data_resampled[:, used_features], y_resampled)        

        else:
            fin_model.fit(neighborhood_data_resampled[:, used_features], y_resampled, sample_weight=weights_resampled)
        
        prediction_score = fin_model.score(
                neighborhood_data_resampled[:, used_features],
                y_resampled, sample_weight=weights_resampled)

        man_score = 0
        for row in range(len(labels_column)):
            man_score += np.power((neighborhood_data[:,used_features][row] - np.log(labels_column[row])/(np.log(1-labels_column[row]))),2) * weights[row]

        local_pred = fin_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if self.verbose:
            print('Intercept', fin_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])  

        coefs = []
        for i in range(0, neighborhood_data[:, used_features].shape[1]):
            coefs.append(fin_model.coef_[0][i])
        norm = [float(i)/sum(coefs) for i in coefs]
        print("Norm sum: ", sum(norm))
        merged_list = zip(used_features, norm)
        return (fin_model.intercept_,
                sorted(merged_list),
                prediction_score, local_pred)