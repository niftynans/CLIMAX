"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from slime_lm._least_angle import lars_path
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import recall_score, precision_score

import sys
sys.path.append('../..')
from grad_utils import grad_logloss_theta_lr
from grad_utils import batch_grad_logloss_lr
from inverse_hvp import inverse_hvp_lr_newtonCG
from dataset import select_from_one_class
from collections import Counter
import dice_ml
from dice_ml.utils import helpers
import jsonschema
import tensorflow as tf
import h5py

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
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

                # Xscaler = sklearn.preprocessing.StandardScaler()
                # Xscaler.fit(weighted_data)
                # weighted_data = Xscaler.transform(weighted_data)
             
                # Yscaler = sklearn.preprocessing.StandardScaler()
                # Yscaler.fit(weighted_labels.reshape(-1, 1))
                # weighted_labels = Yscaler.transform(weighted_labels.reshape(-1, 1)).ravel()

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
        
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        y = (np.rint(labels_column)).astype(int)
        
        print(Counter(y))
        cols = []
        for i in range(neighborhood_data.shape[1]):
            cols.append(i)
        df = pd.DataFrame(neighborhood_data)
        df1 = df.copy()
        df['target'] = y
        zero_counts = 0
        one_counts = 0
        for elem in y:
            if elem == 0:
                zero_counts += 1
            else:
                one_counts += 1
        if zero_counts == num_samples or one_counts == num_samples:
                    flip_ratio = 0.1
                    idxs = np.arange(len(y))
                    np.random.shuffle(idxs)
                    num_flip = int(flip_ratio * len(idxs))
                    y[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y[idxs[:num_flip]]).astype(int)
                    print(Counter(y))
        X_train, X_va, y_train, y_va = train_test_split(neighborhood_data, y, random_state=104, test_size=0.20, shuffle=True)
        sigmoid_k = 10
        C = 0.1
        sample_ratio = 0.6        
        num_tr_sample = X_train.shape[0]
        clf = LogisticRegression(C = C, fit_intercept=False, tol = 1e-8, solver="liblinear", multi_class="ovr", max_iter=100, warm_start=False, verbose=0)

        # Counterfactual Generation Time
        d = dice_ml.Data(dataframe = df, continuous_features = cols, outcome_name = 'target')
        model = clf.fit(X_train,y_train)
        y_pred = model.predict(X_va)
        print("Recall Macro Score: ", recall_score(y_va, y_pred, average='macro'))
        print("Precision Macro Score: ", precision_score(y_va, y_pred, average='macro'))
        print("Roc-Auc Macro Score: ", roc_auc_score(y_va, y_pred, average='macro'))

        # df_cf = pd.DataFrame()
        # m = dice_ml.Model(model=model, backend="sklearn")
        # exp = dice_ml.Dice(d, m, method="random")
        # e1 = exp.generate_counterfactuals(df1.loc[[0]], total_CFs=int(np.abs(zero_counts-one_counts)/10), desired_class="opposite")
        # temp_df = e1.cf_examples_list[0].final_cfs_df
        # df_cf = df_cf.append(temp_df)
        # print(df_cf.values)
        #print(df_cf.values[0][:-1])
        # print(df_cf.values[0][-1])
        

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

        clf.fit(sb_x_train,sb_y_train)
        y_va_pred = clf.predict_proba(X_va)[:,1]

        neighborhood_data = neighborhood_data[sb_idx]
        labels_column = labels_column[sb_idx]
        weights = weights[sb_idx]

        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        
        easy_model = clf
        y = (np.rint(labels_column)).astype(int)

        easy_model.fit(neighborhood_data[:, used_features],
                       y, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            y, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])        
        merged_list = [(used_features[i], easy_model.coef_[0][i]) for i in range(0, len(used_features))]
        return (easy_model.intercept_,
                sorted(merged_list),
                prediction_score, local_pred)


    def if_testing_explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='lasso_path',
                                   model_regressor=None,
                                   alpha=0.05):

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        y = np.round_(labels_column)

        X_train, X_test, y_train, y_test = train_test_split(neighborhood_data, y, random_state=104, test_size=0.20, shuffle=True)
        X_train, X_va, y_train, y_va = train_test_split(X_train, y_train, test_size = 0.2)

        sigmoid_k = 10
        C = 0.1
        sample_ratio = 0.6        
        num_tr_sample = X_train.shape[0]

        flip_ratio = 0.4
        idxs = np.arange(y_train.shape[0])
        np.random.shuffle(idxs)
        num_flip = int(flip_ratio * len(idxs))
        y_train[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y_train[idxs[:num_flip]]).astype(int)


        clf = LogisticRegression(
                C = C,
                fit_intercept=False,
                tol = 1e-8,
                solver="liblinear",
                multi_class="ovr",
                max_iter=100,
                warm_start=False,
                verbose=0,
                )

        clf.fit(X_train,y_train)
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

        clf.fit(sb_x_train,sb_y_train)
        y_va_pred = clf.predict_proba(X_va)[:,1]

        neighborhood_data = neighborhood_data[sb_idx]
        labels_column = labels_column[sb_idx]
        weights = weights[sb_idx]

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
