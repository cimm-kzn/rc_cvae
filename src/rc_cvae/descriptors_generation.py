# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Valentina Afonina <valiaafo@yandex.ru>
#  This file is part of RC_CVAE
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from collections import defaultdict
from itertools import chain, islice, repeat
from CGRtools.files import *
from CGRtools.containers import *
from pickle import load, dump

from CIMtools.preprocessing import Fragmentor
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from os.path import splitext
import numpy as np



class CustomMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, classes, sparse_output=False):
        self.sparse_output = sparse_output
        self.classes = classes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = MultiLabelBinarizer(sparse_output=self.sparse_output, classes=self.classes)
        if isinstance(X, pd.DataFrame):
            return enc.fit_transform(X.iloc[:, 0].values)
        else:
            return enc.fit_transform(X.values)
        

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def cgr_dataset_generation(input_filename, id_field='Transformation_ID'):
    """
    Convert reactions to condensed graph of reactions (CGRs)
    
    Read RDF with reactions and write SDF with CGRs

    Arguments:
        input_filename -- path and name of rdf file with reactions
    """
    with RDFread(input_filename) as input_f, \
         SDFwrite("{filename}_cgr.sdf".format(filename=splitext(input_filename)[0])) as output_f:
             for reaction in input_f:
                 cgr = reaction.compose()
                 cgr.meta[id_field] = reaction.meta[id_field]
                 output_f.write(cgr)


def frg_learning(cgr_input_filename, frg_filename, frg_workpath='./', chunksize=500):

    f = Fragmentor(fragment_type=9, min_length=2, max_length=4, cgr_dynbonds=0, useformalcharge=True, version='2017.x',
                   doallways=False, header=None, workpath=frg_workpath, verbose=False, remove_rare_ratio=0, return_domain=False)
    print('Fragmentation...')
    with SDFread(cgr_input_filename) as cgrfile:
        for num_chunk, chunk in enumerate(chunks(cgrfile, size=chunksize), start=1):
            f.partial_fit(chunk)
            print('Fragmentation chunk No{}'.format(num_chunk))
        f.finalize()
    with open(frg_filename, 'wb') as frg:
        dump(f, frg)
    print('Fitted Fragmentor was saved to file {}'.format(frg_filename))

def ipca_learning(frg_filename, cgr_input_filename, ipca_filename, num_pc):
    with open(frg_filename, 'rb') as frg_file:
        frg = load(frg_file)
        
    ipca = IncrementalPCA(n_components=num_pc, whiten=False, copy=True, batch_size=None)
    
    len_cgr_input_filename = 0
    with SDFread(cgr_input_filename) as cgrfile:
        for cgr in cgrfile:
            len_cgr_input_filename += 1

    for j in range(num_pc, (num_pc+1000), 1):
        if len_cgr_input_filename % j == 0:
            chunk_size = j
            break
    with SDFread(cgr_input_filename) as cgrfile:
        for num_chunk, chunk in enumerate(chunks(cgrfile, size=chunk_size), start=1):
            tmp_obj = list(chunk)
            x_temp = frg.transform(tmp_obj)
            ipca.partial_fit(x_temp.values)
            del x_temp
            del tmp_obj
            print('Incremental PCA chunk (learning) No{} ({} reactions from {})'.format(num_chunk, chunk_size*num_chunk, len_cgr_input_filename))
        print('Explained variance: {0:.3f}'.format(sum(ipca.explained_variance_ratio_)))
        with open('{}_exp_v.txt'.format(splitext(ipca_filename)[0]), 'w') as f:
            f.write(str(sum(ipca.explained_variance_ratio_)))
        with open(ipca_filename, 'wb') as ipca_file:
            dump(ipca, ipca_file)
    print('Fitted IncrementalPCA was saved to file {}'.format(ipca_filename))
        
        
def ipca_descriptors(frg_filename, ipca_filename, cgr_input_filename, filename, id_field='Transformation_ID'):
    with open(frg_filename, 'rb') as frg_file:
        frg = load(frg_file)
    with open(ipca_filename, 'rb') as ipca_file:
        ipca = load(ipca_file)   
    len_cgr_input_filename = 0
    with SDFread(cgr_input_filename) as cgrfile:
        for cgr in cgrfile:
            len_cgr_input_filename += 1
    for j in range(ipca.n_components, (ipca.n_components+1000),1):
        if len_cgr_input_filename % j == 0:
            chunk_size = j
            break
    x_desc = pd.DataFrame()
    with SDFread(cgr_input_filename) as cgrfile:
        for num_chunk, chunk in enumerate(chunks(cgrfile, size=chunk_size), start=1):
            tmp_obj = list(chunk)
            x_temp = frg.transform(tmp_obj)
            x_ipca = ipca.transform(x_temp.values)
            id_r_chunk = [k.meta[id_field] for k in list(tmp_obj)]
            x_temp.index = id_r_chunk
            x_ipca_idr = pd.DataFrame(x_ipca, index=id_r_chunk)
            x_desc = pd.concat([x_desc, x_ipca_idr], axis=0, join="outer", sort=False)
            del x_ipca_idr
            del x_temp
            del x_ipca
            del tmp_obj
            print('Incremental PCA chunk (transformation) No{} ({} reactions from {})'.format(num_chunk, chunk_size*num_chunk, len_cgr_input_filename))

    with open(filename, 'wb') as scal_file:
        dump(x_desc, scal_file, protocol=4)
    return x_desc
        
        
def generate_x_descriptors(input_filename, step, frg_filename, ipca_filename, num_pc, x_filename, id_field='Transformation_ID', chunksize=100):
    cgr_dataset_generation(input_filename=input_filename, id_field=id_field)
    cgr_filename = "{filename}_cgr.sdf".format(filename=splitext(input_filename)[0])
    if step == 'training':
        frg_learning(cgr_input_filename=cgr_filename, 
                    frg_filename=frg_filename, chunksize=chunksize)
        ipca_learning(frg_filename=frg_filename, cgr_input_filename=cgr_filename, ipca_filename=ipca_filename, num_pc=num_pc)
    elif step != 'prediction':
        raise ValueError('Unexpected step value. It should be "training" or "prediction"')
    descriptors = ipca_descriptors(frg_filename=frg_filename, 
                     ipca_filename=ipca_filename, 
                     cgr_input_filename=cgr_filename,
                     filename=x_filename)
    return descriptors
    
def conditions_from_rdf_to_df(input_filename, id_field, condition_id_field):
    data = []
    columns = [id_field, condition_id_field, 'temperature', 'pressure', 'acids_bases_catalytic_poisons', 'catalyst']
    with RDFread(input_filename) as f:
        for r in f:
            rid = r.meta[id_field]
            conditions = defaultdict(dict)
            for key, value in r.meta.items():
                if key != id_field:
                    condition_id = key.split('_')[0]
                    condition_field = key.replace('{}_'.format(condition_id), '', 1)
                    if condition_field in {'acids', 'bases', 'catalytic_poisons'}:
                        conditions[condition_id]['acids_bases_catalytic_poisons'] = value.split('|')
                    elif condition_field == 'catalyst':
                        conditions[condition_id]['catalyst'] = value
                    elif condition_field in {'temperature', 'pressure'}:
                        conditions[condition_id][condition_field] = float(value)
            for condition_id, condition_dict in conditions.items():
                data_record = [rid, condition_id]
                for column in columns[2:4]:
                    data_record.append(condition_dict.get(column, None))
                for column in columns[4:]:
                    data_record.append(condition_dict.get(column, []))
                data.append(data_record)
    df = pd.DataFrame.from_records(data, columns=columns)
    return df.astype({'temperature': 'float64', 'pressure': 'float64'})
    
    
def digitize(X, thresholds=(10, 40), **kwargs):
    """
    _summary_

    _extended_summary_


    Arguments:
        X -- _description_

    Keyword Arguments:
        thresholds -- _description_ (default: {(10, 40)})
        right -- _description_ (default: {False} and thresholds increasing then thresholds[i-1] <= x < thresholds[i])

    Returns:
        _description_
    """
    not_nan_mask = ~np.isnan(X)
    return np.where(not_nan_mask, np.digitize(X, bins=thresholds, right=False), X)


def without_none_column(X):
    return X[:, :-1]

def one_column_reshape(x):
    return x.values.reshape(-1, 1)   

def t_p_columns_names(symbol, thresholds):
    names = []
    for i, threshold in enumerate(thresholds):
        if i == 0:
            names.append('{}<{}'.format(symbol, threshold))
        elif i == (len(thresholds)-1):
            names.append('{}<={}<{}'.format(thresholds[i-1], symbol, threshold))
            names.append('{}>={}'.format(symbol, threshold))
        else:
            names.append('{}<={}<{}'.format(thresholds[i-1], symbol, threshold))
    return names                    

def generate_y(input_filename, y_filename, acids_bases_poisons, catalysts, id_field='Transformation_ID', condition_id_field='Condition_ID',
               t_thresholds =  (10, 40), p_thresholds = (1, 3.5, 100), none_imputer = 1_000_000, drop_duplicates=True
):
    t_categories = list(range(len(t_thresholds)+1)) + [none_imputer]
    p_categories = list(range(len(p_thresholds)+1)) + [none_imputer]

    df = conditions_from_rdf_to_df(input_filename, id_field, condition_id_field)
    
    temperature_transformer = Pipeline(steps=[
        ('binning_by_thresholds', FunctionTransformer(digitize, validate=False, kw_args={'thresholds': t_thresholds})),
        ('imputer', SimpleImputer(strategy='constant', fill_value=none_imputer)),
        ('onehotencoding', OneHotEncoder(categories=[t_categories], sparse=False, dtype=int, handle_unknown='ignore')),
        ('drop_none_column', FunctionTransformer(without_none_column, validate=False))
    ])

    pressure_transformer = Pipeline(steps=[
        ('binning_by_thresholds', FunctionTransformer(digitize, validate=False, kw_args={'thresholds': p_thresholds})),
        ('imputer', SimpleImputer(strategy='constant', fill_value=none_imputer)),
        ('onehotencoding', OneHotEncoder(categories=[p_categories], sparse=False, dtype=int, handle_unknown='ignore')),
        ('drop_none_column', FunctionTransformer(without_none_column, validate=False))
    ])

    acids_bases_poisons_transformer = Pipeline(steps=[
        ('encoding', CustomMultiLabelBinarizer(classes=acids_bases_poisons, sparse_output=False))
    ])

    catalysts_transformer = Pipeline(steps=[
        ('one_column_reshape', FunctionTransformer(one_column_reshape, validate=False)),
        ('onehotencoding', OneHotEncoder(categories=[catalysts], sparse=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('temperature', temperature_transformer, ['temperature']),
        ('pressure', pressure_transformer, ['pressure']),
        ('acids_bases_poisons', acids_bases_poisons_transformer, ['acids_bases_catalytic_poisons']),
        ('catalyst', catalysts_transformer, ['catalyst'])
        
        ], remainder='passthrough')
    

    columns_names = []
    
    for threshold_list, symbol in zip([t_thresholds, p_thresholds], ['t', 'p']):
        columns_names.extend(t_p_columns_names(symbol, threshold_list))
    columns_names = list(chain(columns_names, acids_bases_poisons, catalysts, [id_field, condition_id_field]))

    result = pd.DataFrame(preprocessor.fit_transform(df), columns=columns_names)
    if drop_duplicates:
        result.drop(condition_id_field, axis='columns', inplace=True)
        result.drop_duplicates(inplace=True)
    result.index = result[id_field]
    result.drop(id_field, axis='columns', inplace=True)
    with open(y_filename, 'wb') as f:
        dump(result, f, protocol=4) 
    return result

def load_data(data):
    if isinstance(data, str):
        with open(data, 'rb') as data_file:
            data = load(data_file)
    return data
    

def training_data_preparation(X, y, 
                           x_final_filename, 
                           y_final_filename,
                           number_t_bins=3,
                           number_p_bins=4,
                           number_acids_bases_poisons=131,
                           number_catalysts=227):
    
    
    X = load_data(X)
    y = load_data(y)         
    
    
    X_num_columns = X.shape[1]
    x_y_part = X.join(y)
    X = x_y_part.iloc[:, :X_num_columns] 
    y = x_y_part.iloc[:, X_num_columns:] 
    
    columns = [list(chain(repeat('t', number_t_bins), 
                   repeat('p', number_p_bins),
                   repeat('acids_bases_poisons', number_acids_bases_poisons), 
                   repeat('catalysts', number_catalysts),
                   )),
           list(y.columns)
           ]
    multiindex = pd.MultiIndex.from_tuples(list(zip(*columns)), names=['categories', 'names'])

    y.columns = multiindex
    
    with open(x_final_filename, 'wb') as f:
        dump(X, f, protocol=4)
        
    with open(y_final_filename, 'wb') as f:
        dump(y, f, protocol=4)
    return X, y
        
    
    
if __name__ == "__main__":    
    catalysts = []
    with open('example_data/catalysts_list.txt', 'r') as f:
        for line in f:
            catalysts.append(line.strip())
    acids_bases_poisons = []
    with open('example_data/acids_bases_poisons_list.txt', 'r') as f:
        for line in f:
            acids_bases_poisons.append(line.strip())   
    # generate_x_descriptors(input_filename="example_data/example_hydrogenation_USPTO.rdf",
    #                        step='training',
    #                        frg_filename="fragmentor.pickle", 
    #                        ipca_filename="ipca.pickle", 
    #                        x_filename="x_descriptors.pickle",
    #                        id_field='Transformation_ID',
    #                        chunksize=100,
    #                        num_pc=100)
    generate_y(input_filename="example_data/example_hydrogenation_USPTO.rdf",
                     y_filename='y.pickle', 
                     acids_bases_poisons=acids_bases_poisons, 
                     catalysts=catalysts, 
                     id_field='Transformation_ID', 
                     condition_id_field='Condition_ID',
                     t_thresholds=(10, 40), 
                     p_thresholds=(1, 3.5, 100), 
                     none_imputer=1_000_000, 
                     drop_duplicates=True)
            