"""from dask.distributed import Client, wait
from dask_saturn import SaturnCluster

n_workers = 3
cluster = SaturnCluster(n_workers=n_workers)
client = Client(cluster)
client.wait_for_workers(n_workers)
"""
# Larry and Namrata - I am commenting this script out for some understanding
# As you know dask parallelizes pandas dataframes
# dask_cudf uses dask architectures to to do the same on cuDF dataframes - uses GPU instead of CPU
# The below preamble is if you use a single GPU as a local "cluster" of GPU cores the above preamble is for a
# true cluster of GPUs, this changes the syntax significantly and you have to think of data locality
# meaning that you cannot simply run a dask_cudf script on a local file, to avoid the hassle of naming and assigning
# cluster components, its much easier to just stream the data from S3 straight to the cluster, this requires
# minimal change to the syntax, I used local files in Larry's script temporarily as streaming in over and over again was
# even more run time than it already was.

"""from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster()
client = Client(cluster)
client

#import gzip - using full RRF file but if huge datasets, luckily S3 can stream gz files and dask can read them
#importing dask cudf as pd so we can remember its about the same as pandas for dataframe ops
import dask_cudf as pd
#lol numpy
import numpy as np
#s3fs and boto3 are the two largest libraries for working with S3 data
import s3fs"""
import pandas as pd
if __name__ == '__main__':
    #mrconso_columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS']
    mrconso_columns = ['CUI', 'LAT', 'TS', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPRESS', 'CVF']
    #mrrel_columns = ['CUI1', 'REL', 'CUI2', 'RELA', 'SAB', 'SL', 'MG']
    mrrel_columns = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL','RG','DIR','SUPPRESS','CVF']

    mrconso_iter = pd.read_csv("s3://sagemaker-studio-757570088617-lvhv3fp3v5/MRREL.RRF", iterator=True, sep='|', lineterminator='\n',names=mrconso_columns,index_col=False,encoding = "ISO-8859-1", chunksize=1000)
    mrconso = pd.concat([chunk[chunk['LAT'] == 'ENG'] for chunk in mrconso_iter])
    #mrrel_iter = pd.read_csv('MRREL_test.RRF', iterator=True, sep='|', lineterminator='\n',names=mrrel_columns,index_col=False,encoding = "ISO-8859-1", chunksize=1000)
    #mrrel = pd.concat([chunk[chunk['RELA'] == 'isa'] for chunk in mrrel_iter])

    mrrel.drop_duplicates(subset = "CUI1", inplace = True)
    mrconso.drop_duplicates(subset = "CUI", inplace = True)
    mapper = mrconso.set_index('CUI')['STR']
    """merged = mrrel.copy()
    merged['hyponym'] = merged['CUI1'].map(mapper)
    merged['hypernym'] = merged['CUI2'].map(mapper)
    drop_columns = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL','RG','DIR','SUPPRESS','CVF']
    merged = merged.drop(drop_columns, axis=1)
    merged.to_csv('./sub_output.csv', sep='|', index=False)"""