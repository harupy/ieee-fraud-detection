import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mapper import device_mapper, browser_mapper, os_mapper


def add_screen_size(df):
  ss = df.pop('id_33')
  ss = ss.str.extract(r'(\d+)x(\d+)', expand=True).rename(columns={0: 'width', 1: 'height'})
  print(ss.columns)
  ss['width'] = ss['width'].astype(float)
  ss['height'] = ss['height'].astype(float)
  ss['aspect_ratio'] = ss['width'] / ss['height']
  return pd.concat((df, ss), axis=1)


def split_domain(df):
  p_emaildomain = df('P_emaildomain')
  r_emaildomain = df('R_emaildomain')
  df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = p_emaildomain.str.split('.', expand=True)
  df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = r_emaildomain.str.split('.', expand=True)


def map_text(text, mapper):
  if not isinstance(text, str): return np.nan

  for k, v in mapper.items():
    if re.search(k, text.lower()):
      return v

  return 'other'


def process_identity(df):
  pairs = [
    ('id_30', os_mapper),
    ('id_31', browser_mapper),
    ('DeviceInfo', device_mapper),
  ]
  for col, mapper in pairs:
    df[col] = df[col].map(lambda text: map_text(text, mapper))


def categorical_encode(df):
  le = LabelEncoder()
  str_cols = df.select_dtypes('object')
  for col in str_cols:
    df[col] = le.fit_transform(df[col])


def load_data():
  df_trans = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
  df_ident = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
  return df_trans.merge(df_ident, how='left', left_index=True, right_index=True)


def main():
  df_ident = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
  df_ident = add_screen_size(df_ident)
  process_identity(df_ident)
  categorical_encode(df_ident)
  print(df_ident.head())


if __name__ == '__main__':
  main()
