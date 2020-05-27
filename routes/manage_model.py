from flask import Blueprint, request, render_template

import os
import pandas as pd
import shutil
from configs.server import MODEL_PATH, LOG_FILE

manage_model = Blueprint('manage_model', __name__)

@manage_model.route('/manage_model', methods=['GET', 'POST'])
def showpage():
    # check if model database is exist
    if not os.path.exists(os.path.join(MODEL_PATH, LOG_FILE)):
        return render_template('model_page.html', display='none')
    # read database
    df = pd.read_csv(os.path.join(MODEL_PATH, LOG_FILE))
    df = df[df['_is_confirmed'] == True]
    if request.method == 'POST':
        del_idx = [idx for idx in df['ID'].values \
                   if request.form.get(idx) is not None]

        # delete saved model folder
        for folder in del_idx:
            # shutil.rmtree(os.path.join(MODEL_PATH, folder))                   
            os.remove(os.path.join(MODEL_PATH, '{}.h5'.format(folder)))
        # delete corresponding row in csv file            
        del_idx = df[df['ID'].isin(del_idx)].index
        df.drop(del_idx, axis=0, inplace=True)                
        df.to_csv(os.path.join(MODEL_PATH, LOG_FILE), index=None)

    if len(df) == 0:
        return render_template('model_page.html', display='none')
    # Not display ID and _is_confirmed columns
    cols = list(df.columns)
    cols.pop(-1)
    data_cols = [df[col].values for col in cols]
    data = zip(*data_cols)
    return render_template('model_page.html', cols=cols, data=data)