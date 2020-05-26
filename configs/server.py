# Define error for folder validation result
FOLDER_ERROR = {0: 'valid', 1: 'not enough class', 2: 'subfolder has no image', 3: 'some error'}
BASE_MODELS = ['Simple', 'Mobilenet', 'Xception']
MODEL_PATH = 'trained_models'
LOCAL_HOST = '127.0.0.1'
PUBLIC_HOST = ''

WEB_SERVER_PORT = 8080
UPLOAD_FOLDER = 'static'

MODEL_LOG = ['ID', 'Name', 'Training date', 'Training start time', \
            'Training stop time', 'Training data', 'Size', 'Accuracy', '_is_confirmed']
LOG_FILE = 'model_database.csv'  
FLAG = 'is_busy.pickle'   
HIST = 'temp_hist.pickle'  
NEED_CONFIRM = 'need_confirm.pickle'  