import matplotlib
from utils import get_predictions
matplotlib.use( 'tkagg' )

detection_threshold = 0.2

IMAGE_PATH = "D:\\pcb_project\\test_images"

#PATH_TO_CKPT = "D:\\TensorflowObjectDetection\\TensorFlow\\workspace\\training_demo-fcis-pre-augumented-data\\exported-models\\11\\saved_model"
PATH_TO_CKPT = "D:\\TensorflowObjectDetection\\TensorFlow\\workspace\\training_demo-fcis-pre-augumented-data\\exported-models\\14\\saved_model"


PATH_TO_LABELS = "D:\\TensorflowObjectDetection\\TensorFlow\\workspace\\training_demo-fcis-pre-augumented-data\\annotations\\data_used_for_training_and_testing_linked_with_excel_result_numbers\\14\\label_map.pbtxt"
#PATH_TO_LABELS = "D:\\TensorflowObjectDetection\\TensorFlow\\workspace\\training_demo-fcis-pre-augumented-data\\annotations\\data_used_for_training_and_testing_linked_with_excel_result_numbers\\11\\label_map.pbtxt"

get_predictions(IMAGE_PATH, PATH_TO_CKPT, PATH_TO_LABELS, detection_threshold)


        
    
    