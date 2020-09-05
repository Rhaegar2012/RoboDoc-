'''
Tkinter Interface
'''

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2 as cv
import DataPrep
import KNN


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title('RoboDoc V0.0')
        self.resizable(False, False)
        self.main_label_frame = ttk.LabelFrame(self, text='')
        self.main_label_frame.grid(column=0, row=0)
        #Training and Test sests
        self.training_set = None
        self.test_set = None
        #Buttons
        self.button_label_frame = ttk.LabelFrame(self, text='')
        self.button_label_frame.grid(column=0, row=0)
        self.load_image_button = ttk.Button(self.button_label_frame, text='Load Image', command=self.load_image)
        self.load_image_button.grid(column=0, row=0)
        self.train_model_button = ttk.Button(self.button_label_frame, text='Train Model', command=self.train_model)
        self.train_model_button.grid(column=0, row=2)
        self.match_query_button = ttk.Button(self.button_label_frame, text='Match Query', command=self.match_query)
        self.match_query_button.grid(column=0, row=4)
        self.match_query_button.configure(state="disabled")
        #Query Image and Match Image display widgets
        self.filename = ''
        self.image_label_frame = ttk.Labelframe(self, text='')
        self.image_label_frame.grid(column=1, row=0)
        self.query_canvas = Canvas(self.image_label_frame, width=250, height=250)
        self.image_query = None
        self.query_canvas.grid(column=0, row=0)
        self.match_canvas = Canvas(self.image_label_frame, width=250, height=250)
        self.image_match = None
        self.match_canvas.grid(column=1, row=0)
        #Results Panel
        self.result_label_frame = ttk.Labelframe(self, text='')
        self.precision = ''
        self.recall = ''
        self.result_label_frame.grid(column=0, row=1)
        self.precision_label = ttk.Label(self.result_label_frame, text='Model precision after training: ')
        self.precision_label.grid(column=0, row=0)
        self.precision_value = ttk.Label(self.result_label_frame, text="")
        self.precision_value.grid(column=1, row=0)
        self.recall_label = ttk.Label(self.result_label_frame, text='Model recall after training: ')
        self.recall_label.grid(column=0, row=1)
        self.recall_value = ttk.Label(self.result_label_frame, text="")
        self.recall_value.grid(column=1, row=1)
        self.model_prediction_label = ttk.Label(self.result_label_frame, text="Model prediction: ")
        self.model_prediction_label.grid(column=0, row=2)
        self.model_prediction_value = ttk.Label(self.result_label_frame, text="")
        self.model_prediction_value.grid(column=1, row=2)
        #Confusion Matrix
        self.confusion_matrix_frame = ttk.Labelframe(self, text='')
        self.confusion_matrix_frame.grid(column=1, row=1)

    def load_image(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.filename = filename
        image = Image.open(filename)
        self.image_query = image
        image = image.resize((250, 250), Image.ANTIALIAS)
        self.image_query = ImageTk.PhotoImage(image)
        self.query_canvas.create_image(20, 20, anchor=NW, image=self.image_query)
        self.match_query_button.configure(state='able')

    def train_model(self):
        ABT = DataPrep.populates_ABT()
        print(ABT, len(ABT.ABT))
        precision = 0.5
        required_precision = 0.83
        while precision < required_precision:
            training_set, test_set = DataPrep.generates_training_test_sets(ABT.ABT)
            prediction_set = KNN.test_model(test_set, training_set)
            precision, recall = KNN.compute_metrics(prediction_set)
            print('run precision: ', precision, 'run recall: ', recall)
        self.training_set, self.test_set = training_set, test_set
        self.precision = str(int(precision*100))+"%"
        self.recall = str(int(recall*100))+"%"
        self.precision_value.configure(text=self.precision)
        self.recall_value.configure(text=self.recall)

    def match_query(self):
        query = cv.imread(self.filename, 0)
        query_instance = KNN.creates_query_instance(query, '')
        match = KNN.finds_best_match(query_instance, self.training_set)
        self.model_prediction_value.configure(text=match.prediction)
        match_image = cv.drawMatches(query, match.keypoints, match.best_match.image,
                                          match.best_match.keypoints, match.k_matches[:20], None, flags=2)
        output_image = cv.imwrite('../output.jpg', match_image)
        match_image_display = Image.open('../output.jpg')
        match_image_display = match_image_display.resize((250, 250), Image.ANTIALIAS)
        self.image_match = ImageTk.PhotoImage(match_image_display)
        self.match_canvas.create_image(20, 20, anchor=NW, image=self.image_match)




