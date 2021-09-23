import os
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from keras.utils import to_categorical


def rec_search(search_dir, fname):
    results = []
    flist = os.listdir(search_dir)
    for f in flist:
        fpath = '{}\\{}'.format(search_dir, f)
        if os.path.isdir(fpath):
            results += rec_search(fpath, fname)
        else:
            if f == fname:
                results.append(fpath)
    return results


def time_tag():
    now = datetime.datetime.now()
    year = str(now.year)[2:]
    month = str(now.month)
    if len(month) == 1:
        month = '0'+month
    day = str(now.day)
    if len(day) == 1:
        day = '0'+day
    hour = str(now.hour)
    if len(hour) == 1:
        hour = '0'+hour
    minute = str(now.minute)
    if len(minute) == 1:
        minute = '0'+minute
    sec = str(now.second)
    if len(sec) == 1:
        sec = '0'+sec
    microsecond = str(now.microsecond)
    time_str = '{}{}{}_{}{}{}_{}'.format(month,day,year,hour,minute,sec,microsecond)
    return time_str


class CNN:
    def __init__(self, model_type = 'classifier', cmode = 'rgb', img_size = 64, size = 'medium',
                 loss_type = 'mean_squared_error', learn_rate = 0.001, final_act = 'sigmoid', opt_type = 'rms', dropout = True, eval_metric = 'accuracy', n_classes = 2):
        self.model_type = model_type; self.learn_rate = learn_rate; self.loss_type = loss_type; self.histories = [];
        self.val_losses = []; self.val_accs = []; self.runtimes = []; self.cmode = cmode; self.img_size = img_size;
        self.trained = False; self.validated = False; self.scores = None; self.epochs = None; self.batch_size = None;
        self.save_name = None;self.final_act = final_act;self.dropout = dropout; self.opt_type = opt_type; self.models = []
        self.metric = eval_metric; self.size = size; self.n_classes = n_classes
        
        if self.cmode == 'rgb':
            self.channels = 3
        else:
            self.channels = 1
            
        return
    
    def train(self, x_train, y_train, x_test, y_test, epochs = 5, batch_size = 32, folds = 1):
        self.epochs = epochs; self.batch_size = batch_size; self.trained = True
        self.acc_list = []; self.loss_list = []; self.folds = folds
        
        x_train = self.loadImages(x_train)
        x_test = self.loadImages(x_test)
        
        if self.model_type == 'classifier':
            if len(np.shape(y_train)) == 1:
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)
        print('\n'+'\t'.join([self.model_type, self.final_act, self.loss_type, str(self.epochs), self.opt_type, str(self.learn_rate)]))
        
        if self.folds <= 0:
            print('Number of folds reset to 1.')
            self.folds = 1
            
        for fold in range(self.folds):
            print('\n{} {}'.format(self.model_type.capitalize(), fold+1))
            print('------------------------------------------')

            st = time.time()
            if self.model_type == 'classifier':
                model = cnn_classifier(self.img_size, n_classes = self.n_classes, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            elif self.model_type == 'regressor':
                model = cnn_regressor(self.img_size, n_classes = self.n_classes, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            self.models.append(model)
            self.histories.append(model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = False))
            if self.model_type == 'regressor':
                self.acc_list.append(self.histories[-1].history['val_acc'])
            else:
                self.acc_list.append(self.histories[-1].history['val_categorical_accuracy'])
            self.loss_list.append(self.histories[-1].history['loss'])
            dt = time.time() - st
            self.runtimes.append(dt)
        self.model = model
        
        if self.folds > 1:
            print('\n{}-Model Summary Stats:'.format(self.folds))
        else:
            print('Training Results:')
        
        print('------------------------------------------')
        print('Accuracy:    {} +/- {}'.format(round(np.mean(self.acc_list),4), round(np.std(self.acc_list),2)))
        print('Loss:        {} +/- {}'.format(round(np.mean(self.loss_list),4), round(np.std(self.loss_list),2)))
        print('Runtime:     {} +/- {}'.format(round(np.mean(self.runtimes),2), round(np.std(self.runtimes),2)))
        self.trained = True
        self.code = codify_model(m_type = self.model_type, color_type = self.cmode, opt_type = self.opt_type,
                            final_act = self.final_act, lr = self.learn_rate)  
        return
    
    def evaluate(self, x_valid, y_valid, plots_on = True):
        x_valid = self.loadImages(x_valid)
        if self.trained:
            for model in self.models:
                if self.model_type == 'classifier':
                    if len(np.shape(y_valid)) == 1:
                        y_valid = to_categorical(y_valid)
                i = 0
                mean_acc_list = np.zeros(self.epochs)
                mean_loss_list = np.zeros(self.epochs)
                
                for history in self.histories:
                    if self.model_type == 'regressor':
                        acc_list = history.history['val_acc']
                    else:
                        acc_list = history.history['val_categorical_accuracy']
    
                    loss_list = history.history['loss']
                    if i >= 1:
                        for j in range(self.epochs):
                            mean_acc_list[j] += acc_list[j]
                            mean_loss_list[j] += loss_list[j]
                    i+=1
                mean_acc_list = np.divide(mean_acc_list, len(mean_acc_list))
                mean_loss_list = np.divide(mean_loss_list, len(mean_loss_list))
                if plots_on:
                    plt.plot(mean_acc_list)
                    plt.plot(mean_loss_list)
                    plt.show()
                    for history in self.histories:
                        plt.figure(figsize = [7,4])
                        plt.plot(history.history['loss'],'r',linewidth = 2.0, linestyle = '--')
                        plt.plot(history.history['val_loss'],'b',linewidth = 2.0, linestyle = '--')
                        if self.model_type == 'classifier':
                            plt.plot(history.history['categorical_accuracy'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_categorical_accuracy'],'b',linewidth = 2.0)
                        elif self.model_type == 'regressor':
                            plt.plot(history.history['acc'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_acc'],'b',linewidth = 2.0)
                        plt.legend(['Training Data', 'Test Data'], fontsize = 12)
                        plt.xlabel('Epochs', fontsize = 16)
                        plt.ylabel('Loss / Acc',fontsize = 16)
                        plt.title('{} {} {} ({}, {}, {})'.format(self.img_size, self.cmode.upper(), self.model_type.capitalize(), self.opt_type, self.final_act, self.learn_rate), fontsize = 16)
                        plt.show()
                self.scores = model.evaluate(x_valid, y_valid, verbose = 1)
                self.val_losses.append(self.scores[0])
                self.val_accs.append(self.scores[1])
            self.validated = True
            
            # Confusion Matrix
            if self.model_type == 'classifier':
                self.val_acc_list = []; self.prec_list = []; self.spec_list = []
                self.sens_list = []; self.cm_list = []
                print('\nCLASSIFICATION ARCHITECTURE')
                print('------------------------------------------')
                y_valid_reduced = list(np.zeros(len(y_valid)))
                for i in range(len(y_valid)):
                    y_valid_reduced[i] = np.argmax(y_valid[i])
                i = 0
                for model in self.models:
                    model_acc = self.acc_list[i]
#                    cm = confusion_matrix(model.predict_classes(x_valid), y_valid_reduced)
#                    tn, fp, fn, tp = cm.ravel()
#                    model_acc = sum(cm.diagonal()) / cm.sum()
#                    model_prec = tp / (tp + fp)
#                    model_sens = tp / (tp + fn)
                    self.val_acc_list.append(model_acc)
#                    self.prec_list.append(model_prec)
#                    self.sens_list.append(model_sens)
#                    self.cm_list.append(cm)
                    i += 1
                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))

            # ROC & AUC Scores
            if self.model_type == 'regressor':
                self.auc_list = []
                print('\nREGRESSION ARCHITECTURE')
                print('------------------------------------------')
                for model in self.models:
                    fpr, tpr, threshold = roc_curve(y_valid, model.predict(x_valid))
                    auc_score = auc(fpr, tpr)
                    if plots_on:
                        plt.figure(figsize = (9,6))
                        plt.xlim(0, 0.2)
                        plt.ylim(0.8, 1)
                        plt.plot(fpr, tpr)
                        plt.plot([0, 1], [0, 1], 'k--', lw = 2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.xlim(left = 0, right = 1)
                        plt.ylim(top = 1, bottom = 0)
                    self.auc_list.append(auc_score)
                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))
                print('AUC:                {} +/- {}'.format(round(np.mean(self.auc_list),4), round(np.std(self.auc_list),2)))
        else:
            print('Model Not Trained!')
        return
    
    def load_images(self, paths):
        img_stack = []
        for path in paths:
            img = Image.open(path)
            img = np.array(img)
            img_stack.append(img)
        img_stack = np.array(img_stack)
        #h,w,d = np.shape(img)
        #img_stack = np.reshape(img_stack, (h,w,d,1))
        return img_stack
    
    def load(self, fname = 'keras_model.h5'):
        self.models = []
        if 'classif' in fname.lower():
            self.model_type = 'classifier'
        if 'regress' in fname.lower():
            self.model_type = 'regressor'
        self.model = load_model(fname)
        self.models.append(self.model)
        print('Model loaded from: ', fname)
        return
    
    def save(self, fname = 'keras_model.h5'):
        self.model.save(fname)
        self.save_name = fname
        self.fsize = os.path.getsize(fname)
        print('Model saved to: {}\nSize:    {} MB'.format(fname, round(self.fsize/1024/1024,2)))
        return
    
    def plot_filters(self, img, img_class = None):
        if img_class != None:
            print(f'Class: {img_class}')
        plt.imshow(img)
        plt.show()
        print('\nConvolution Outputs')
        i = 0
        for layer in self.model.layers:
            if 'Conv2D' in str(layer):
                plotLayer(self.model, i, img, normalize = False)
            i += 1
        return

