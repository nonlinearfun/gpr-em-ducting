import numpy as np
import pandas

def shuffle(shufflelist):
    """ Returns list of shuffled arrays
        [shufflelist]           list of arrays to be shuffled together
    """
    index_array = np.arange(len(shufflelist[0]))
    np.random.shuffle(index_array)
    s_list = []
    for element in shufflelist:
        s_elem = []
        for value in index_array:
            s_elem.append(element[value])
        s_list.append(np.array(s_elem))
    return s_list

class dataset:
    def __init__(self, args):
        """ Initialize class atributes and call process_data method
            [self]              class instance
            [args]              arguments from parser
        """
        self.x_train, self.y_train = None, None # stores train x and y
        self.y_test = None # stores test y
        self.x_test_clean, self.x_test_naive, self.x_test_MC, self.x_test_IVW = [], [], [], [] # stores test x (list with white & pink noise)
        self.process_data(args)

    def load_dataset(self, args, filename):
        """ Import dataset from csv and format into feature and label np.float arrays; save and load from npz
            [self]              class instance
            [args]              arguments from parser
            [filename]          name of file to import data from
        """
        try:
            dataset = np.load('data/'+filename+'.npz')
        except:
            df = pandas.read_csv('data/'+filename+'.csv')
            data = df.values
            inputs, labels = [], []
            header = list(df)

            # seperate dataframe into inputs and labels
            for ind, val in enumerate(header):
                inp = np.float32(data[:,ind])
                label = np.float32(val[1:].replace('_','.'))
                inputs.append(inp)
                labels.append(label)
            np.savez('data/'+filename+'.npz', X=inputs, y=labels)
            dataset = np.load('data/'+filename+'.npz')

        # from loaded dataset, get inputs and labels
        inputs = dataset['X']
        labels = dataset['y']

        n, d = inputs.shape
        # take out endpoints
        inputs_end = np.concatenate([np.expand_dims(inputs[0], axis=0), np.expand_dims(inputs[n-1], axis=0)], axis=0)
        labels_end = np.array([labels[0], labels[n-1]])
        # dataset without endpoints
        inputs = inputs[1:n-1]
        labels = labels[1:n-1]

        return inputs, labels, inputs_end, labels_end

    def process_data(self, args):
        """ Split dataset into train and test sets; process noise-contaminated data
            [self]              class instance
            [args]              arguments from parser
        """
        # load datasets
        case_str = args.csv
        cleanfile = case_str
        whitefile = case_str+'beta0'
        pinkfile = case_str+'beta1'
        inputs, labels, inputs_end, labels_end = self.load_dataset(args, cleanfile)
        white_inputs, _, _, _ = self.load_dataset(args, whitefile)
        pink_inputs, _, _, _ = self.load_dataset(args, pinkfile)

        # shuffle and calculate train-test split index
        data = shuffle((white_inputs, pink_inputs, inputs, labels))
        ind = np.int64(np.floor((len(labels)+2)*(args.ratio/100)))

        # TRAINING DATA
        self.x_train, self.y_train = data[2][ind:,:], data[3][ind:]
        # put back the endpoints into the training set
        self.x_train = np.concatenate([self.x_train, inputs_end])
        self.y_train = np.concatenate([self.y_train, labels_end])

        # TESTING DATA (CLEAN)
        self.x_test_clean, self.y_test = (data[2][0:ind,:]), data[3][0:ind]

        # TESTING DATA (NOISY)
        if args.noise:
            n, d = inputs.shape
            for i in range(2):
                self.x_test_naive.append(data[i][0:ind,0:d]) # naive approach
                self.x_test_MC.append(data[i][0:ind,:].reshape([1000*ind,d])) # MC approach
                for j in range(len(args.aug_num)):
                    self.x_test_IVW.append(data[i][0:ind,0:args.aug_num[j]*d].reshape([args.aug_num[j]*ind,d])) # inverse variance weighting approach
                    # e.g. white 5, white 10, pink 5, pink 10
