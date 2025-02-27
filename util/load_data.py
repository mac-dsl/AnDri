import os

from util.TranAD_base import *
from util.TSB_AD.models.feature import Window
from util.util_overlap import find_length

def load_data_TSB_UAD(dataset, data_type, ratio, d_type, cd1, cd2, character, val, width=None):
# dataset: name of dataset
# data_type: 0: from TSB-UAD, 1: from TranAD
# ratio: training ratio    
    if cd1 > cd2:
        print('Error: CD2', cd2, 'should be bigger than CD1', cd1)
        return -1
    if data_type == 0:
        data, label, name = preprocess_univariate(dataset, max_length=30000)

        # impute simple incremental conecpt drift here
        win = find_length(data)
        data, label, label_cd = add_drift(data, label, d_type, cd1, cd2, character, val, win, width)

        # Prepare data for semisupervised method. 
        # Here, the training ratio = 0.1
        data_train = data[:int(ratio*len(data))]

    elif data_type ==1:
        # Get data from TranAD (single attribute)
        folderpath = './processed/' + dataset + '/'
        name = dataset

        train_d = np.load(os.path.join(folderpath, 'train.npy'))
        test_d = np.load(os.path.join(folderpath, 'test.npy'))
        label_d = np.load(os.path.join(folderpath, 'labels.npy'))       

        data = test_d[:].astype(float)
        data = data.reshape(len(data),)
        label = label_d[:].reshape(len(label_d),)
        # label = label.reshape(len(label),)

        # impute simple incremental conecpt drift here
        data, label, label_cd = add_drift(data, label, d_type, cd1, cd2, character, val, width)

        data_train = train_d[:].astype(float)
        data_train = data_train.reshape(len(data_train),)

    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()

    data_test = data
    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    # X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

    return data, label, label_cd, X_data, X_train, name


def get_data(dataset, model, d_type, character, val, start, end, width=None):
    ######################################################################################
    ## Load data in TSB-UAD (for univariate) or TranAD (for multivariate)
    ## In here, we only load 'univariate' data (In case of TranAD, only load 'single' variable)
    ## width: 0~1
    uni_data = ['Dodgers', 'ECG', 'KDD21', 'MGAB', 'NAB', 'SensorScope', 'YAHOO', 'IOPS', 'NASA-MSL', 'NASA-SMAP']    
    
    if any(k in dataset for k in uni_data):
        data_type = 0 # univariate
    else:
        data_type = 1 # multivariate
    
    if model in ['IForest', 'NORMA', 'MatrixProfile', 'LOF', 'PCA', 'POLY', 'LSTM', 'OCSVM', 'CNN', 'AE']:
        model_type = 0 # TSB-UAD
        data, label, label_cd, X_data, X_train, name = load_data_TSB_UAD(dataset, data_type, 0.1, d_type, start, end, character, val, width)
        return data, label, label_cd, X_data, X_train, name

    elif model in ['TranAD', 'GDN', 'MAD_GAN', 'Attention', 'CAE_M', 'DAGMM', 'OmniAnomaly', 'USAD', 'MTAD_GAT', 'MSCRED']:
        model_type = 1 # TranAD
        if data_type == 0:
            train_loader, test_loader, labels, label_cd, loader, name = load_Uni_dataset(dataset, d_type, start, end, character, val, width)
        elif data_type == 1:
            train_loader, test_loader, labels, label_cd, loader  = load_dataset(dataset, d_type, start, end, character, val, width)
            name = dataset
        
        return train_loader, test_loader, labels, label_cd, loader, name


def training_model(train_loader, test_loader, NN_model, labels, dataset):
    ########################################################################
    ## For training DNNs (in terms of TranAD and related references)
    if NN_model in ['MERLIN']:
        eval(f'run_{NN_model.lower()}(test_loader, labels, dataset)')

    model, optimizer, scheduler, epoch, accuracy_list = load_model(NN_model, labels.shape[1])
    
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
        
    ### Training phase
    print(f'{color.HEADER}Training {NN_model} on {dataset}{color.ENDC}')
    num_epochs = 5; e = epoch + 1; start = time()
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
    save_model(model, optimizer, scheduler, e, accuracy_list)
	# plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
	# plot_accuracies(accuracy_list)
    return model, testD, testO, optimizer, scheduler

def list_chunk(lst, n, l):
    # list_rt = []
    # for i in range(0, len(lst) - n, n):
        ## 1st sample
        # if i == 0:
            # list_rt.append(lst[i:i+n])
        ## the last sample
        # elif i == len(lst) -n :
            # list_rt.append(lst[i-l+1:])
        ## others
        # else:
            # list_rt.append(lst[i-l+1:i+n])
    # return list_rt
    return [lst[i:i+n] for i in range(0, len(lst)-n, n)]

def load_data_path(filename, data_type, character, val, start, end, width, max_len):
    data_org = pd.read_csv(filename, header=None).to_numpy()
    if max_len == None:
        max_len = len(data_org)
    data = data_org[:max_len,0].astype(float)
    label=data_org[:max_len,1]

    win = find_length(data)

    data, label, label_cd = add_drift(data, label, data_type, start, end, character, val, win, width)
    return data, label, label_cd, win