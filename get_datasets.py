from datasets import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from os.path import join

prefix_data = "./data/"

def get_dataset(opt):

    if 'nordland' in opt.dataset.lower():
        dataset = Dataset('nordland', 'nordland_train_d-40_d2-10.db', 'nordland_test_d-1_d2-1.db', 'nordland_val_d-1_d2-1.db', opt)  # train, test, val structs
        if 'sw' in opt.dataset.lower():
            ref, qry = 'summer', 'winter'
        elif 'sf' in opt.dataset.lower():
            ref, qry = 'spring', 'fall'
        ft1 = np.load(join(prefix_data,"descData/{}/nordland-clean-{}.npy".format(opt.descType,ref)))
        ft2 = np.load(join(prefix_data,"descData/{}/nordland-clean-{}.npy".format(opt.descType,qry)))
        trainInds, testInds, valInds = np.arange(15000), np.arange(15100,18100), np.arange(18200,21200)

        dataset.trainInds = [trainInds, trainInds]
        dataset.valInds = [valInds, valInds]
        dataset.testInds = [testInds, testInds]
        encoder_dim = dataset.loadPreComputedDescriptors(ft1,ft2)

    elif 'oxford' in opt.dataset.lower():
        ref, qry = '2015-03-17-11-08-44', '2014-12-16-18-44-24'
        structStr = "{}_{}_{}".format(opt.dataset,ref,qry)
        # note: for now temporarily use ox_test as ox_val
        if 'v1.0' in opt.dataset:
            testStr = '_test_d-25_d2-5.db'
        elif 'pnv' in opt.dataset:
            testStr = '_test_d-25_d2-5.db'
        dataset = Dataset(opt.dataset, structStr+'_train_d-20_d2-5.db', structStr+testStr, structStr+testStr, opt)  # train, test, val structs
        ft1 = np.load(join(prefix_data,"descData/{}/oxford_{}_stereo_left.npy".format(opt.descType,ref)))
        ft2 = np.load(join(prefix_data,"descData/{}/oxford_{}_stereo_left.npy".format(opt.descType,qry)))
        splitInds = np.load("./structFiles/{}_splitInds.npz".format(opt.dataset), allow_pickle=True)

        dataset.trainInds = splitInds['trainInds'].tolist()
        dataset.valInds = splitInds['valInds'].tolist()
        dataset.testInds = splitInds['testInds'].tolist()
        encoder_dim = dataset.loadPreComputedDescriptors(ft1,ft2)

    else:
        raise Exception('Unknown dataset')

    return dataset, encoder_dim


def get_splits(opt, dataset):
    whole_train_set, whole_training_data_loader, train_set, whole_test_set = None, None, None, None
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set,
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                pin_memory=not opt.nocuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training whole set:', len(whole_train_set))
        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() in ['val']:
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)

    return whole_train_set, whole_training_data_loader, train_set, whole_test_set
