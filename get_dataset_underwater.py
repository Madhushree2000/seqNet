from datasets import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from os.path import join, exists, isdir
from itertools import product
import os
import glob

prefix_data = "./data/"

def get_dataset(opt):

    if 'underwater' in opt.dataset.lower():
        # Define the paths for your underwater dataset
        traj1_path = "/Users/madhushreesannigrahi/Documents/GitHub/seqNet/data/underwater_data/Traj_1_frames"
        traj3_path = "/Users/madhushreesannigrahi/Documents/GitHub/seqNet/data/underwater_data/Traj_3_frames"
        
        # Create database files names for the underwater dataset
        dataset = Dataset('underwater', 'underwater_train_d-20_d2-5.db', 'underwater_test_d-20_d2-5.db', 'underwater_val_d-20_d2-5.db', opt)
        
        # Check if the trajectories exist
        if not all(isdir(path) for path in [traj1_path, traj3_path]):
            raise FileNotFoundError(f"One or more of the trajectory folders don't exist: {traj1_path}, {traj3_path}")
        
        # Set reference and query trajectories
        ref, qry = 'Traj_1', 'Traj_3'
        
        # Load or generate feature descriptors
        # If you have pre-computed descriptors, load them like this:
        ft1_path = join(prefix_data, f"descData/{opt.descType}/underwater-{ref}.npy")
        ft2_path = join(prefix_data, f"descData/{opt.descType}/underwater-{qry}.npy")
        
        # Check if descriptors exist, if not, you need to generate them first
        if not all(exists(path) for path in [ft1_path, ft2_path]):
            print(f"Warning: Pre-computed descriptors not found at {ft1_path} or {ft2_path}")
            print("Using placeholder descriptors for now. Please generate actual descriptors.")
            
            # Count images in each trajectory
            cam0_path1 = join(traj1_path, "cam0")
            cam0_path3 = join(traj3_path, "cam0")
            
            # Count PNG files in each trajectory
            num_images1 = len(glob.glob(join(cam0_path1, "*.png")))
            num_images3 = len(glob.glob(join(cam0_path3, "*.png")))
            
            # Generate placeholder descriptors (you'll need to replace these with actual descriptors)
            desc_dim = 512  # Adjust based on your descriptor type
            ft1 = np.random.rand(num_images1, desc_dim).astype(np.float32)
            ft2 = np.random.rand(num_images3, desc_dim).astype(np.float32)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(ft1_path), exist_ok=True)
            os.makedirs(os.path.dirname(ft2_path), exist_ok=True)
            
            # Save placeholder descriptors
            np.save(ft1_path, ft1)
            np.save(ft2_path, ft2)
        else:
            # Load pre-computed descriptors
            ft1 = np.load(ft1_path)
            ft2 = np.load(ft2_path)
        
        # Define train/val/test splits (adjust these based on your dataset size)
        total_imgs1 = ft1.shape[0]
        total_imgs3 = ft2.shape[0]
        
        # Example split: 70% train, 15% val, 15% test
        train_ratio, val_ratio = 0.7, 0.15
        
        train_size1 = int(total_imgs1 * train_ratio)
        val_size1 = int(total_imgs1 * val_ratio)
        test_size1 = total_imgs1 - train_size1 - val_size1
        
        train_size3 = int(total_imgs3 * train_ratio)
        val_size3 = int(total_imgs3 * val_ratio)
        test_size3 = total_imgs3 - train_size3 - val_size3
        
        # Create indices for each split
        trainInds1 = np.arange(0, train_size1)
        valInds1 = np.arange(train_size1, train_size1 + val_size1)
        testInds1 = np.arange(train_size1 + val_size1, total_imgs1)
        
        trainInds3 = np.arange(0, train_size3)
        valInds3 = np.arange(train_size3, train_size3 + val_size3)
        testInds3 = np.arange(train_size3 + val_size3, total_imgs3)
        
        # Save indices for future use
        split_file = f"./structFiles/underwater_splitInds.npz"
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        np.savez(split_file, 
                 trainInds=[trainInds1, trainInds3], 
                 valInds=[valInds1, valInds3], 
                 testInds=[testInds1, testInds3])
        
        # Set indices for dataset
        dataset.trainInds = [trainInds1, trainInds3]
        dataset.valInds = [valInds1, valInds3]
        dataset.testInds = [testInds1, testInds3]
        
        # Load descriptors into dataset
        encoder_dim = dataset.loadPreComputedDescriptors(ft1, ft2)
        
        # Generate and save sequence bounds if needed
        # This is optional and depends on your dataset structure
        # For now, we'll use simple bounds (one sequence per trajectory)
        seqBounds1 = np.array([[0, total_imgs1 - 1]])
        seqBounds3 = np.array([[0, total_imgs3 - 1]])
        
        sb_file1 = f"./structFiles/seqBoundsFiles/underwater_{ref}_seqBounds.txt"
        sb_file3 = f"./structFiles/seqBoundsFiles/underwater_{qry}_seqBounds.txt"
        
        os.makedirs(os.path.dirname(sb_file1), exist_ok=True)
        np.savetxt(sb_file1, seqBounds1, fmt='%d')
        np.savetxt(sb_file3, seqBounds3, fmt='%d')

    elif 'nordland' in opt.dataset.lower():
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
            testStr = '_test_d-10_d2-5.db'
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

    elif 'msls' in opt.dataset.lower():
        def get_msls_modImgNames(names):
            return ["/".join(n.split("/")[8:]) for n in names]
        trav1, trav2 = 'database', 'query'
        trainCity, valCity = opt.msls_trainCity, opt.msls_valCity
        dbFileName_train = f'msls_{trainCity}_d-20_d2-5.db'
        dbFileName_val = f'msls_{valCity}_d-20_d2-5.db'
        dataset = Dataset('msls', dbFileName_train, dbFileName_val, dbFileName_val, opt)  # train, test, val structs
        ftReadPath = join(prefix_data,"descData/{}/msls_{}_{}.npy")
        seqBounds_all, ft_all = [], []
        for splitCity, trav in product([trainCity, valCity],[trav1, trav2]):
            seqBounds_all.append(np.loadtxt(f"./structFiles/seqBoundsFiles/msls_{splitCity}_{trav}_seqBounds.txt",int))
            ft_all.append(np.load(ftReadPath.format(opt.descType,splitCity,trav)))
        ft_train_ref, ft_train_qry, ft_val_ref, ft_val_qry = ft_all
        sb_train_ref, sb_train_qry, sb_val_ref, sb_val_qry = seqBounds_all
        dataset.trainInds = [np.arange(ft_train_ref.shape[0]),np.arange(ft_train_qry.shape[0])] # append ref & qry
        dataset.valInds = [ft_train_ref.shape[0]+np.arange(ft_val_ref.shape[0]),ft_train_qry.shape[0]+np.arange(ft_val_qry.shape[0])] # shift val by train count
        dataset.testInds = dataset.valInds
        encoder_dim = dataset.loadPreComputedDescriptors(np.vstack([ft_train_ref,ft_val_ref]), np.vstack([ft_train_qry,ft_val_qry]), \
            [np.vstack([sb_train_ref,sb_val_ref]),np.vstack([sb_train_qry,sb_val_qry])])

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


