import scipy.io
import numpy as np
import random

def label_unique_tuples(person_num, trace_num, walk_num, sensor_num=None):
    # Stack the provided arrays. Include sensor_num only if it is not None.
    if sensor_num is not None:
        data = np.column_stack((person_num, trace_num, walk_num, sensor_num))
    else:
        data = np.column_stack((person_num, trace_num, walk_num))
    
    # Find the unique tuples and create a dictionary with their labels
    unique_tuples = np.unique(data, axis=0)
    label_dict = {tuple(tup): label for label, tup in enumerate(unique_tuples)}
    
    # Assign labels to the original data based on the unique tuples
    labels = [label_dict[tuple(tup)] for tup in data]
    
    return labels

    
def feature_extract(person_nums, is_top3 = 1):
    # Load features for all people
    
    ## Labels 
    gts = np.empty((0,2))
    walk_nums = np.empty((0,1))
    trace_nums = np.empty((0,1))
    people_nums = np.empty((0,1))
    sensor_nums = np.empty((0,1))

    # Features
    wlk_fre = np.empty((0,1))
    wlk_fres_trace = np.empty((0,1))
    cwt_figs_all = np.empty((0,46,176))
    cwt_sum_all_0 = np.empty((0,18))
    cwt_sum_all_1 = np.empty((0,18))
    cwt_sum_all_2 = np.empty((0,18))
    cwt_sum_all_3 = np.empty((0,18))
    pitchs = np.empty((0,31))
    auto_corrs = np.empty((0, 351))
    real_hils = np.empty((0, 176))
    imag_hils = np.empty((0,176))
    
    dur_time_1_alls = np.empty((0,1))
    dur_time_2_alls = np.empty((0,1))
    jitters = np.empty((0,1))
    shimmers = np.empty((0,1))
    jitter_rap = np.empty((0,1))
    hrs = np.empty((0,33))
    high_fre_compos = np.empty((0,176))
    low_fre_compos = np.empty((0, 176))
    feature = np.empty((0,10))
    slope = np.empty((0, 4))
    zcrs = np.empty((0,176))
    fft_features = np.empty((0, 89))
    energy_alls = np.empty((0,152))
    log_energy_alls = np.empty((0,152))
    smoothe_energy_alls = np.empty((0,152))
    
    legendres = np.empty((0,101))
    
    
    double_support_time = np.empty((0,1))
    pdps_new = np.empty((0,1))
    
    lpcs = np.empty((0,176))
    ceps_features = np.empty((0,176))
    
    # spes
    spe_centr = np.empty((0,33))
    delta_spe_centr = np.empty((0,33))
    spe_crest = np.empty((0,33))
    delta_spe_crest = np.empty((0,33))
    spe_decrease = np.empty((0,33))
    delta_spe_decrease = np.empty((0,33))
    spe_entropy = np.empty((0,33))
    delta_spe_entropy = np.empty((0,33))
    spe_flatness = np.empty((0,33))
    delta_spe_flatness = np.empty((0,33))
    spe_flux = np.empty((0,33))
    delta_spe_flux = np.empty((0,33))
    spe_kurtosis = np.empty((0,33))
    delta_spe_kurtosis = np.empty((0,33))
    spe_skewness = np.empty((0,33))
    delta_spe_skewness = np.empty((0,33))
    spe_rfp = np.empty((0,33))
    delta_spe_rfp = np.empty((0,33))
    spe_slope = np.empty((0,33))
    delta_spe_slope = np.empty((0,33))
    spe_spread = np.empty((0,33))
    delta_spe_spread = np.empty((0,33))
    
    for person_num in person_nums:
        filename = '../../features_v3_all/p'+str(person_num)+'_features_all.mat'
        mat_data = scipy.io.loadmat(filename)
        
        filename_2 = '../../features_v3_all/p'+str(person_num)+'_features_all_new.mat'
        mat_data_2 = scipy.io.loadmat(filename_2)
        
        # Load Ground Truth
        gts_tmp = mat_data['gts']
        # Load trace_nums, walk_nums
        sensor_nums_tmp = mat_data['sensor_nums']
        walk_nums_tmp = mat_data['walk_nums']
        trace_nums_tmp = mat_data['trace_nums']
        
        wlk_fres_tmp = mat_data['wlk_fres'] #
        wlk_fres_trace_tmp = mat_data['wlk_fres_trace'] #
        dur_time_1_alls_tmp = mat_data['dur_time_1_alls'] #
        dur_time_2_alls_tmp = mat_data['dur_time_2_alls'] #
        jitters_tmp = mat_data['jitters'] #
        shimmers_tmp = mat_data['shimmers'] #
        jitter_rap_tmp = mat_data['jitter_rap'] #
        hrs_tmp = mat_data['hrs'] #
        hrs_tmp = hrs_tmp.reshape(len(gts_tmp),-1)
        high_fre_compos_tmp = mat_data['high_fre_compos'] #
        low_fre_compos_tmp = mat_data['low_fre_compos'] #
        feature_tmp = mat_data['feature']
        slope_tmp = mat_data['slope']
        zcrs_tmp = mat_data['zcrs']
        fft_features_tmp = mat_data['fft_features']
        energy_alls_tmp = mat_data['energy_alls']
        log_energy_alls_tmp = mat_data['log_energy_alls']
        smoothe_energy_alls_tmp = mat_data['smoothe_energy_alls']
        cwt_sum_all_tmp = mat_data['c_sum_all']
        legendres_tmp = mat_data['legendres']
        cwt_figs_tmp = mat_data['cwt_figs']
    
        pitches_tmp = mat_data_2['pitchs']
        double_support_time_tmp = mat_data_2['double_support_new']
        pdps_new_tmp = mat_data_2['pdps_new']
        auto_corrs_tmp = mat_data_2['auto_corrs']
        lpcs_tmp = mat_data_2['lpcs']
        ceps_features_tmp = mat_data_2['ceps_features']
        real_hils_tmp = mat_data_2['real_hils']
        imag_hils_tmp = mat_data_2['imag_hils']
    
        spe_centr_tmp = mat_data['spe_centr']
        spe_centr_tmp = spe_centr_tmp.reshape(len(gts_tmp),-1)
        delta_spe_centr_tmp = mat_data_2['delta_centrs']
        spe_crest_tmp = mat_data['spe_crest']
        spe_crest_tmp = spe_crest_tmp.reshape(len(gts_tmp),-1)
        delta_spe_crest_tmp = mat_data_2['delta_crests']
        spe_decrease_tmp = mat_data['spe_decrease']
        spe_decrease_tmp = spe_decrease_tmp.reshape(len(gts_tmp),-1)
        delta_spe_decrease_tmp = mat_data_2['delta_decreases']
        spe_entropy_tmp = mat_data['spe_entropy']
        spe_entropy_tmp = spe_entropy_tmp.reshape(len(gts_tmp),-1)
        delta_spe_entropy_tmp = mat_data_2['delta_entropys']
        spe_flatness_tmp = mat_data['spe_flatness']
        spe_flatness_tmp = spe_flatness_tmp.reshape(len(gts_tmp),-1)
        delta_spe_flatness_tmp = mat_data_2['delta_flatnesss']
        spe_flux_tmp = mat_data['spe_flux']
        spe_flux_tmp = spe_flux_tmp.reshape(len(gts_tmp),-1)
        delta_spe_flux_tmp = mat_data_2['delta_fluxs']
        spe_kurtosis_tmp = mat_data['spe_kurtosis']
        spe_kurtosis_tmp = spe_kurtosis_tmp.reshape(len(gts_tmp),-1)
        delta_spe_kurtosis_tmp = mat_data_2['delta_kurtosiss']
        spe_skewness_tmp = mat_data['spe_skewness']
        spe_skewness_tmp = spe_skewness_tmp.reshape(len(gts_tmp),-1)
        delta_spe_skewness_tmp = mat_data_2['delta_skewnesss']
        spe_rfp_tmp = mat_data['spe_rfp']
        spe_rfp_tmp = spe_rfp_tmp.reshape(len(gts_tmp),-1)
        delta_spe_rfp_tmp = mat_data_2['delta_rfps']
        spe_slope_tmp = mat_data['spe_slope']
        spe_slope_tmp = spe_slope_tmp.reshape(len(gts_tmp),-1)
        delta_spe_slope_tmp = mat_data_2['delta_slopes']
        spe_spread_tmp = mat_data['spe_spread']
        spe_spread_tmp = spe_spread_tmp.reshape(len(gts_tmp),-1)
        delta_spe_spread_tmp = mat_data_2['delta_spreads']
                    
    
        cwt_figs_new_tmp = np.zeros([cwt_figs_tmp.shape[2],cwt_figs_tmp.shape[0],cwt_figs_tmp.shape[1]])
        for i in range(cwt_figs_tmp.shape[2]):
            cwt_figs_new_tmp[i,:,:] = cwt_figs_tmp[:,:,i]
        cwt_figs_tmp = cwt_figs_new_tmp
        
        cwt_sum_all_0_tmp = cwt_sum_all_tmp[0::4, :]
        cwt_sum_all_1_tmp = cwt_sum_all_tmp[1::4, :]
        cwt_sum_all_2_tmp = cwt_sum_all_tmp[2::4, :]
        cwt_sum_all_3_tmp = cwt_sum_all_tmp[3::4, :]
    
        person_num_tmp = person_num*np.ones((len(gts_tmp),1))
    
        gts = np.vstack((gts, gts_tmp))
        walk_nums = np.vstack((walk_nums, walk_nums_tmp))
        trace_nums = np.vstack((trace_nums, trace_nums_tmp))
        people_nums = np.vstack((people_nums, person_num_tmp))
        sensor_nums = np.vstack((sensor_nums, sensor_nums_tmp))
        
        wlk_fre = np.vstack((wlk_fre, wlk_fres_tmp))
        wlk_fres_trace = np.vstack((wlk_fres_trace, wlk_fres_trace_tmp))
        cwt_sum_all_0 = np.vstack((cwt_sum_all_0, cwt_sum_all_0_tmp))
        cwt_sum_all_1 = np.vstack((cwt_sum_all_1, cwt_sum_all_1_tmp))
        cwt_sum_all_2 = np.vstack((cwt_sum_all_2, cwt_sum_all_2_tmp))
        cwt_sum_all_3 = np.vstack((cwt_sum_all_3, cwt_sum_all_3_tmp))
        cwt_figs_all = np.concatenate((cwt_figs_all, cwt_figs_tmp), axis=0)
        fft_features = np.vstack((fft_features, fft_features_tmp))
        energy_alls = np.vstack((energy_alls, energy_alls_tmp))
        log_energy_alls = np.vstack((log_energy_alls, log_energy_alls_tmp))
        smoothe_energy_alls = np.vstack((smoothe_energy_alls, smoothe_energy_alls_tmp))
    
    
        high_fre_compos = np.vstack((high_fre_compos, high_fre_compos_tmp))
        low_fre_compos = np.vstack((low_fre_compos, low_fre_compos_tmp))
        feature = np.vstack((feature, feature_tmp))
    
        pitchs = np.vstack((pitchs, pitches_tmp))
        auto_corrs = np.vstack((auto_corrs, auto_corrs_tmp))
        real_hils = np.vstack((real_hils, real_hils_tmp))
        imag_hils = np.vstack((imag_hils, imag_hils_tmp))
    
        dur_time_1_alls = np.vstack((dur_time_1_alls, dur_time_1_alls_tmp))
        dur_time_2_alls = np.vstack((dur_time_2_alls, dur_time_2_alls_tmp))
        jitters = np.vstack((jitters, jitters_tmp))
        shimmers = np.vstack((shimmers, shimmers_tmp))
        jitter_rap = np.vstack((jitter_rap, jitter_rap_tmp))
        hrs = np.vstack((hrs, hrs_tmp))
        slope = np.vstack((slope, slope_tmp))
        zcrs_tmp = zcrs_tmp.reshape(len(gts_tmp),-1)
        zcrs = np.vstack((zcrs, zcrs_tmp))
        
        legendres = np.vstack((legendres, legendres_tmp))
        double_support_time = np.vstack((double_support_time, double_support_time_tmp))
        pdps_new = np.vstack((pdps_new, pdps_new_tmp))
        lpcs = np.vstack((lpcs, lpcs_tmp))
        ceps_features = np.vstack((ceps_features, ceps_features_tmp))
        
    
        spe_centr = np.vstack((spe_centr, spe_centr_tmp))
        delta_spe_centr = np.vstack((delta_spe_centr, delta_spe_centr_tmp))
        spe_crest = np.vstack((spe_crest, spe_crest_tmp))
        delta_spe_crest = np.vstack((delta_spe_crest, delta_spe_crest_tmp))
        spe_decrease = np.vstack((spe_decrease, spe_decrease_tmp))
        delta_spe_decrease = np.vstack((delta_spe_decrease, delta_spe_decrease_tmp))
        spe_entropy = np.vstack((spe_entropy, spe_entropy_tmp))
        delta_spe_entropy = np.vstack((delta_spe_entropy, delta_spe_entropy_tmp))
        spe_flatness = np.vstack((spe_flatness, spe_flatness_tmp))
        delta_spe_flatness = np.vstack((delta_spe_flatness, delta_spe_flatness_tmp))
        spe_flux = np.vstack((spe_flux, spe_flux_tmp))
        delta_spe_flux = np.vstack((delta_spe_flux, delta_spe_flux_tmp))
        spe_kurtosis = np.vstack((spe_kurtosis, spe_kurtosis_tmp))
        delta_spe_kurtosis = np.vstack((delta_spe_kurtosis, delta_spe_kurtosis_tmp))
        spe_skewness = np.vstack((spe_skewness, spe_skewness_tmp))
        delta_spe_skewness = np.vstack((delta_spe_skewness, delta_spe_skewness_tmp))
        spe_rfp = np.vstack((spe_rfp, spe_rfp_tmp))
        delta_spe_rfp = np.vstack((delta_spe_rfp, delta_spe_rfp_tmp))
        spe_slope = np.vstack((spe_slope, spe_slope_tmp))
        delta_spe_slope = np.vstack((delta_spe_slope, delta_spe_slope_tmp))
        spe_spread = np.vstack((spe_spread, spe_spread_tmp))
        delta_spe_spread = np.vstack((delta_spe_spread, delta_spe_spread_tmp))

    walk_nums_all = np.squeeze(walk_nums)
    trace_nums_all = np.squeeze(trace_nums)
    people_nums_all = np.squeeze(people_nums)
    if is_top3 ==1:
        trace_wlk_num = label_unique_tuples(people_nums_all, walk_nums_all, trace_nums_all, sensor_nums)
        trace_wlk_num = np.array(trace_wlk_num)
        trace_walks = trace_wlk_num
        # Get the index of the largest 3 steps of a trace
        t_unique = np.unique(trace_walks)
        energy_ind = np.zeros(len(trace_walks), dtype=int)
        
        for i in t_unique:
            t_idx = np.where(trace_walks == i)[0]
            tmp_energy = energy_alls[t_idx, :]
            tmp_s = np.sum(tmp_energy, axis=1)
            ind_sort = np.argsort(tmp_s)
            energy_ind[t_idx[ind_sort[-3:]]] = 1
        
        # Calculate the score only for the selected index
        idx_max = np.where(energy_ind == 1)[0]
    
    
        gts = gts[idx_max,:]
        sensor_nums = sensor_nums[idx_max,:]
        walk_nums = walk_nums[idx_max,:]
        trace_nums = trace_nums[idx_max,:]
        people_nums = people_nums[idx_max,:]
        
        
        spe_centr = spe_centr[idx_max,:]
        delta_spe_centr = delta_spe_centr[idx_max,:]
        spe_crest = spe_crest[idx_max,:]
        delta_spe_crest = delta_spe_crest[idx_max,:]
        spe_decrease = spe_decrease[idx_max,:]
        delta_spe_decrease = delta_spe_decrease[idx_max,:]
        spe_entropy = spe_entropy[idx_max,:]
        delta_spe_entropy = delta_spe_entropy[idx_max,:]
        spe_flatness = spe_flatness[idx_max,:]
        delta_spe_flatness = delta_spe_flatness[idx_max,:]
        spe_flux = spe_flux[idx_max,:]
        delta_spe_flux = delta_spe_flux[idx_max,:]
        spe_kurtosis = spe_kurtosis[idx_max,:]
        delta_spe_kurtosis = delta_spe_kurtosis[idx_max,:]
        spe_skewness = spe_skewness[idx_max,:]
        delta_spe_skewness = delta_spe_skewness[idx_max,:]
        spe_rfp = spe_rfp[idx_max,:]
        delta_spe_rfp = delta_spe_rfp[idx_max,:]
        spe_slope = spe_slope[idx_max,:]
        delta_spe_slope = delta_spe_slope[idx_max,:]
        spe_spread = spe_spread[idx_max,:]
        delta_spe_spread = delta_spe_spread[idx_max,:]
        wlk_fre = wlk_fre[idx_max,:]
        wlk_fres_trace = wlk_fres_trace[idx_max,:]
        cwt_figs_all = cwt_figs_all[idx_max,:,:]
        cwt_sum_all_0 = cwt_sum_all_0[idx_max,:]
        cwt_sum_all_1 = cwt_sum_all_1[idx_max,:]
        cwt_sum_all_2 = cwt_sum_all_2[idx_max,:]
        cwt_sum_all_3 = cwt_sum_all_3[idx_max,:]
        high_fre_compos = high_fre_compos[idx_max,:]
        pitchs = pitchs[idx_max,:]
        low_fre_compos = low_fre_compos[idx_max,:]
        auto_corrs = auto_corrs[idx_max,:]
        real_hils = real_hils[idx_max,:]
        imag_hils = imag_hils[idx_max,:]
        dur_time_1_alls = dur_time_1_alls[idx_max,:]
        dur_time_2_alls = dur_time_2_alls[idx_max,:]
        jitters = jitters[idx_max,:]
        shimmers = shimmers[idx_max,:]
        jitter_rap = jitter_rap[idx_max,:]
        hrs = hrs[idx_max,:]
        feature = feature[idx_max,:]
        slope = slope[idx_max,:]
        zcrs = zcrs[idx_max,:]
        fft_features = fft_features[idx_max,:]
        energy_alls = energy_alls[idx_max,:]
        log_energy_alls = log_energy_alls[idx_max,:]
        smoothe_energy_alls = smoothe_energy_alls[idx_max,:]
        legendres = legendres[idx_max,:]
        double_support_time = double_support_time[idx_max,:]
        pdps_new = pdps_new[idx_max,:]
        lpcs = lpcs[idx_max,:]
        ceps_features = ceps_features[idx_max,:]
    return gts, sensor_nums, walk_nums, trace_nums, people_nums, spe_centr, delta_spe_centr, spe_crest, delta_spe_crest, spe_decrease, delta_spe_decrease, spe_entropy, delta_spe_entropy, spe_flatness, delta_spe_flatness, spe_flux, delta_spe_flux, spe_kurtosis, delta_spe_kurtosis, spe_skewness, delta_spe_skewness, spe_rfp, delta_spe_rfp, spe_slope, delta_spe_slope, spe_spread, delta_spe_spread, wlk_fre, wlk_fres_trace, cwt_figs_all, cwt_sum_all_0, cwt_sum_all_1, cwt_sum_all_2, cwt_sum_all_3, high_fre_compos, pitchs, low_fre_compos, auto_corrs, real_hils, imag_hils, dur_time_1_alls, dur_time_2_alls, jitters, shimmers, jitter_rap, hrs, feature, slope, zcrs, fft_features, energy_alls, log_energy_alls, smoothe_energy_alls, legendres, double_support_time, pdps_new, lpcs, ceps_features

def split_data(walk_nums_all, trace_nums_all, people_nums_all, rand_seed = 42):
    trace_wlk_num = label_unique_tuples(people_nums_all, trace_nums_all, walk_nums_all)
    u = np.unique(trace_wlk_num)
    u1 = u
    np.random.seed(rand_seed)
    np.random.shuffle(u1)
    # Calculate the sizes of each split
    total_length = len(u1)
    split_1_length = int(total_length * 0.8)  # 80% of the total length
    split_2_length = int(total_length * 0.1)  # 10% of the total length
    
    # Split the array
    train_trace = u1[:split_1_length]
    val_trace = u1[split_1_length:split_1_length + split_2_length]
    test_trace = u1[split_1_length + split_2_length:]
    
    ## the flag aray indicating train / val/ test, 0:train 1:val 2:test
    flag_tr_val_te = np.zeros((len(trace_wlk_num),))
    for i in train_trace:
      idx = np.where(trace_wlk_num == i)
      flag_tr_val_te[idx] = 0
    for i in val_trace:
      idx = np.where(trace_wlk_num == i)
      flag_tr_val_te[idx] = 1
    for i in test_trace:
      idx = np.where(trace_wlk_num == i)
      flag_tr_val_te[idx] = 2

    return flag_tr_val_te

def split_data_unknown(test_person_id, walk_nums_all, trace_nums_all, people_nums_all, person_nums, rand_seed = 42):

    test_idx = np.empty((0,1))
    for test_person_id_i in test_person_id:
      tmp_test_idx = np.where(people_nums_all == test_person_id_i)[0]
      tmp_test_idx = tmp_test_idx[:,np.newaxis]
      test_idx = np.vstack((test_idx, tmp_test_idx))
    train_val_person_id = np.setdiff1d(person_nums, test_person_id)
    train_val_idx = np.empty((0,1))
    for train_val_person_id_i in train_val_person_id:
      tmp_test_idx = np.where(people_nums_all == train_val_person_id_i)[0]
      tmp_test_idx = tmp_test_idx[:,np.newaxis]
      train_val_idx = np.vstack((train_val_idx, tmp_test_idx))
    
    test_idx = np.squeeze(test_idx)
    train_val_idx = np.squeeze(train_val_idx)
    test_idx = test_idx.astype(int)
    train_val_idx = train_val_idx.astype(int)
    walk_nums_train_val = np.squeeze(walk_nums_all[train_val_idx])
    trace_nums_train_val = np.squeeze(trace_nums_all[train_val_idx])
    people_nums_train_val = np.squeeze(people_nums_all[train_val_idx])
   
    trace_wlk_num = label_unique_tuples(people_nums_all, trace_nums_all, walk_nums_all)
    # trace_wlk_num = label_unique_tuples(people_nums_all, walk_nums_all, trace_nums_all)
    u = np.unique(trace_wlk_num)
    u1 = u
    np.random.seed(rand_seed)
    np.random.shuffle(u1)
    # Calculate the sizes of each split
    total_length = len(u1)
    split_1_length = int(total_length * 0.9)  # 80% of the total length
    split_2_length = int(total_length * 0.1)  # 10% of the total length
    
    # Split the array
    train_trace = u1[:split_1_length]
    val_trace = u1[split_1_length:]
    
    ## the flag aray indicating train / val/ test, 0:train 1:val 2:test
    flag_tr_val_te = np.zeros((len(walk_nums_all),))
    for i in train_trace:
      idx = np.where(trace_wlk_num == i)
      flag_tr_val_te[train_val_idx[idx]] = 0
    for i in val_trace:
      idx = np.where(trace_wlk_num == i)
      flag_tr_val_te[train_val_idx[idx]] = 1
    for i in test_idx:
      flag_tr_val_te[i] = 2

    return flag_tr_val_te