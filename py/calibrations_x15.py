import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import json
import scipy.io as sio
from   scipy.optimize import curve_fit

from common import *

CMOS_MAX      = 4096
LED_MAX       = 8192 # an automation constraint
LED_MIN       = 16   # an automation constraint

READ_FROM_CSV = True
READ_FROM_MAT = False
SINGLE_LINE   = False

if SINGLE_LINE:
    n_bands = 5
else:       
    n_bands = 15
n_pixels                 = 99
n_leds                   = 6

wl_set                   = [450 , 520 , 550 , 580 , 610, 635 , 660 , 700 , 740 , 770 , 810, 840 , 870 , 900 , 930]
calibration_band_indices = [4,14]
nir_start_index          = 11

track_visible            = [4  , 36 , 72]
track_nir                = [16 , 48 , 90]

max_threshold            = 3750
min_threshold            = 500
global_ceil              = 4096 #verify on tool
global_floor             = 324  #verify on tool

max_led_threshold        = 1024
min_led_threshold        = 256

plot_vis_at              = 36
plot_nir_at              = 48

cmos_checks = [1200,1600,2000,2400]
led_checks  = [300,400,500,600]

""" Sigmoids , parameterized """
sigmoid = lambda x: 1/(1+(np.exp(-1*x)))
def logistic(x , a , b):

    fn = np.zeros([np.shape(x)[0]])

    for p in range(0,np.shape(x)[0]):
        fn[p] = 1/(1 + np.exp(-1*a*x[p] + b))
    
    return fn
""" Sigmoids , parameterized """

""" fit function """
def approximate(channel , leds , pixel_register , PRINT = True , SHOW = True , band_index = -1 , results_path = ''):

    global max_threshold , min_threshold
    global global_ceil   , global_floor
    m = np.shape(channel)[0]
    
    h     = {}
    
    led_mins_abs  = []
    led_maxs_abs  = []
    led_means     = []
    led_stds      = []
    cmos_mins     = []
    cmos_maxs     = []
    led_mins_norm = []
    led_maxs_norm = []
    calib_as      = []
    calib_bs      = []
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    #In the future , change this to map into good LED ranges , derive start and stop indices from there.
    for i in range(0,len(pixel_register)):

        primary_signal = channel[:,pixel_register[i]]

        min_index = -1
        max_index = -1

        start_offset = 32 # choose right - basically the row after which you actually start this cycle.
        stop_offset  = 32

        for j in range(start_offset,m - stop_offset):
            if primary_signal[j] > max_threshold:
                max_index = j 
                break
            
            if j == m - stop_offset - 1:
                max_index = j

        for j in range(start_offset,m - stop_offset):
            if primary_signal[j] > min_threshold:
                min_index = j 
                break

        l = leds[min_index:max_index]
        l_ = l.copy()


        if PRINT:
            print("Pixel                        : " , pixel_register[i])
            print("Offsets                      : " , min_index , max_index)
            print("Led min  , Led max (absolute): " , l[0] , l[-1])
            print("Led mean , std     (absolute): " , np.mean(l_) , " , " , np.std(l_))

        led_mins_abs.append(l[0])
        led_maxs_abs.append(l[-1])
        led_means.append(np.mean(l_))
        led_stds.append(np.std(l_))

        z =  primary_signal[min_index:max_index]
        z_ = z.copy()
        
        if PRINT:
            print("Mins and maxs cmos           :" , np.min(z_) , np.max(z_))
        
        cmos_mins.append(min_threshold) 
        cmos_maxs.append(max_threshold)
         
        #z_ = z_ - global_floor
        #z_ = z_/(global_ceil - global_floor)

        z_ = z_ - min_threshold                     #changed
        z_ = z_/(max_threshold - min_threshold)     #changed
        
        l_ = l_ - np.mean(l_)
        l_ = l_/np.std(l_)

        if PRINT:
            print("Led mins , maxs (normalized) : " , np.min(l_) , np.max(l_))
        
        led_mins_norm.append(np.min(l_))
        led_maxs_norm.append(np.max(l_))
        
        (a_,b_),_ = curve_fit(logistic , l_ , z_)

        if PRINT:
            print("Logistic coeffs: " , a_, " , " , b_)

        calib_as.append(a_)
        calib_bs.append(b_)

        app = logistic(l_, a_, b_)
        
        """plotting observed vs approximated"""
        plt.figure()
        
        plt.title('truth vs a sigmoidal approximation at ' + 'pixel_' + str(pixel_register[i]))
        plt.xlabel('Normalized LED intensities')
        plt.ylabel('Normalized CMOS responses.')
        
        plt.plot(l_ , app , 'b-')
        plt.plot(l_ , z_  , 'r-')
        
        plt.legend(['approximation' , 'true response'])
        plt.savefig(os.path.join(results_path , 'pixel_' + str(pixel_register[i]) + '_band_' + str(band_index)))
        if SHOW:
            plt.show()
        """plotting observed vs appoximated"""



        if PRINT:
            print("")

    #h['led_mins_abs'] = led_mins_abs
    #h['led_maxs_abs'] = led_maxs_abs
    h['led_means']         = led_means
    h['led_stds']          = led_stds
    
    h['cmos_mins']         = cmos_mins
    h['cmos_maxs']         = cmos_maxs
    
    h['led_mins_norm']     = led_mins_norm
    h['led_maxs_norm']     = led_maxs_norm
    
    h['calib_as']          = calib_as
    h['calib_bs']          = calib_bs

    return h
""" fit function """

def plot_auxillary(leds , data_cube , vis = 1 , save_at = 'secondary_plots' , PLOT = False , SAVE = True):
    
    start_offset = 32 # choose right - basically the row after which you actually start the calibration sweep
    stop_offset  = 32

    global max_led_threshold , min_led_threshold , nir_start_index
    global plot_vis_at , plot_nir_at , wl_set
    global cmos_checks , led_checks

    #Plot absolute observed curves , nir and visible

    if not os.path.exists(save_at):
        os.mkdir(save_at)

    m = np.shape(leds)[0]
    for j in range(start_offset,m - stop_offset):
        if leds[j] > max_led_threshold:
            max_index = j
            break
        
        if j == m - stop_offset - 1:
            max_index = j

    for j in range(start_offset,m - stop_offset):
        if leds[j] > min_led_threshold:
            min_index = j 
            break

    x = leds[min_index:max_index]
    y = data_cube[:,min_index:max_index,:]

    if vis:
        plot_legend = []
        
        plt.figure()
        plt.title('Visible WL bands , absolute responses.')
        plt.xlabel('LED Intensity (absolute)')
        plt.ylabel('CMOS count (absolute)')
        for j in range(0,4):
            plt.plot(x , y[j,:,plot_vis_at])
            plot_legend.append(wl_set[j])
        for j in range(0,len(cmos_checks)):
            plt.plot(x,np.ones([np.shape(x)[0]]) * cmos_checks[j])
        for j in range(0,len(led_checks)):
            plt.axvline(led_checks[j])
        plt.legend(plot_legend)
        plt.savefig(os.path.join(save_at,'vis_450_to_580' + '_pxl_' + str(plot_vis_at) + '.pdf'))

        plot_legend = []
        plt.figure()
        plt.title('Visible WL bands , absolute responses.')
        plt.xlabel('LED Intensity (absolute)')
        plt.ylabel('CMOS count (absolute)')
        for j in range(4,8):
            plt.plot(x , y[j,:,plot_vis_at])
            plot_legend.append(wl_set[j])
        for j in range(0,len(cmos_checks)):
            plt.plot(x,np.ones([np.shape(x)[0]]) * cmos_checks[j])
        for j in range(0,len(led_checks)):
            plt.axvline(led_checks[j])
        plt.legend(plot_legend)
        plt.savefig(os.path.join(save_at,'vis_610_to_700' + '_pxl_' + str(plot_vis_at) + '.pdf'))

        plot_legend = []
        plt.figure()
        plt.title('Visible WL bands , absolute responses.')
        plt.xlabel('LED Intensity (absolute)')
        plt.ylabel('CMOS count (absolute)')
        for j in range(7,nir_start_index):
            plt.plot(x , y[j,:,plot_vis_at])
            plot_legend.append(wl_set[j])
        for j in range(0,len(cmos_checks)):
            plt.plot(x,np.ones([np.shape(x)[0]]) * cmos_checks[j])
        for j in range(0,len(led_checks)):
            plt.axvline(led_checks[j])
        plt.legend(plot_legend)
        plt.savefig(os.path.join(save_at,'vis_700_to_810' + '_pxl_' + str(plot_vis_at) + '.pdf'))
    
    else:
        plot_legend = []
        plt.figure()
        plt.title('NIR WL bands , absolute responses.')
        plt.xlabel('LED Intensity (absolute)')
        plt.ylabel('CMOS count (absolute)')
        for j in range(nir_start_index,len(wl_set)):
            plt.plot(x , y[j,:,plot_nir_at])
            plot_legend.append(wl_set[j])
        for j in range(0,len(cmos_checks)):
            plt.plot(x , np.ones([np.shape(x)[0]]) * cmos_checks[j])
        for j in range(0,len(led_checks)):
            plt.axvline(led_checks[j])
        plt.legend(plot_legend)
        plt.savefig(os.path.join(save_at,'nir' + '_pxl_' + str(plot_nir_at) + '.pdf'))

    #Plot spectral and spatial curves at the point where 610 nm takes a value of 1800 at pixel 48
    

    return

if __name__ == '__main__':
    
    T           = '2022629_x15_calib'
    theta_calib = '20'

    root      = os.getcwd()
    root      = os.path.abspath(os.path.join(root,os.pardir))
    parent    = os.path.join(root   , 'x15_calib')
    target    = os.path.join(parent , T)
    save_path = target[:]

    calib_result_store_path = 'x15_calib_results'
    calib_result_store_path = os.path.join(root , calib_result_store_path)
    if not os.path.exists(calib_result_store_path):
        os.mkdir(calib_result_store_path)
    
    calib_result_store_path = os.path.join(calib_result_store_path , T)
    if not os.path.exists(calib_result_store_path):
        os.mkdir(calib_result_store_path)

    if READ_FROM_CSV:

        calibration_targets = glob.glob(os.path.join(target , '*.csv'))


        jaw_angle = calibration_targets[0].split('/')[-1].split('_ja_')[-1].split('_')[0]
        tool      = calibration_targets[0].split('/')[-1].split('_tool_')[-1].split('_line_')[0]

        calib_dict = {}
        calib_dict['bands']         = calibration_band_indices
        calib_dict['track_visible'] = track_visible
        calib_dict['track_nir']     = track_nir

        theta = None

        for calibration_target in calibration_targets:

            name = ((calibration_target.split('/')[-1]).split('_led_')[-1]).split('.')[0]

            theta = ((calibration_target.split('/')[-1]).split('_ja_')[-1]).split('_')[0]

            data_frame = pd.read_csv(calibration_target)
            data       = data_frame.values
            
            leds       = np.max(data[:,0:n_leds] , axis = 1)
            
            bands      = [ data[:,int(n_leds + index*n_pixels):int(n_leds + (index+1)*n_pixels)] for index in range(0,n_bands)] 
            bands      = np.array(bands)


            if name == '135' and theta == theta_calib:
                print('135 hit.')
                calib_results_visible = approximate(bands[calibration_band_indices[0] , : , :] , leds , track_visible , \
                                                    band_index = calibration_band_indices[0],results_path = os.path.join(calib_result_store_path,'primary_ja_' + str(jaw_angle)))
                calib_dict['parameters_visible'] = calib_results_visible     

                plot_auxillary(leds , bands , 1 ,  os.path.join(calib_result_store_path,'auxillary' + '_ja_' + str(jaw_angle)))

            if name == '246' and theta == theta_calib:
                print('246 hit.')
                calib_results_nir     = approximate(bands[calibration_band_indices[1] , : , :] , leds , track_nir ,    \
                                                    band_index = calibration_band_indices[1],results_path = os.path.join(calib_result_store_path , 'primary_ja_' + str(jaw_angle)))
                calib_dict['parameters_nir']     = calib_results_nir

                plot_auxillary(leds , bands , 0 ,  os.path.join(calib_result_store_path,'auxillary') + '_ja_' + str(jaw_angle))


        #make folders and filenames for data save
        date        = T.split('_x')[0]
        theta       = theta_calib
        calib_bands = str(calibration_band_indices[0]+1) + '-' + str(calibration_band_indices[1]+1)
        pixels_vis  = str(track_visible[0]) + '-' + str(track_visible[1]) + '-' + str(track_visible[2])
        pixels_nir  = str(track_nir[0]) + '-' + str(track_nir[1]) + '-' + str(track_nir[2])

        save_name   = 'd_' + date + '_ja_' + theta + '_b_' + \
                      calib_bands + '_pxl_vis_' + pixels_vis + '_pxl_nir_' + pixels_nir + '_tool_' + tool
        save_json_at = os.path.join(calib_result_store_path , save_name + '.json')
        print('Saving json at : ' , save_json_at)
        #make folders and filenames for datasave
        
        #save as json
        content_to_file = json.dumps(calib_dict, cls=NumpyEncoder)
        with open(save_json_at, 'w') as file_pointer:
            json.dump(content_to_file, file_pointer)
        #save as json

        #save as a loadable mat 
        save_mat_at = os.path.join(calib_result_store_path, save_name + '.mat')
        print('Save mat at : ' , save_mat_at)
        sio.savemat(save_mat_at , calib_dict)
        #save as a loadable mat

    if READ_FROM_MAT:
        print('Unimplemented.')

    print('')
    print('At main exit line.')
    print('')