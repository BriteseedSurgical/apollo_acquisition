import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

n_bands   = 4
n_control = 6
n_pixels  = 124

MIN_LED   = 16      #adapt constraint
MAX_LED   = 8192    #adapt constraint
PXL_MIN   = 1
PXL_MAX   = 124
CMOS_MAX  = 4096
CMOS_MIN  = 256

#Apollo Git Access A : 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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_dict_xy(data_path, ref = 1600 , show = True , date = 'unspecified'):
    global n_bands,n_control,n_pixels,colours
    dict = {}
    ctr = 0
    targets = glob.glob(data_path + '/*.npy')

    for target in targets:
        
        current  = {}
        local    = target.split('/')[-1]
        
        category  = local.split('_')[0]
        reference = (local.split('mm_')[-1]).split('_')[0]
        pig_id    = (local.split('-')[0])[-1]

        data     = np.load(target)
        leds     = data[0,0:n_control]
        
        cmos     = np.zeros([n_bands,n_pixels])
        for i in range(0,n_bands):
            cmos[i,:] = data[i,n_control:n_control+n_pixels]

        if int(reference) == ref:
            current['class']  = category
            current['leds']   = leds
            current['cmos']   = cmos
            current['date']   = date
            current['pig_id'] = pig_id 

            dict[ctr] = current
            ctr = ctr + 1
    
    dict['length'] = ctr
    print("Files per baseline : ", dict['length'])
    return dict

def json_read_rt_4wl(path_to_file):

    dict_copy = {}
    ctr = 0
    global n_bands,n_control,n_pixels,colours

    file_pointer = open(path_to_file)
    
    content_in_file = json.load(file_pointer)
    local_dict      = json.loads(content_in_file)

    dict_length         = local_dict['length']
    dict_copy['length'] = dict_length
    dict_copy['path']   = path_to_file

    for i in range(0,dict_length):

        d = {}
        sample_dict = local_dict[str(i)]

        leds     = sample_dict['leds']
        category = sample_dict['class']
        cmos     = np.asarray(sample_dict['cmos'])
        date     = sample_dict['date']
        pig_id   = sample_dict['pig_id']

        d['leds']      = leds
        d['cmos']      = cmos
        d['category']  = category
        d['date']      = date
        d['pig_id']    = pig_id

        dict_copy[ctr] = d
        ctr = ctr + 1

    file_pointer.close()
    return dict_copy

def combine_dicts_rt_4wl(dict_x , dict_y = {}):
    union_dict= {}
    if len(dict_y) == 0:
        union_dict = dict_x
    else:

        lx = dict_x['length']
        ly = dict_y['length']

        lu = lx + ly
        union_dict['length'] = lu

        for i in range(0,lu):
            if i < lx:
                element = dict_x[i]
            else:
                element = dict_y[i-lx]
            union_dict[i] = element

    return union_dict

def make_xy_binary(full_dict , leds = False):
    N = full_dict['length']

    n_ups = 0
    for i in range(0,N):
        element = full_dict[i]
        if element['category'] == 'Ureter' or element['category'] == 'Peritoneum':
            n_ups = n_ups + 1

    if leds:
        X = np.zeros([n_ups,4,130])
    else:
        X = np.zeros([n_ups,4,124])
    T = np.zeros([n_ups,2])
    
    ctr_ureter = 0
    ctr_peri   = 0
    index_ctr  = 0

    for i in range(0,N):
        element = full_dict[i]
        if element['category'] == 'Ureter' or element['category'] == 'Peritoneum':
            if element['category'] == 'Ureter':
                T[index_ctr,0] = 1
                ctr_ureter = ctr_ureter + 1
            else:
                T[index_ctr,1] = 1
                ctr_peri = ctr_peri + 1

            if not leds:
                X[index_ctr,:,:] = element['cmos']
            else:
                X[index_ctr,:,6:130] = element['cmos']
                led_stack = np.array([element['leds'],element['leds'],element['leds'],element['leds']])
                X[index_ctr,:,0:6]   = led_stack
            index_ctr = index_ctr + 1
    
    print("Ureters found     : " , ctr_ureter)
    print("Peritoneums found : " , ctr_peri)
    sample_count = ctr_ureter + ctr_peri
    
    return [[X,T] , sample_count]

def process_initial_refine( x_in , normalize = False):

    x_refined = None
    N , channels , length = np.shape(x_in)[0] , np.shape(x_in)[1] , np.shape(x_in)[2]
    global n_control , n_pixels

    x_refined = np.zeros([N,channels,length])
    zero_ctr = 0
    if not normalize:
        for i in range(0,N):
            x_refined[i,:,0:n_control] = x_in[i,:,0:n_control]
            if np.max(x_in[:,:,n_control:n_control+n_pixels]) > 1024:
                x_refined[i,:,n_control:n_control+n_pixels] = x_in[i,:,n_control:n_control+n_pixels]/CMOS_MAX
            else:
                x_refined[i,:,n_control:n_control+n_pixels] = x_in[i,:,n_control:n_control+n_pixels]
    else:
        for i in range(0,N):
            M = np.max(x_in[i , : , n_control:n_control + n_pixels])
            if M == 0:
                zero_ctr = zero_ctr + 1
            else:
                x_refined[ i, : , 0:n_control] = x_in[i, : , 0:n_control]
                x_refined[ i , : , n_control:n_control + n_pixels] = x_in[i , : , n_control:n_control + n_pixels]/M
        
        print('Zeros seen : ' , zero_ctr)
        print('')

    return x_refined

def categorize(t_in):
    
    n_elements = np.shape(t_in)[0]
    categorized = np.zeros([n_elements])
    
    for i in range(0,n_elements):
        
        if t_in[i,0] == 1 and t_in[i,1] == 0:   #ureter
            categorized[i] = 0
        elif t_in[i,1] == 1 and t_in[i,0] == 0: #peritoneum
            categorized[i] = 1
        else:
            _ = 0
    
    return categorized

def compose_dataset(x_processed,t_processed, leds = True , normalize = True , metrics = False):
    
    n_elements = np.shape(x_processed)[0]   
    X = []
    T = []
    
    for i in range(0,n_elements):

        [ex_x,ex_t] = expand_input(x_processed[i,:,:] , t_processed[i], leds = leds , \
                                   normalize = normalize , metrics = metrics)
        X.append(ex_x)
        T.append(ex_t)

    T = np.vstack(T)
    
    return [X,T]

def expand_input(input_element, input_truth , leds = True ,  normalize = True , metrics = False):

    
    global n_pixels,n_control,MAX_LED,PXL_MAX,PXL_MIN
    
    expanded_element = []
    expanded_truth   = None
    
    if leds:
        
        state = input_element[0,0:n_control] #state is now a single dimensional vector.
        
        if normalize:
            expanded_element.append(state/MAX_LED)
        else:
            expanded_element.append(state/MAX_LED)
        #always normalize LEDs

    expanded_cmos = np.expand_dims(input_element[:,n_control:n_control+n_pixels] , axis = 0)
    expanded_element.append(expanded_cmos)

    expanded_truth = input_truth

    return [expanded_element , expanded_truth]