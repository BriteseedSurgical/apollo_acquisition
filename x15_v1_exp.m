function varargout = x15_v1_exp(varargin)
% AUTO_TRANSMISSION_GUI_V1 MATLAB code for auto_transmission_gui_v1.fig
%      AUTO_TRANSMISSION_GUI_V1, by itself, creates a new AUTO_TRANSMISSION_GUI_V1 or raises the existing
%      singleton*.
%
%      H = AUTO_TRANSMISSION_GUI_V1 returns the handle to a new AUTO_TRANSMISSION_GUI_V1 or the handle to
%      the existing singleton*.
%
%      AUTO_TRANSMISSION_GUI_V1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AUTO_TRANSMISSION_GUI_V1.M with the given input arguments.
%
%      AUTO_TRANSMISSION_GUI_V1('Property','Value',...) creates a new AUTO_TRANSMISSION_GUI_V1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before auto_transmission_gui_v1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to auto_transmission_gui_v1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help auto_transmission_gui_v1

% Last Modified by GUIDE v2.5 21-Jun-2022 09:52:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @auto_transmission_gui_v1_OpeningFcn, ...
                   'gui_OutputFcn',  @auto_transmission_gui_v1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before auto_transmission_gui_v1 is made visible.
 function auto_transmission_gui_v1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to auto_transmission_gui_v1 (see
% VARARGIN)
settings.sys_name = 'root';

% 0 for super pixel, 1 for line-by-line all pixel
settings.operation_mode = 0;
if settings.operation_mode == 1
    settings.var_out = [];
end

evalin('base','clc');
evalin('base','warning off');

nargin = numel(varargin);
if (nargin == 0)
  settings.N_channel = 1;
elseif (nargin == 1)
  if ~isnumeric(varargin{1}) || ~any(varargin{1} == 1:2)
    error('The input argument must be an interger between 1 and 2');
  end
  settings.N_channel = varargin{1};
else
  error('Can only have zero or one input argument.');
end

%% configure_ports

remotePort  = 60000;
remotePort2 = 60002;
localPort   = 60001;
localPort2  = 60003;

% Issue a close in case didn't exit program cleanly during previous instance.
instrobjs = instrfind('Type', 'udp', 'Status', 'open');
for i=1:length(instrobjs)
    fclose(instrobjs(i));
end

serialPort                  = udp('192.168.1.10', remotePort, 'LocalPort', localPort);
serialPort.ReadAsyncMode    = 'continuous';
serialPort.OutputBufferSize = 20;
serialPort.InputBufferSize  = 10110000;

if settings.operation_mode == 0
    serialPort.BytesAvailableFcnCount = 2989;
    settings.bufferSize = serialPort.BytesAvailableFcnCount;
else
    serialPort.BytesAvailableFcnCount = 5005;
    settings.bufferSize = 5005;
end

serialPort.BytesAvailableFcnMode = 'byte';
serialPort.BytesAvailableFcn = {@serial_full_callback, hObject, handles};

% Configure second UDP Ethernet Port for the GUI Overlay
udpPort = udp('192.168.1.20', remotePort2, 'LocalPort', localPort2);
udpPort.OutputBufferSize = 250;
settings.headerSize = 1;

%% codewords

settings.unusual_encounters_ctr = 0;

settings.code_led_update_mode_0 = 'set_led_update_mode 0';
settings.code_led_update_mode_1 = 'set_led_update_mode 1';

settings.code_set_pixel_shift_0 = 'set_pixel_shift 0';
settings.code_set_pixel_shift_1 = 'set_pixel_shift 1';
settings.code_set_pixel_shift_2 = 'set_pixel_shift 2';
settings.code_set_pixel_shift_3 = 'set_pixel_shift 3';
settings.code_set_pixel_shift_4 = 'set_pixel_shift 4';
settings.code_clear_adapt_queue = 'clear_adapt_queue';
settings.code_disp_adapt_queue  = 'disp_adapt_queue';
settings.code_add_to_adapt_queue= 'aq_add_';
settings.code_set_as_pixels     = 'take_pixel_units';
settings.code_set_as_markers    = 'take_marker_units';
settings.code_plot_virtual      = 'plot_virtual';
settings.code_plot_real         = 'plot_real';
settings.code_replay_final_state= 'replay_final';
settings.code_keep_final_state  = 'keep_final';
settings.code_stop_early        = 'stop_early';
settings.code_no_early_stopping = 'stop_at_end';
settings.code_restore_init_conf = 'restore_default_configs';

settings.code_user_verify_spectral   = 'user_verify_spectral_content';
settings.code_user_ignore_spectral   = 'user_ignore_spectral_content';

%% primary_variables
settings.cmos_length    = 4992;

settings.line1_shift    = 3; %set from Oct 07 expts.
settings.line2_shift    = 0; %set from Oct 07 expts.
settings.line3_shift    = 2; %set from Oct 07 expts.
settings.line_shifts    = [settings.line1_shift,settings.line2_shift,settings.line3_shift];

settings.n_leds                 = 6;
settings.n_lines                = 3;
settings.bands_per_line         = 5;
settings.super_pixel_width      = 5;
settings.bytes_per_value        = 2;
settings.data_bytes_per_line    = 990;

if settings.operation_mode == 0
    settings.n_pixels   = 99; %changed today from 124
    settings.meta_bytes = 5;
    settings.jump_step  = settings.bands_per_line * settings.bytes_per_value; %changed today from 2
    settings.meta_end   = settings.headerSize + settings.meta_bytes;
    settings.data_end   = 2989;
else
    settings.n_pixels   = 99;
    settings.jump_step  = settings.bands_per_line * settings.super_pixel_width;
    settings.data_start = 0;
    settings.data_end   = 2475;
end

settings.hsi_imagebuff = zeros(496,1);

settings.line1_data = zeros(settings.bands_per_line,settings.n_pixels);
settings.line2_data = zeros(settings.bands_per_line,settings.n_pixels);
settings.line3_data = zeros(settings.bands_per_line,settings.n_pixels);
settings.all_lines  = zeros(settings.n_lines * settings.bands_per_line , settings.n_pixels);

settings.band_filler = zeros(1,settings.n_pixels);

settings.free_vars = zeros(1,8);

settings.wl_set = [450,520,550,580,610,635,660,700,740,770,810,840,870,900,930];

%Tool specific.
settings.shuffle_order = [14,2,4,15,13,1,11,10,8,5,12,3,6,9,7]; %x15_ud1 : default

settings.line1_legend = {'635', '520', '840', '550','770','Reference'};
settings.line2_legend = {'870', '930', '740', '900','700','Reference'};
settings.line3_legend = {'660', '810', '610', '450','580','Reference'};

%Tool specific. x15_ud1 : default
settings.line1_sources= [1,1,2,1,1];
settings.line1_indices= [6,2,1,3,10];

settings.line2_sources= [2,2,1,2,1];
settings.line2_indices= [2,4,9,3,8];

settings.line3_sources= [1,1,1,1,1];
settings.line3_indices= [7,11,5,1,4];
%Tool specific.

%% calibration_constants

settings.calibration_band_indices     = [5 ,15];
settings.track_visible                = [5 , 36 , 73];
settings.track_nir                    = [17, 49 , 91];

settings.parameters_visible.cmos_mins = [400,400,400];
settings.parameters_visible.cmos_maxs = [3850,3850,3850];
settings.parameters_visible.led_mins  = [-1.73,-1.73,-1.73];
settings.parameters_visible.led_maxs  = [1.73,1.73,1.73];
settings.parameters_visible.led_means = [334,486,768];
settings.parameters_visible.led_stds  = [189,277,440];
settings.parameters_visible.calib_as  = [2.93,3.00,2.39];
settings.parameters_visible.calib_bs  = [1.39,-0.32,-0.69];
settings.parameters_visible.c1        = -1.*(settings.parameters_visible.calib_bs./settings.parameters_visible.calib_as);
settings.parameters_visible.c2        = 1./settings.parameters_visible.calib_as;

settings.parameters_nir.cmos_mins     = [400,400,400];
settings.parameters_nir.cmos_maxs     = [3850,3850,3850];
settings.parameters_nir.led_mins      = [-1.73,-1.73,-1.73];
settings.parameters_nir.led_maxs      = [1.73,1.73,1.73];
settings.parameters_nir.led_means     = [1042,1394,1988];
settings.parameters_nir.led_stds      = [499,799,1144];
settings.parameters_nir.calib_as      = [3.68,2.81,1.37];
settings.parameters_nir.calib_bs      = [-2.93,-1.76,0.39];
settings.parameters_nir.c1            = -1.*(settings.parameters_nir.calib_bs./settings.parameters_nir.calib_as);
settings.parameters_nir.c2            = 1./settings.parameters_nir.calib_as;

settings.pixels_all  = [settings.track_visible(1) ,settings.track_nir(1),settings.track_visible(2)  ...
                        settings.track_nir(2),settings.track_visible(3) ,settings.track_nir(3)];

%% for_automation

settings.global_cmos_max        = 4096;
settings.global_led_max         = 8192;

settings.set_rt                 = 0;
settings.set_to                 = 0;
settings.converged              = 0;
settings.giving_up              = 0;
settings.max_optimization_steps = 400;
settings.optimization_count     = 0;
settings.distance               = [4096,4096,4096,4096,4096,4096];
settings.lp1                    = 4096;
settings.range                  = 2048;
settings.mean_error             = 2048;
settings.plot_distance          = [];
settings.delta                  = 0;
settings.delta1                 = 64;
settings.epsilon1               = 6e-08;
settings.saturation             = [3964,3964,3964,3964,3964,3964];
settings.led_lowers             = [16,16,16,16,16,16];
settings.led_start_state        = [64,64,128,128,16,16];
settings.post_conv_ctr          = 0;
settings.post_conv_length       = 0;
settings.start_optimize         = 0;
settings.update_scale           = 0.01;

settings.y_now                  = [];
settings.X_calib_now            = [];
settings.mean_error_threshold   = 128;

%0 for 1 step automation, 1 for 2 step automation
settings.led_update_mode        = 1;
settings.update_state           = -1;
settings.n_updates              = 0;
settings.n_visible_updates      = 0;
settings.n_nir_updates          = 0;

settings.memory_visible         = zeros(99,1);
settings.memory_nir             = zeros(99,1);
settings.memory_visible_leds    = [16,16,16,16,16,16];
settings.memory_nir_leds        = [16,16,16,16,16,16];

settings.visible_stop_index     = 11;
settings.n_visible_wls          = 11;
settings.n_nir_wls              = 4;

settings.stable_cmos_state      = zeros(settings.n_lines * settings.bands_per_line,settings.n_pixels);
settings.stable_vis_cmos_state  = zeros(settings.n_visible_wls,settings.n_pixels);
settings.stable_nir_cmos_state  = zeros(settings.n_nir_wls , settings.n_pixels);
settings.stable_led_state       = [16,16,16,16,16,16];

settings.adapt_queue            = [];
settings.adapt_end              = 0;

settings.config_index           = 1;
settings.y_config               = zeros(1,6);
settings.y_config(1,:)          = [1,1,1,1,1,1];
settings.n_configs              = 1;

settings.profile_config_list    = [];
settings.n_profiles_loaded      = 0;
settings.exec_config            = 0;
settings.pixels_or_markers      = 0; %pixels is 0
settings.plot_virtual           = 0; %1 plots virtually in adapt runtime.

settings.replay_final           = 0; %1 replays final state.
settings.replay_wait_count      = 2;
settings.replay_counter         = 0;
settings.replay_update_state    = 0; %1 for NIR , 0 for Vis , -1 waits.
settings.replay_led_vis         = [16,16,16,16,16,16];
settings.replay_led_nir         = [16,16,16,16,16,16];
settings.replay_cmos_vis        = zeros(settings.n_visible_wls,settings.n_pixels);
settings.replay_cmos_nir        = zeros(settings.n_nir_wls,settings.n_pixels);

settings.adapt_memory_length    = 8;
settings.adapt_memory_vis_ctr   = 0;
settings.adapt_memory_nir_ctr   = 0;
settings.adapt_memory_vis_leds  = zeros(settings.adapt_memory_length,settings.n_leds); %even # controls assumed
settings.adapt_memory_nir_leds  = zeros(settings.adapt_memory_length,settings.n_leds); %even # controls assumed
settings.adapt_memory_vis_cmos  = zeros(settings.adapt_memory_length,settings.n_pixels);
settings.adapt_memory_nir_cmos  = zeros(settings.adapt_memory_length,settings.n_pixels);

settings.adapt_dev_threshold    = 32;
settings.stop_early             = 0;
settings.monitor_updates_trigger= 48;

settings.user_spectral_verification = 0;

%% rt_data_save

settings.rt_save_parent = 'x15_rt';
if ~exist(settings.rt_save_parent,'dir')
    mkdir(settings.rt_save_parent);
end

settings.Date       = evalin('base','date');
settings.Month      = evalin('base','clock');

rt_columns = 'led1,led2,led3,led4,led5,led6,';
for i = 1:settings.n_pixels * (settings.n_lines * settings.bands_per_line)
    rt_columns = [rt_columns 'cmos_' num2str(i) ','];
end
rt_columns = [rt_columns 'set_to,opt_count,del1,del2,del3,del4,del5,del6,'];
rt_columns = [rt_columns 'set_rt,giving_up,converged,d1,d2,d3,d4,d5,d6,lu_mode,up_state,cy1,cy2,cy3,cy4,cy5,cy6 \n'];
settings.rt_columns = rt_columns;

rt_types = '%d,%d,%d,%d,%d,%d,';
rt_types = [rt_types repmat('%f,',1, settings.n_pixels * (settings.n_lines * settings.bands_per_line) )];
rt_types = [rt_types '%f,%d,%f,%f,%f,%f,%f,%f,'];
rt_types = [rt_types '%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f' '\n'];
settings.rt_types = rt_types;

settings.rt_file_name = '';
settings.ctr1 = 0;

settings.rt_folder_name = strcat(settings.Date(8:end),num2str(settings.Month(1,2)),settings.Date(1:2),'_x15_rt');
settings.rt_folder_name = fullfile(settings.rt_save_parent,settings.rt_folder_name);

%% settling_intensities_data_save
settings.fp_scan = 0;

settings.sample_scan_target = 'today';
settings.sample_scan_parent = 'now';

fpscan_columns   = 'led1,led2,led3,led4,led5,led6,';
for i = 1:settings.n_pixels * settings.n_lines * settings.bands_per_line
    fpscan_columns = [fpscan_columns 'cmos_' num2str(i) ','];
end
fpscan_columns          = [fpscan_columns 'valid\n'];
settings.fpscan_columns = fpscan_columns;

fpscan_types          = '%d,%d,%d,%d,%d,%d,';
fpscan_types          = [fpscan_types repmat('%f,',1,settings.n_pixels * settings.n_lines * settings.bands_per_line)];
fpscan_types          = [fpscan_types '%d' '\n'];
settings.fpscan_types = fpscan_types;

settings.fp_folder_name = 'x15_rt_settling';

settings.fpscan_start_offset = 512;
settings.fpscan_delay        = 32;

settings.start_scan = 0;

%% ua_data_save

settings.ua_save_parent = 'x15_ua';
if ~exist(settings.ua_save_parent,'dir')
    mkdir(settings.ua_save_parent);
end

columns = 'led1,led2,led3,led4,led5,led6,';

if settings.operation_mode == 0
    dataType = '%d,%d,%d,%d,%d,%d,';
    dataType = [dataType repmat('%f,',1,(settings.n_pixels * (settings.n_lines * settings.bands_per_line) )-1)];
    dataType = [dataType '%f' '\n'];
    for i = 1:(settings.n_pixels * (settings.n_lines * settings.bands_per_line) )-1
        columns = [columns 'sp_' num2str(i) ','];
    end
    columns = [columns 'sp_' num2str(settings.n_pixels * settings.n_lines * settings.bands_per_line) '\n'];
else    
    dataType = '%d,%d,%d,%d,%d,%d,';
    dataType = [dataType repmat('%f,',1,2474)];
    dataType = [dataType '%f' '\n'];
    for i = 1:2474
        columns = [columns 'pixel_' num2str(i) ','];
    end
    columns = [columns 'pixel_2475' '\n'];
end

settings.folderName = strcat(settings.Date(8:end),num2str(settings.Month(1,2)),settings.Date(1:2),'_x15_ua');
settings.folderName = fullfile(settings.ua_save_parent,settings.folderName);
settings.fileName = '';

settings.columnNames = columns;
settings.dataType    = dataType;

%% Misc - ua_save_time , ledI stores.
settings.count = 0;
settings.ledI = zeros(1,6);    %Variable to store LEDi
settings.inds = [1,2,3,4,5,6]; % LED indices 
settings.save_data_time = 5;   % in seconds

%% Angle sensor mapping coefficients
% USE THESE for meddux handheld
settings.angle_coeffs = [1.09410084751401e-05,-0.0454385062540506, 48.3045312178481];

%% Axis 0 / Line 1 

h_dc_shadow_0 = findobj('Tag','plot_axis_0');

settings.h_dc_shadow_0 = plot(h_dc_shadow_0,1:settings.n_pixels, ...
    [zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1)],...
     'LineWidth',1);
 
set(h_dc_shadow_0,'XTick',[]);
xlim(h_dc_shadow_0,[1 settings.n_pixels]);
handles.yAxesMinEdit = 256;
handles.yAxesMaxEdit = 4096;

ylim(h_dc_shadow_0,[handles.yAxesMinEdit handles.yAxesMaxEdit]);

for i = 1:length(settings.pixels_all)
    line([settings.pixels_all(i),settings.pixels_all(i)],ylim,'Color','k');
end

legend(h_dc_shadow_0,settings.line1_legend);

%% Axis 1 / Line 2
h_dc_shadow_1 = findobj('Tag','plot_axis_1');

settings.h_dc_shadow_1 = plot(h_dc_shadow_1,1:settings.n_pixels, ...
    [zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1)],...
     'LineWidth',1);
 
set(h_dc_shadow_1,'XTick',[]);
xlim(h_dc_shadow_1,[1 settings.n_pixels]);
handles.yAxesMinEdit = 256;
handles.yAxesMaxEdit = 4096;

ylim(h_dc_shadow_1,[handles.yAxesMinEdit handles.yAxesMaxEdit]);

for i = 1:length(settings.pixels_all)
    line(h_dc_shadow_1,[settings.pixels_all(i),settings.pixels_all(i)],ylim,'Color','k');
end

legend(h_dc_shadow_1,settings.line2_legend);

%% Axis 2 / Line 3

h_dc_shadow_2 = findobj('Tag','plot_axis_2');

settings.h_dc_shadow_2 = plot(h_dc_shadow_2,1:settings.n_pixels, ...
    [zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1), ...
     zeros(settings.n_pixels,1), zeros(settings.n_pixels,1)],...
     'LineWidth',1);
 
set(h_dc_shadow_2,'XTick',[]);
xlim(h_dc_shadow_2,[1 settings.n_pixels]);
handles.yAxesMinEdit = 256;
handles.yAxesMaxEdit = 4096;

ylim(h_dc_shadow_2,[handles.yAxesMinEdit handles.yAxesMaxEdit]);

for i = 1:length(settings.pixels_all)
    line(h_dc_shadow_2 , [settings.pixels_all(i),settings.pixels_all(i)],ylim,'Color','k');
end

legend(h_dc_shadow_2,settings.line3_legend);

[settings.refminloc, settings.refjawangle, settings.refdipmin] = deal([]);

settings.T = cell(5,1);

%% Localization plots 1
settings.color_array_1 = zeros(1,settings.n_pixels,3);
settings.update_localization_plot_1 = 0;
settings.localization_1_indicator_array = zeros(1,settings.n_pixels);

bar = findobj('Tag','localization_plot_0');
set(bar,'Xlim',[1 settings.n_pixels]);
set(bar,'Xtick',0:9:settings.n_pixels);
set(bar,'Ylim',[0 1]);
set(bar,'Ytick',[]);
settings.color_array_1 = zeros(1,settings.n_pixels,3);

%% Localization p[ots 2
settings.color_array_2 = zeros(1,settings.n_pixels,3);
settings.update_localization_plot_2 = 0;
settings.localization_2_indicator_array = zeros(1,settings.n_pixels);

bar = findobj('Tag','localization_plot_1');
set(bar,'Xlim',[1 settings.n_pixels]);
set(bar,'Xtick',0:9:settings.n_pixels);
set(bar,'Ylim',[0 1]);
set(bar,'Ytick',[]);
settings.color_array_2 = zeros(1,settings.n_pixels,3);

%% Opening the serial port 

set(handles.led1_hsi_edit, 'Visible',1);
set(handles.led3_hsi_edit, 'Visible',1);
set(handles.led5_hsi_edit, 'Visible',1);
set(handles.LED2_Edit, 'Visible',1);
set(handles.LED4_Edit, 'Visible',1);
set(handles.LED6_Edit, 'Visible',1);

set(handles.scan_push,'UserData',0);
tempVesselStr = cellstr(get(handles.VesselMenu,'String'));
set(handles.VesselMenu,'UserData',tempVesselStr{1});
tempTissueStr = cellstr(get(handles.TissueMenu,'String'));
set(handles.TissueMenu,'UserData',tempTissueStr{1});

b = instrfind;

try
    for i = 1:length(b)
        if regexpi(b.Status{i},'open')
            fclose(b(i));
        end
    end
catch
end

% Start things rolling.
try
    fopen(serialPort);
catch
    errordlg('Please Reconnect the Board. The MATLAB seems to be having difficulty reading it','Error!');
    close all;
end

% Important that this be asynchronous.  Synchronous read doesn't work
% so well with GUI callbacks and neither does it get the BytesAvailable
% event until the whole buffer is full.
readasync(serialPort);

% Save the serial port in the GUIDATA for later retrieval.
handles.serialport = serialPort;
settings.serialport = serialPort;
handles.troubleshoot = 0;

code=zeros(20,1);
code(1:16) = sprintf('%16s','LED_intensity');
fwrite(handles.serialport,code);

%% Calibration Module

settings.calib_save_parent = 'x15_calib';
if ~exist(settings.calib_save_parent,'dir')
    mkdir(settings.calib_save_parent);
end

settings.max_calib_intensity = 4096;
settings.n_calib_led_sweeps  = 1024;

settings.calibrate   = 0;

settings.calibfolder = [settings.Date(8:end) num2str(settings.Month(1,2)) settings.Date(1:2),'_x15_calib'];
settings.calibfolder = fullfile(settings.calib_save_parent , settings.calibfolder);

if exist(settings.calibfolder,'dir')~=7
    calibfile = 1;
else
    oldFolder = cd(settings.calibfolder);
    calibfile = numel(dir('*.csv'))+1;
    cd(oldFolder);
end
settings.calibfile = calibfile;

% Calibration Parameters
settings.calibLED   = 1;
settings.calibcheck = zeros(5,settings.n_leds);
settings.calibcount = 1;

if settings.operation_mode == 0
    settings.calib_data = zeros(settings.n_calib_led_sweeps,(settings.n_pixels * settings.n_lines * settings.bands_per_line) + settings.n_leds);
else
    settings.calib_data = zeros(settings.n_calib_led_sweeps,settings.n_leds + (settings.n_pixels * settings.bands_per_line)); 
end
set(handles.calibrateButton,'Value',0);
settings.beginCalibrate = 0;
settings.calibwait      = 1;

settings.sp_calib_type    = '%d,%d,%d,%d,%d,%d,';
settings.sp_calib_type    = [settings.sp_calib_type repmat('%f,',1,(settings.n_pixels * settings.bands_per_line) -1)];
settings.sp_calib_type    = [settings.sp_calib_type '%f' '\n'];

settings.sp_calib_columns = 'led1,led2,led3,led4,led5,led6,';

for i = 1:(settings.n_pixels * settings.bands_per_line)-1
    settings.sp_calib_columns = [settings.sp_calib_columns 'sp_' num2str(i) ','];
end

settings.sp_calib_columns = [settings.sp_calib_columns 'sp_' num2str(settings.n_pixels * settings.bands_per_line) '\n'];

%%
set(hObject, 'UserData', settings);
% Choose default command line output for auto_transmission_gui_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% UIWAIT makes auto_transmission_gui_v1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);
% --- Outputs from this function are returned to the command line.
function varargout = auto_transmission_gui_v1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in close_button.
function close_button_Callback(hObject, ~, handles)
% hObject    handle to close_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fclose(handles.serialport);
delete(handles.figure1);

%% Main Function % --- Called when serial port is full.
function serial_full_callback (obj, ~, hSettings, handles)
settings = get(hSettings,'UserData');
Data = fread(obj,settings.bufferSize);
if ~isempty(Data)

    %% Sensor data unpacking
        if size(Data,1) == 2989 || size(Data,1) == 5005
            
            %% Line 1 Read
                        
            if settings.operation_mode == 1
                
                settings.line_shift = 0; % for rt correction /verification (3 , 0 , 2 for x15_ud1)
               
                cmos      = Data(settings.headerSize+1:1:settings.headerSize+4992);
                cmos_msb  = cmos(1:2:4992);
                cmos_msb  = mod(cmos_msb,16);
                cmos_lsb  = cmos(2:2:4992);
                settings.cmos_data = double(uint16(bitshift(cmos_msb,8)) + uint16(cmos_lsb));
                settings.cmos_data = settings.cmos_data(1+settings.line_shift:1:2475+settings.line_shift);
                
                settings.line1_data = zeros(settings.bands_per_line,settings.n_pixels);
                for i = 1:settings.bands_per_line
                    for j = 2:settings.super_pixel_width - 1
                        settings.line1_data(i,:) = settings.line1_data(i,:) + ...
                        settings.cmos_data((settings.bands_per_line*(i-1))+j:settings.jump_step:settings.data_end)';
                    end
                end
                settings.line1_data = settings.line1_data./3;

            else
                
                settings.start_at = settings.meta_end  + 1;
                settings.stop_at  = settings.start_at + settings.data_bytes_per_line - 1;
                read              = Data(settings.start_at : 1 : settings.stop_at);
                
                ctr   = 1;
                for i  = 1 : 2 : settings.bytes_per_value * settings.bands_per_line
                    msbs = read( i     : settings.jump_step : settings.data_bytes_per_line); 
                    lsbs = read( i + 1 : settings.jump_step : settings.data_bytes_per_line);
                    msbs = mod(msbs,16);
                    settings.line1_data(ctr,:) = double(uint16(bitshift(msbs,8)) + uint16(lsbs))';
                    settings.line1_data(ctr,:) = sgolayfilt(medfilt1(settings.line1_data(ctr,:),13),1,21);                    
                    ctr = ctr + 1;
                end
                

            end
            
            %% Line 2 Read

             if settings.operation_mode == 1
                 
                 settings.line2_data = zeros(settings.bands_per_line,settings.n_pixels);
                 settings.free_vars(1,1) = settings.free_vars(1,1) + 1; 

             else
                
                settings.start_at = settings.stop_at  + 1;
                settings.stop_at  = settings.start_at + settings.data_bytes_per_line - 1;
                read              = Data(settings.start_at : 1 : settings.stop_at);
                
                ctr   = 1;
                for i  = 1 : 2 : settings.bytes_per_value * settings.bands_per_line
                    msbs = read( i     : settings.jump_step : settings.data_bytes_per_line); 
                    lsbs = read( i + 1 : settings.jump_step : settings.data_bytes_per_line);
                    msbs = mod(msbs,16);
                    settings.line2_data(ctr,:) = double(uint16(bitshift(msbs,8)) + uint16(lsbs))';
                    settings.line2_data(ctr,:) = sgolayfilt(medfilt1(settings.line2_data(ctr,:),13),1,21);
                    ctr = ctr + 1;
                end
                
             end
                        
            %% Line 3 Read
            
             if settings.operation_mode == 1
                             
                 settings.line3_data = zeros(settings.bands_per_line,settings.n_pixels);
                 settings.free_vars(1,1) = settings.free_vars(1,1) + 1;

             else

                settings.start_at = settings.stop_at  + 1;
                settings.stop_at  = settings.start_at + settings.data_bytes_per_line - 1;
                read              = Data(settings.start_at : 1 : settings.stop_at);
                
                ctr   = 1;
                for i  = 1 : 2 : settings.bytes_per_value * settings.bands_per_line
                    msbs = read( i     : settings.jump_step : settings.data_bytes_per_line); 
                    lsbs = read( i + 1 : settings.jump_step : settings.data_bytes_per_line);
                    msbs = mod(msbs,16);
                    settings.line3_data(ctr,:) = double(uint16(bitshift(msbs,8)) + uint16(lsbs))';
                    settings.line3_data(ctr,:) = sgolayfilt(medfilt1(settings.line3_data(ctr,:),13),1,21);
                    ctr = ctr + 1;
                end
                
             end
                
            settings.all_lines = [settings.line1_data;settings.line2_data;settings.line3_data];
            settings.all_lines = settings.all_lines(settings.shuffle_order,:);
                                    
            %% jaw angle decode
            if settings.operation_mode == 0
                tempangledata = double(uint16(bitshift(Data(settings.headerSize+1),8)) + ...
                        uint16(Data(settings.headerSize+2)));
                jawangle = round(polyval(settings.angle_coeffs,tempangledata));

                settings.theta = jawangle;
                if settings.theta > 60
                    settings.theta = 60;
                end

                if settings.theta < 0
                    settings.theta = 0;
                end
            else
                settings.theta = -1;
            end
            set(handles.JawAngleEdit,'String',num2str(settings.theta));
            
            %% led intensities
            if settings.operation_mode == 0
                ledmsb = Data(settings.stop_at + 2 : 2 : settings.data_end);   
                ledlsb = Data(settings.stop_at + 3 : 2 : settings.data_end);
            else
                ledmsb = Data(4994:2:5005);
                ledlsb = Data(4995:2:5005);
            end
            settings.ledI = double(uint16(bitshift(ledmsb,8)) + uint16(ledlsb))';
            settings.ledI = settings.ledI(settings.inds);
            for i = 1:3
                eval(['set(handles.led' num2str((i*2)-1) '_hsi_edit,''String'',settings.ledI(' num2str((i*2)-1) '));']);
                eval(['set(handles.LED' num2str(i*2) '_Edit,''String'',settings.ledI(' num2str(i*2) '));']);
            end
            
            %% set 

            % initialize , set up rt data save            
            if settings.set_rt == 1 && ~isempty(settings.adapt_queue)
                
                settings.set_to           = settings.adapt_queue(end);
                set(handles.baselineEdit,'String',num2str(settings.set_to));
                
                if settings.config_index == 1
                    disp('About to run baseline : ');
                    disp(settings.set_to);
                end
                
                disp(    'Current shape config  :');
                disp(settings.y_config(settings.config_index,:));
                
                current_config_name = 'other';
                if sum(settings.y_config(settings.config_index,:) == [1,1,1,1,1,1]) == 6
                    current_config_name = 'def';
                end
                
                settings.config_plot = zeros(settings.n_pixels,1);
                for j = 1:3
                    settings.config_plot(settings.track_visible(j),1) = settings.y_config(settings.config_index,2*j-1) .* settings.set_to; 
                    settings.config_plot(settings.track_nir(j)    ,1) = settings.y_config(settings.config_index,2*j)   .* settings.set_to; 
                end
                settings.config_plot = resample(settings.config_plot,99,99);

                vessel     = get(handles.VesselMenu,'UserData');
                vesselsize = get(handles.VesselSizeEdit,'String');
                tissue     = get(handles.TissueMenu,'UserData');
                thickness  = get(handles.ThicknessEdit,'String');
                baseline   = num2str(settings.set_to);
                comment    = get(handles.CommentsEdit,'String');
                tool       = handles.tool_select_pop_up.String{handles.tool_select_pop_up.Value};
                line       = handles.line_select_pop_up.String{handles.line_select_pop_up.Value};
                user       = handles.user_select_pop_up.String{handles.user_select_pop_up.Value};
                orientation= handles.orientation_select_pop_up.String{handles.orientation_select_pop_up.Value};
                approach   = handles.approach_select_pop_up.String{handles.approach_select_pop_up.Value};
              
                pig_1_id        = get(handles.pig_1_id_edit,'String');
                pig_1_sample    = get(handles.sample_1_id_edit,'String');
                pig_1_subsample = get(handles.subsample_1_id_edit,'String');
                
                pig_2_id        = get(handles.pig_2_id_edit,'String');
                pig_2_sample    = get(handles.sample_2_id_edit,'String');
                pig_2_subsample = get(handles.subsample_2_id_edit,'String');
                
                ja_manual  = get(handles.manual_jaw_angle_edit,'String');
                pri_start  = get(handles.primary_start_edit,'String');
                pri_end    = get(handles.primary_end_edit,'String');
                sec_start  = get(handles.secondary_start_edit,'String');
                sec_end    = get(handles.secondary_end_edit,'String');
                
                current_time = strsplit(datestr(datetime('now')));
                current_time = strrep(current_time,':','_');
                current_date = current_time{1};    
                
                p_or_m = num2str(settings.pixels_or_markers);
       
                settings.rt_file_name = strcat('x15_rt','_sys_',settings.sys_name,'_d_',current_date,'_v_',vessel,'_vs_',vesselsize,'_t_',tissue,'_th_', thickness,...
                                                '_b_',baseline, '_c_',comment,'_tool_', tool,'_line_',line,'_user_', user ,...
                                                '_o_',orientation,'_app_',approach,'_ja_',ja_manual,'_s1_',pig_1_id,'-',pig_1_sample,'-',pig_1_subsample,...
                                                '_s2_',pig_2_id,'-',pig_2_sample,'-',pig_2_subsample,...
                                                '_pos1_',pri_start,'-',pri_end,...
                                                '_pos2_',sec_start,'-',sec_end,'_pix_',p_or_m,'_config_',current_config_name,'_clock_',current_time{2},'.csv');

                settings.rt_file_name = fullfile(pwd, settings.rt_folder_name,settings.rt_file_name);
                
                if exist(settings.rt_folder_name,'dir')~=7
                    mkdir(settings.rt_folder_name);
                end

                settings.rt_fid = fopen(settings.rt_file_name,'w');
                fprintf(settings.rt_fid,settings.rt_columns);
                
                settings.start_optimize = 1;
                settings.set_rt         = settings.set_rt + 1;
                settings.converged      = 0;
                settings.giving_up      = 0;
               
                if settings.led_update_mode == 0
                    settings.update_state   = -1; %white or nir : white is 0
                    settings.change = -1;
                else
                    settings.update_state   = 0;
                    settings.change = 0;
                end
                
                settings.n_updates         = 0;
                settings.n_visible_updates = 0;
                settings.n_nir_updates     = 0;
                settings.plot_distance     = [];
                
                settings.delta_visible = [-1024,-1024,-1024];
                settings.delta_nir     = [-1024,-1024,-1024];
                settings.delta         = [-1024,-1024,-1024,-1024,-1024,-1024];
                
                settings.stable_cmos_state      = zeros(settings.n_lines * settings.bands_per_line,settings.n_pixels);
                settings.stable_vis_cmos_state  = zeros(settings.n_visible_wls,settings.n_pixels);
                settings.stable_nir_cmos_state  = zeros(settings.n_nir_wls , settings.n_pixels);
                settings.stable_led_state       = [16,16,16,16,16,16];
                
                settings.replay_counter         = 0;
                settings.replay_update_state    = 0; 
                settings.replay_led_vis         = [16,16,16,16,16,16];
                settings.replay_led_nir         = [16,16,16,16,16,16];
                settings.replay_cmos_vis        = zeros(settings.n_visible_wls,settings.n_pixels);
                settings.replay_cmos_nir        = zeros(settings.n_nir_wls,settings.n_pixels);
                
                settings.adapt_memory_vis_ctr   = 0;
                settings.adapt_memory_nir_ctr   = 0;
                settings.adapt_memory_vis_leds  = zeros(settings.adapt_memory_length,settings.n_leds); %even # controls assumed
                settings.adapt_memory_nir_leds  = zeros(settings.adapt_memory_length,settings.n_leds); %even # controls assumed
                settings.adapt_memory_vis_cmos  = zeros(settings.adapt_memory_length,settings.n_pixels);
                settings.adapt_memory_nir_cmos  = zeros(settings.adapt_memory_length,settings.n_pixels);

            end
            
            % start to optimize
            if settings.set_rt > 1  && settings.post_conv_ctr == 0
                settings.optimization_count = settings.optimization_count + 1;

                %set up automation/led setting
            
                if settings.led_update_mode == 0
                    y_wl = settings.all_lines(settings.calibration_band_indices(1) ,:)';
                    y_ni = settings.all_lines(settings.calibration_band_indices(2) ,:)';
                else
                    if settings.n_updates == 0
                        y_wl = settings.all_lines(settings.calibration_band_indices(1) ,:)';
                        y_ni = settings.all_lines(settings.calibration_band_indices(2) ,:)';
                    elseif settings.n_visible_updates == 1 && settings.n_nir_updates == 0
                        y_wl = settings.memory_visible;
                        y_ni = settings.all_lines(settings.calibration_band_indices(2) , :)';
                    else
                        y_wl = settings.memory_visible;
                        y_ni = settings.memory_nir;
                    end

                end
                                
                %visible current
                settings.y_visible_now          = [y_wl(settings.track_visible(1),1),...
                                                   y_wl(settings.track_visible(2),1),...
                                                   y_wl(settings.track_visible(3),1)];


                p_visible                       = settings.y_visible_now;

                settings.y_visible_now          = min(settings.parameters_visible.cmos_maxs,settings.y_visible_now);
                settings.y_visible_now          = max(settings.parameters_visible.cmos_mins,settings.y_visible_now);
                settings.y_visible_now          = settings.y_visible_now - settings.parameters_visible.cmos_mins;
                settings.y_visible_now          = settings.y_visible_now./(settings.parameters_visible.cmos_maxs - ...
                                                    settings.parameters_visible.cmos_mins);

                x1                              = max(1e-06 , settings.y_visible_now);
                x1                              = min(1-1e-06,x1);
            
                settings.X_calib_visible_now    = settings.parameters_visible.c1 + settings.parameters_visible.c2 .*(log(x1./(1-x1)));                
                settings.X_calib_visible_now    = max(settings.parameters_visible.led_mins , settings.X_calib_visible_now);
                settings.X_calib_visible_now    = min(settings.parameters_visible.led_maxs , settings.X_calib_visible_now);
                settings.X_calib_visible_now    = settings.X_calib_visible_now.*settings.parameters_visible.led_stds + settings.parameters_visible.led_means;      
                %visible current

                %nir current
                settings.y_nir_now              = [y_ni(settings.track_nir(1),1),...
                                                   y_ni(settings.track_nir(2),1),...
                                                   y_ni(settings.track_nir(3),1)];
            
                p_nir                           = settings.y_nir_now;

                settings.y_nir_now              = min(settings.parameters_nir.cmos_maxs,settings.y_nir_now);
                settings.y_nir_now              = max(settings.parameters_nir.cmos_mins,settings.y_nir_now);
                settings.y_nir_now              = settings.y_nir_now - settings.parameters_nir.cmos_mins;
                settings.y_nir_now              = settings.y_nir_now./(settings.parameters_nir.cmos_maxs - ...
                                                    settings.parameters_nir.cmos_mins);

                x2                              = max(1e-06 , settings.y_nir_now);
                x2                              = min(1-1e-06,x2);

                settings.X_calib_nir_now        = settings.parameters_nir.c1 + settings.parameters_nir.c2 .*(log(x2./(1-x2)));                
                settings.X_calib_nir_now        = max(settings.parameters_nir.led_mins , settings.X_calib_nir_now);
                settings.X_calib_nir_now        = min(settings.parameters_nir.led_maxs , settings.X_calib_nir_now);
                settings.X_calib_nir_now        = settings.X_calib_nir_now.*settings.parameters_nir.led_stds + settings.parameters_nir.led_means;      
                %nir current
                
                ydes                        = settings.set_to;
                thr                         = 8192;
                %y                           = [ydes,ydes,ydes,ydes,ydes,ydes];
                %change here
                y                           = settings.y_config(settings.config_index,:);
                y                           = y .* settings.set_to;
                p                           = [p_visible(1),p_nir(1),p_visible(2),p_nir(2),p_visible(3),p_nir(3)];
                
                settings.distance           = y - p;
                settings.range              = max(settings.distance) - min(settings.distance);
                settings.lp1                = sum(abs(settings.distance));
                settings.mean_error         = settings.lp1/6;

                settings.plot_distance      = [settings.plot_distance sum(settings.distance)];

                %vis
                if settings.update_state == -1 || settings.update_state == 0
                    
                    %y_visible               = [ydes,ydes,ydes];
                    %change here
                    %y_visible               = settings.y_visible_config(settings.config_index,:);
                    y_visible               =[settings.y_config(settings.config_index,1),settings.y_config(settings.config_index,3),settings.y_config(settings.config_index,5)];
                    y_visible               = y_visible .* settings.set_to;
                    
                    y_visible               = min(settings.parameters_visible.cmos_maxs,y_visible);
                    y_visible               = max(settings.parameters_visible.cmos_mins,y_visible);
                    y_visible               = y_visible - settings.parameters_visible.cmos_mins;
                    y_visible               = y_visible./(settings.parameters_visible.cmos_maxs - settings.parameters_visible.cmos_mins);
                    x1                      = max(1e-06 , y_visible);
                    x1                      = min(1-1e-06,x1);
                    X_calib_visible_ydes    = settings.parameters_visible.c1 + settings.parameters_visible.c2 .*(log(x1./(1-x1)));                
                    X_calib_visible_ydes    = max(settings.parameters_visible.led_mins , X_calib_visible_ydes);
                    X_calib_visible_ydes    = min(settings.parameters_visible.led_maxs , X_calib_visible_ydes);
                    X_calib_visible_ydes    = X_calib_visible_ydes.*settings.parameters_visible.led_stds + settings.parameters_visible.led_means;

                    settings.delta_visible  = (X_calib_visible_ydes - settings.X_calib_visible_now)./settings.X_calib_visible_now;
                    settings.delta1_visible = sum(settings.delta_visible);
                
                end
                %vis
            
                %nir
                if settings.update_state == -1 || settings.update_state == 1
                    
                    y_nir               = [ydes,ydes,ydes];
                    %change here
                    %y_nir               = settings.y_nir_config(settings.config_index,:);
                    y_nir               =[settings.y_config(settings.config_index,2),settings.y_config(settings.config_index,4),settings.y_config(settings.config_index,6)];
                    y_nir               = y_nir .* settings.set_to;
                    
                    y_nir               = min(settings.parameters_nir.cmos_maxs,y_nir);
                    y_nir               = max(settings.parameters_nir.cmos_mins,y_nir);
                    y_nir               = y_nir - settings.parameters_nir.cmos_mins;
                    y_nir               = y_nir./(settings.parameters_nir.cmos_maxs - settings.parameters_nir.cmos_mins);
                    x2                  = max(1e-06 , y_nir);
                    x2                  = min(1-1e-06,x2);
                    X_calib_nir_ydes    = settings.parameters_nir.c1 + settings.parameters_nir.c2 .*(log(x2./(1-x2)));                
                    X_calib_nir_ydes    = max(settings.parameters_nir.led_mins , X_calib_nir_ydes);
                    X_calib_nir_ydes    = min(settings.parameters_nir.led_maxs , X_calib_nir_ydes);
                    X_calib_nir_ydes    = X_calib_nir_ydes.*settings.parameters_nir.led_stds + settings.parameters_nir.led_means;

                    settings.delta_nir  = (X_calib_nir_ydes - settings.X_calib_nir_now)./settings.X_calib_nir_now;
                    settings.delta1_nir = sum(settings.delta_nir);
                
                end
                %nir
                
                if settings.update_state == -1
                    
                    settings.delta = [settings.delta_visible(1),settings.delta_nir(1),settings.delta_visible(2),...
                                      settings.delta_nir(2),settings.delta_visible(3),settings.delta_nir(3)];
                    settings.delta1= settings.delta1_visible + settings.delta1_nir;
                
                end
                
                if settings.update_state == 0
                    settings.delta  = [settings.delta_visible(1),0,settings.delta_visible(2),0,settings.delta_visible(3),0];
                    settings.delta1 = settings.delta1_visible;
                end
                
                if settings.update_state == 1
                    settings.delta  = [0,settings.delta_nir(1),0,settings.delta_nir(2),0,settings.delta_nir(3)];
                    settings.delta1 = settings.delta1_nir;
                end
                
                %R0
                if settings.update_state == -1
                
                mags                   = abs(settings.delta);
                clamping_magnitude_wl  = mags(1) + mags(3);
                clamping_magnitude_nir = mags(2) + mags(4);
  
                if settings.delta(1,5) > clamping_magnitude_wl 
                    settings.delta(1,5) = sign(settings.delta(1,5))*clamping_magnitude_wl;
                end
                
                if settings.delta(1,6) > clamping_magnitude_nir
                    settings.delta(1,6) = sign(settings.delta(1,6))*clamping_magnitude_nir;
                end
                
                end
                 
%                 if settings.update_state == 0
%                     mags = abs(settings.delta_visible);
%                     clamping_magnitude_visible = mags(1) + mags(2);
%                     if settings.delta(1,5) > clamping_magnitude_visible
%                         settings.delta(1,5) = sign(settings.delta(1,5)) * clamping_magnitude_visible;
%                     end
%                 end
%                 
%                 if settings.update_state == 1
%                     mags = abs(settings.delta_nir);
%                     clamping_magnitude_nir = mags(1) + mags(2);
%                     if settings.delta(1,6) > clamping_magnitude_nir
%                         settings.delta(1,6) = sign(settings.delta(1,6)) * clamping_magnitude_nir;
%                     end
%                 end
%                 %R0
                
                
                if settings.change == 0 || settings.led_update_mode == 0
                    
                    new_led_Is = settings.ledI.*(1+settings.delta);
                    
                    if settings.update_state == 0
                    if settings.n_visible_updates > 1
                        new_led_Is = settings.memory_visible_leds.*(1+settings.delta);
                    end
                    end
                    
                    if settings.update_state == 1
                    if settings.n_nir_updates > 1
                        new_led_Is = settings.memory_nir_leds.*(1+settings.delta);
                    end
                    end
                    
                    % get deltas from here. How do you combine
                    % synchronously ?
                    
                    v          = round(new_led_Is);
                    %v(v>thr)   = thr; %can make it a global
                    v(v>9999)  = 9999;
                    v(v<16)    = 16;
                    v          = uint16(v);
                    v          = [v(1),v(2),v(3),v(4),v(5),v(6)];
                
                end
                
                if settings.change == 0
                    
                    settings.estimated_led_update = v;
                    if settings.update_state == 0
                        settings.estimated_led_update = [v(1),16,v(3),16,v(5),16];
                    else
                        settings.estimated_led_update = [16,v(2),16,v(4),16,v(6)];
                    end
                    
                    settings.change = 1;
                
                end
                
                    
                if settings.led_update_mode
                    
                    if settings.ledI == settings.estimated_led_update 
                    
                        nir_sum = settings.ledI(2) + settings.ledI(4) + settings.ledI(6);
                        vis_sum = settings.ledI(1) + settings.ledI(3) + settings.ledI(5);
                        
                        if settings.update_state == 0
                            
                            if nir_sum == 48
                            
                                settings.memory_visible      = settings.all_lines(settings.calibration_band_indices(1) ,:)';
                                settings.memory_visible_leds = settings.ledI;
                                
                                %can use an external counter -
                                %settings.optimization count to start
                                %analyzing.
                                settings.adapt_memory_vis_ctr = settings.adapt_memory_vis_ctr + 1;
                                if settings.adapt_memory_vis_ctr > settings.adapt_memory_length
                                    settings.adapt_memory_vis_ctr = 1;
                                end
                                settings.adapt_memory_vis_leds(settings.adapt_memory_vis_ctr,:) = settings.ledI;
                                
                                
                                settings.stable_vis_cmos_state = settings.all_lines(1:settings.visible_stop_index,:);
                                for j = 1:3
                                    settings.stable_led_state(1,2*j - 1) = settings.ledI(1,2*j - 1);
                                end
                                
                                settings.update_state        = 1;
                                settings.n_visible_updates   = settings.n_visible_updates + 1;
                                settings.n_updates = settings.n_updates + 1;
                                settings.change              = 0; 
                            
                            end
                            
                        elseif settings.update_state == 1
                            
                            if vis_sum == 48
                                
                                settings.memory_nir          = settings.all_lines(settings.calibration_band_indices(2) ,:)';
                                settings.memory_nir_leds     = settings.ledI;
                                
                                
                                %can use an external counter -
                                %settings.optimization count to start
                                %analyzing.
                                settings.adapt_memory_nir_ctr = settings.adapt_memory_nir_ctr + 1;
                                if settings.adapt_memory_nir_ctr > settings.adapt_memory_length
                                    settings.adapt_memory_nir_ctr = 1;
                                end
                                settings.adapt_memory_nir_leds(settings.adapt_memory_nir_ctr,:) = settings.ledI;
                                
                                
                                settings.stable_nir_cmos_state = settings.all_lines(settings.visible_stop_index+1:settings.n_lines * settings.bands_per_line,:);
                                for j = 1:3
                                    settings.stable_led_state(1,2*j) = settings.ledI(1,2*j);
                                end
                                
                                settings.update_state        = 0;
                                settings.n_nir_updates       = settings.n_nir_updates + 1;
                                settings.n_updates           = settings.n_updates + 1;
                                settings.change              = 0;    
                            
                            end
                            
                        end
                        
                        settings.stable_cmos_state(1:settings.visible_stop_index,:) = settings.stable_vis_cmos_state;
                        settings.stable_cmos_state(settings.visible_stop_index+1:settings.n_lines * settings.bands_per_line,:) = settings.stable_nir_cmos_state;
                        
                    else
                        code          = zeros(20,1);
                        code(1:8)     = sprintf('%8s','LED_INT');
                        code(9:2:20)  = uint8(bitshift(settings.estimated_led_update,-8));
                        code(10:2:20) = uint8(mod(settings.estimated_led_update,256));
                        fwrite(settings.serialport,code);
                        %what if I wait here instead ? 
                    end
                    
                else
                    
                    code          = zeros(20,1);
                    code(1:8)     = sprintf('%8s','LED_INT');
                    code(9:2:20)  = uint8(bitshift(v,-8));
                    code(10:2:20) = uint8(mod(v,256));
                    fwrite(settings.serialport,code);
                    
                end
                

                file_content = [settings.ledI];
                for j = 1:settings.n_lines * settings.bands_per_line
                    file_content = [file_content settings.all_lines(j,:)];
                end
                file_content = [file_content settings.set_to ...
                settings.optimization_count settings.delta settings.set_rt settings.giving_up ...
                settings.converged settings.distance settings.led_update_mode settings.update_state settings.y_config(settings.config_index,:)];
                
                fprintf(settings.rt_fid, settings.rt_types, file_content);
            end
            
            %Max optimization steps reached
            if settings.optimization_count == settings.max_optimization_steps
                       
                settings.adapt_end      = 1;
                settings.giving_up      = 1;
                settings.start_optimize = 0;
                settings.mean_error     = 2048;

                %run a loop here, collect a few more samples to average
                if settings.post_conv_ctr < settings.post_conv_length
                    
                    file_content = [settings.ledI];
                    for j = 1:settings.n_lines * settings.bands_per_line
                        file_content = [file_content settings.all_lines(j,:)];
                    end
                    file_content = [file_content settings.set_to ...
                    settings.optimization_count settings.delta settings.set_rt settings.giving_up ...
                    settings.converged settings.distance settings.led_update_mode settings.update_state settings.y_config(settings.config_index,:)];
                    
                    fprintf(settings.rt_fid, settings.rt_types, file_content);
                    settings.post_conv_ctr = settings.post_conv_ctr + 1;
                else
                    settings.optimization_count = 0;
                    settings.post_conv_ctr      = 0;
                    settings.set_rt             = -1;
                    
                    file_content = [settings.ledI];
                    for j = 1:settings.n_lines * settings.bands_per_line
                        file_content = [file_content settings.all_lines(j,:)];
                    end
                    file_content = [file_content settings.set_to ...
                    settings.optimization_count settings.delta settings.set_rt settings.giving_up ...
                    settings.converged settings.distance settings.led_update_mode settings.update_state settings.y_config(settings.config_index,:)];
                   
                    fprintf(settings.rt_fid, settings.rt_types, file_content);
                    
                    set(handles.baselineEdit,'BackgroundColor',[255,4,64]/255);
                    fclose(settings.rt_fid);                    
                    settings.rt_file_name = 'rt_';  
                end                    
            end

            %Convergence in mean l1 distance
            if settings.mean_error <= settings.mean_error_threshold 
                %disp('Good solution arrived at.');
                %disp(settings.mean_error);
                set(handles.baselineEdit,'BackgroundColor',[64,255,64]/255);

                settings.adapt_end          = 1;
                settings.converged          = 1;
                settings.optimization_count = 0;
                settings.set_rt             = -1;
                settings.mean_error         = 2048;
                settings.start_optimize     = 0;
                settings.delta1             = 64;

                %run a loop here, collect a few more samples to average
                file_content = [settings.ledI];
                for j = 1:settings.n_lines * settings.bands_per_line
                    file_content = [file_content settings.all_lines(j,:)]; 
                end
                file_content = [file_content settings.set_to ...
                settings.optimization_count settings.delta settings.set_rt settings.giving_up ...
                settings.converged settings.distance settings.led_update_mode settings.update_state settings.y_config(settings.config_index,:)];
                
                fprintf(settings.rt_fid, settings.rt_types, file_content);

                fclose(settings.rt_fid);
                settings.rt_file_name = 'rt_';
            end          
            
            %stagnating updates
            if settings.optimization_count > settings.monitor_updates_trigger
                if settings.stop_early
                    m1       = mean(settings.adapt_memory_vis_leds,1);
                    m2       = mean(settings.adapt_memory_nir_leds,1);
                    vis_copy = settings.adapt_memory_vis_leds;
                    nir_copy = settings.adapt_memory_nir_leds;
                    for i = 1:settings.n_leds
                        for j = 1:settings.adapt_memory_length
                            vis_copy(j,i) = vis_copy(j,i) - m1(1,i);
                            nir_copy(j,i) = nir_copy(j,i) - m2(1,i);
                        end
                    end
                    s1       = sum(abs(vis_copy),1);
                    s2       = sum(abs(nir_copy),1);
                    s        = [s1(1,1),s2(1,2),s1(1,3),s2(1,4),s1(1,5),s2(1,6)];
                    
                    if max(abs(s)) < settings.adapt_dev_threshold
                        
                        disp('LED deltas flatline/low amplitude oscillation.Stopping early.');
                        set(handles.baselineEdit,'BackgroundColor',[64,64,255]/255);

                        settings.adapt_end          = 1;
                        settings.converged          = 1;
                        settings.optimization_count = 0;
                        settings.set_rt             = -1;
                        settings.mean_error         = 2048;
                        settings.start_optimize     = 0;
                        settings.delta1             = 64;

                        %run a loop here, collect a few more samples to average
                        file_content = [settings.ledI];
                        for j = 1:settings.n_lines * settings.bands_per_line
                            file_content = [file_content settings.all_lines(j,:)]; 
                        end
                        file_content = [file_content settings.set_to ...
                        settings.optimization_count settings.delta settings.set_rt settings.giving_up ...
                        settings.converged settings.distance settings.led_update_mode settings.update_state settings.y_config(settings.config_index,:)];

                        fprintf(settings.rt_fid, settings.rt_types, file_content);

                        fclose(settings.rt_fid);
                        settings.rt_file_name = 'rt_';
                    
                    end
                    
                end
            end

            %run next , if queues not empty.
            if settings.set_rt == -1
                
                if settings.config_index == settings.n_configs
                    if ~isempty(settings.adapt_queue)
                        settings.config_index = 1; %update along with baseline list in iteration 1?
                        settings.adapt_queue(end) = [];
                        settings.set_rt = 1;
                        if ~isempty(settings.adapt_queue)
                            set(handles.baselineEdit,'BackgroundColor','White');
                        else
                            disp('Baseline-Config grid executed.');
                            settings.set_rt = -1;
                            
                            %display figure here ? Spectral content ? 
                            if settings.user_spectral_verification
                                figure(128);
                                plot(settings.wl_set,[settings.stable_vis_cmos_state(:,48)' settings.stable_nir_cmos_state(:,48)']);
                                title('Spectral content , midpoint of the primary RoI');
                                ylim([256,4096]);
                                xlim([450,930]);
                            end
                            
                        end
                    end
                else
                    settings.set_rt = 1;
                    settings.config_index = settings.config_index + 1;
                end
                
            end
            
            %replay final state , for demos and observation , guarded against iterative runs by code order.
            if settings.set_rt == -1 && settings.replay_final
                
                settings.replay_led_vis = [settings.stable_led_state(1,1),16,...
                                           settings.stable_led_state(1,3),16,...
                                           settings.stable_led_state(1,5),16];
                settings.replay_led_nir = [16,settings.stable_led_state(1,2),...
                                           16,settings.stable_led_state(1,4),...
                                           16,settings.stable_led_state(1,6)];
                if mod(settings.replay_counter,settings.replay_wait_count)

                    %new vis and nir memory
                    %save this data ? 
                    %stable led state , stable vis cmos state , stable vis
                    %cmos state
                    if sum(settings.ledI == settings.replay_led_vis) == 6
                        settings.replay_update_state = 1; %needed if need for verification.Keep for now.
                        settings.replay_cmos_vis = settings.all_lines(1:settings.visible_stop_index,:);
                        %send nir values to board.
                        
                        code          = zeros(20,1);
                        code(1:8)     = sprintf('%8s','LED_INT');
                        code(9:2:20)  = uint8(bitshift(settings.replay_led_nir,-8));
                        code(10:2:20) = uint8(mod(settings.replay_led_nir,256));
                        fwrite(settings.serialport,code);
                        
                    elseif sum(settings.ledI == settings.replay_led_nir) == 6
                        settings.replay_update_state = 0; %needed if need for verification.Keep for now.
                        settings.replay_cmos_nir = settings.all_lines(settings.visible_stop_index+1:...
                                                                      settings.n_lines * settings.bands_per_line,:);
                        
                        code          = zeros(20,1);
                        code(1:8)     = sprintf('%8s','LED_INT');
                        code(9:2:20)  = uint8(bitshift(settings.replay_led_vis,-8));
                        code(10:2:20) = uint8(mod(settings.replay_led_vis,256));
                        fwrite(settings.serialport,code);
                        %send vis values to board.
                    else
                        settings.replay_update_state = -1;
                        %disp('In wait state.Forcing WL LEDs.'); 
                        code          = zeros(20,1);
                        code(1:8)     = sprintf('%8s','LED_INT');
                        code(9:2:20)  = uint8(bitshift(settings.replay_led_vis,-8));
                        code(10:2:20) = uint8(mod(settings.replay_led_vis,256));
                        fwrite(settings.serialport,code);
                    end
                
                else
                    settings.replay_counter = settings.replay_counter + 1;
                end
                
            end

            %% Analysis - Localization defaults 
            
            if settings.adapt_end
                
                settings.adapt_end = 0;
                
                cmos_standardized  = settings.stable_cmos_state;
                control_final      = settings.stable_led_state;
                
                %calculate metric and set flags to plot bar.
                %Example localization metric for ureters , discussed in the
                %grant progress report.
                
                vis_ref = cmos_standardized(settings.calibration_band_indices(1),:)./settings.global_cmos_max;
                nir_ref = cmos_standardized(settings.calibration_band_indices(2),:)./settings.global_cmos_max;
                
                v1 = cmos_standardized(8,:)./vis_ref;
                v2 = vis_ref./cmos_standardized(2,:);
                v1 = v1 - mean(v1);
                v2 = v2 - mean(v2);
                v3 = 0.5*(v1+v2);
                
                for j=1:settings.n_pixels
                    if v3(1,j) > 0
                        v3(1,j) = 1;
                    else
                        v3(1,j) = 0;
                    end
                end
                
                settings.localization_1_indicator_array = v3;
                settings.update_localization_plot_1     = 1;
                
                v_840 = cmos_standardized(12,:)./settings.global_cmos_max;
                v_840 = v_840 - mean(v_840);
                for j  = 1:99
                    if v_840(1,j) < 0
                        v_840(1,j) = 1;
                    end
                end
                
                settings.localization_2_indicator_array = v_840 ;
                settings.update_localization_plot_2     = 1;                
            end    
            
            %% Plot axis 0
            
            if settings.operation_mode == 0
                
                if settings.plot_virtual 
                    if settings.set_rt == 2
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line1_sources(1,j) == 1
                                    settings.T{j} = settings.stable_vis_cmos_state(settings.line1_indices(1,j),:);
                                else
                                    settings.T{j} = settings.stable_nir_cmos_state(settings.line1_indices(1,j),:);
                                end
                            end
                        end
                        
                    elseif settings.set_rt == -1 && settings.replay_final == 1
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line1_sources(1,j) == 1
                                    settings.T{j} = settings.replay_cmos_vis(settings.line1_indices(1,j),:);
                                else
                                    settings.T{j} = settings.replay_cmos_nir(settings.line1_indices(1,j),:);
                                end
                            end
                        end
                    
                    else
                    settings.T{1} = settings.line1_data(1,:); 
                    settings.T{2} = settings.line1_data(2,:); 
                    settings.T{3} = settings.line1_data(3,:); 
                    settings.T{4} = settings.line1_data(4,:);
                    settings.T{5} = settings.line1_data(5,:);
                    settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    
                    end    
                else
                    settings.T{1} = settings.line1_data(1,:); 
                    settings.T{2} = settings.line1_data(2,:); 
                    settings.T{3} = settings.line1_data(3,:); 
                    settings.T{4} = settings.line1_data(4,:);
                    settings.T{5} = settings.line1_data(5,:);
                    %settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    if settings.set_rt == 2
                        settings.T{6} = settings.config_plot;
                    else
                        settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    end
                end
                set(settings.h_dc_shadow_0, {'YData'},settings.T);
            else
                settings.T{1} = settings.line1_data(1,:)'; 
                settings.T{2} = settings.line1_data(2,:)'; 
                settings.T{3} = settings.line1_data(3,:)'; 
                settings.T{4} = settings.line1_data(4,:)';
                settings.T{5} = settings.line1_data(5,:)';
                settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                set(settings.h_dc_shadow, {'YData'},settings.T);
            end
            
            %% Plot axis 1
            if settings.operation_mode == 0
                
                if settings.plot_virtual
                    if settings.set_rt == 2
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line2_sources(1,j) == 1
                                    settings.T{j} = settings.stable_vis_cmos_state(settings.line2_indices(1,j),:);
                                else
                                    settings.T{j} = settings.stable_nir_cmos_state(settings.line2_indices(1,j),:);
                                end
                            end
                        end
                    
                    elseif settings.set_rt == -1 && settings.replay_final == 1
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line2_sources(1,j) == 1
                                    settings.T{j} = settings.replay_cmos_vis(settings.line2_indices(1,j),:);
                                else
                                    settings.T{j} = settings.replay_cmos_nir(settings.line2_indices(1,j),:);
                                end
                            end
                        end
                        
                    else
                    settings.T{1} = settings.line2_data(1,:); 
                    settings.T{2} = settings.line2_data(2,:); 
                    settings.T{3} = settings.line2_data(3,:); 
                    settings.T{4} = settings.line2_data(4,:);
                    settings.T{5} = settings.line2_data(5,:);
                    settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    end  
                else
                    settings.T{1} = settings.line2_data(1,:); 
                    settings.T{2} = settings.line2_data(2,:); 
                    settings.T{3} = settings.line2_data(3,:); 
                    settings.T{4} = settings.line2_data(4,:);
                    settings.T{5} = settings.line2_data(5,:);
                    %settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    if settings.set_rt == 2
                        settings.T{6} = settings.config_plot;
                    else
                        settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    end
                end
                set(settings.h_dc_shadow_1, {'YData'},settings.T);
                
                
            else
                settings.T{1} = settings.line1_data(1,:); 
                settings.T{2} = settings.line1_data(2,:);
                settings.T{3} = settings.line1_data(3,:); 
                settings.T{4} = settings.line1_data(4,:);
                settings.T{5} = settings.line1_data(5,:);
                settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                set(settings.h_dc_shadow_1, {'YData'},settings.T);
            end
            
            %% Plot axis 2
            if settings.operation_mode == 0
                 
                if settings.plot_virtual
                    if settings.set_rt == 2
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line3_sources(1,j) == 1
                                    settings.T{j} = settings.stable_vis_cmos_state(settings.line3_indices(1,j),:);
                                else
                                    settings.T{j} = settings.stable_nir_cmos_state(settings.line3_indices(1,j),:);
                                end
                            end
                        end
                        
                    elseif settings.set_rt == -1 && settings.replay_final == 1
                        for j = 1:6
                            if j == 6
                                settings.T{j} = ones(settings.n_pixels,1)*settings.set_to; 
                            else
                                if settings.line3_sources(1,j) == 1
                                    settings.T{j} = settings.replay_cmos_vis(settings.line3_indices(1,j),:);
                                else
                                    settings.T{j} = settings.replay_cmos_nir(settings.line3_indices(1,j),:);
                                end
                            end
                        end
                        
                    else
                    settings.T{1} = settings.line3_data(1,:); 
                    settings.T{2} = settings.line3_data(2,:); 
                    settings.T{3} = settings.line3_data(3,:); 
                    settings.T{4} = settings.line3_data(4,:);
                    settings.T{5} = settings.line3_data(5,:);
                    settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    end  
                else
                    settings.T{1} = settings.line3_data(1,:); 
                    settings.T{2} = settings.line3_data(2,:); 
                    settings.T{3} = settings.line3_data(3,:); 
                    settings.T{4} = settings.line3_data(4,:);
                    settings.T{5} = settings.line3_data(5,:);
                    %settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    if settings.set_rt == 2
                        settings.T{6} = settings.config_plot;
                    else
                        settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                    end
                end
                set(settings.h_dc_shadow_2, {'YData'},settings.T);
                
            else
                settings.T{1} = settings.line1_data(1,:); 
                settings.T{2} = settings.line1_data(2,:);
                settings.T{3} = settings.line1_data(3,:); 
                settings.T{4} = settings.line1_data(4,:);
                settings.T{5} = settings.line1_data(5,:);
                settings.T{6} = ones(settings.n_pixels,1)*settings.set_to;
                set(settings.h_dc_shadow_2, {'YData'},settings.T);
            end
 
            %% Plot on Localization plot 1
            if settings.update_localization_plot_1    
                
                color = [1,0.75,0];

                for j = 1:settings.n_pixels
                    settings.color_array_1(1,j,:) = settings.localization_1_indicator_array(1,j).*color;
                end
                
                set(handles.localization_plot_0,'Ytick',[]);
                imagesc(handles.localization_plot_0,settings.color_array_1);
                
                settings.update_localization_plot_1 = 0;
            
            end
            
            %% Plot on localization plot 2
            if settings.update_localization_plot_2    
                
                color = [0,0.75,1];

                for j = 1:settings.n_pixels
                    settings.color_array_2(1,j,:) = settings.localization_2_indicator_array(1,j).*color;
                end
                
                set(handles.localization_plot_1,'Ytick',[]);
                imagesc(handles.localization_plot_1,settings.color_array_2);
                
                settings.update_localization_plot_2 = 0;
            
            end
            
            %% sample load-scan
            if settings.fp_scan && settings.set_rt == -1
                
                target        = settings.sample_scan_target;
                target_local  = settings.sample_scan_parent; 
                disp(target);
                
                if isfolder(target)
                    disp('Path exists , collecting information ... ');
                    file_list = dir(fullfile(target,'*.csv'));
                    nfiles = length(file_list);
                    
                    if settings.led_update_mode == 0
                        settings.nscans = nfiles;
                        settings.scan_intensities = zeros(settings.nscans,6);
                    elseif settings.led_update_mode == 1
                        settings.nscans = 2 * nfiles;
                        settings.scan_intensities = zeros(settings.nscans,6);
                    end
                    
                    for i = 1:nfiles

                        current = fullfile(target,file_list(i).name);
                        disp(current);
                        content = readtable(current);
                        content = content(:,1:6);
                        content = unique(content,'stable');
                        
                        if height(content) > 1
                        
                            if settings.led_update_mode == 0
                                end_state = content(height(content),1:6);
                                settings.scan_intensities(i,:) = [end_state.led1,...
                                end_state.led2,end_state.led3,end_state.led4,...
                                end_state.led5,end_state.led6];
                        
                            elseif settings.led_update_mode == 1
                                end_state   = content(height(content),1:6);
                                sum_visible = end_state.led1 + end_state.led3 + end_state.led5;
                                sum_nir     = end_state.led2 + end_state.led4 + end_state.led6;
                            
                                if sum_visible == 48 && sum_nir == 48
                                    end_state_visible = end_state;
                                    end_state_nir     = end_state;
                            
                                elseif sum_visible == 48 && sum_nir ~=48
                                    end_state_nir     = end_state;
                                    end_state_visible = content(height(content)-1,1:6);
                                elseif sum_nir == 48 && sum_visible ~=48
                                    end_state_visible = end_state;
                                    end_state_nir = content(height(content)-1,1:6);
                                end
                            
                                settings.scan_intensities(2*i-1,:) = [end_state_visible.led1,...
                                    end_state_visible.led2,end_state_visible.led3,end_state_visible.led4,...
                                    end_state_visible.led5,end_state_visible.led6];
                                
                                settings.scan_intensities(2*i,:) = [end_state_nir.led1,end_state_nir.led2,...
                                    end_state_nir.led3,end_state_nir.led4,end_state_nir.led5,end_state_nir.led6];
                            
                            end
                        end
                    end
                    disp('Creating file to save settling intensities...');
                    if ~isfolder(settings.fp_folder_name)
                        disp('Creating settling intensities folder ... ');
                        mkdir(settings.fp_folder_name);
                    end
                    settings.fpscan_file_name = fullfile(pwd,settings.fp_folder_name,strcat(target_local,'_rt_settling_states.csv'));
                    settings.fpscan_fid       = fopen(settings.fpscan_file_name,'w');
                    disp(settings.fpscan_file_name);
                    fprintf(settings.fpscan_fid,settings.fpscan_columns);

                    settings.start_scan  = 1;
                    settings.fp_scan     = 0;
                    settings.scan_length = settings.nscans;
                    settings.scan_ctr    = 1;
                else
                    disp('Folder does not exist, enter valid path.');
                    settings.fp_scan = 0;
                end
            end

            if settings.start_scan 
                if settings.scan_ctr < settings.fpscan_start_offset
                    settings.scan_ctr = settings.scan_ctr + 1;
                else
                    if settings.scan_ctr == settings.fpscan_start_offset
                        
                        disp('Scanning...');
                        settings.scan_ctr   = settings.scan_ctr + 1;
                        settings.fp_counter = 0;
                        settings.scan_index = 1;
                        settings.verified   = 1;
                        
                    else
                        if settings.verified ...
                                && settings.scan_index <= settings.scan_length

                            settings.state = settings.scan_intensities(settings.scan_index,:);
                            v              = round(settings.state);
                            v              = uint16(v);
                            v              = [v(1),v(2),v(3),v(4),v(5),v(6)];

                            send          = zeros(20,1);
                            send(1:8)     = sprintf('%8s','LED_INT');
                            send(9:2:20)  = uint8(bitshift(v,-8));
                            send(10:2:20) = uint8(mod(v,256));
                            fwrite(settings.serialport,send);

                            settings.verified = 0;

                        else

                            if settings.state == settings.ledI
                                
                                settings.scan_index = settings.scan_index + 1;
                                disp(settings.scan_index);
                                settings.verified   = 1;
                                
                                file_content = [settings.ledI];
                                for j = 1:settings.n_lines * settings.bands_per_line
                                    file_content = [file_content settings.all_lines(j,:)];
                                end
                                file_content = [file_content settings.verified];
                                fprintf(settings.fpscan_fid,settings.fpscan_types,file_content);
                            
                            else
                                
                                settings.verified = 0;
                            
                            end

                            if settings.scan_index == settings.scan_length + 1
                                
                                fclose(settings.fpscan_fid);
                                disp('Finishing up scan, saving file to disk ...');
                                disp('Done.');
                                settings.start_scan = 0;
                                settings.scan_ctr   = 0;
                            
                            end

                        end
                        
                        settings.fp_counter = settings.fp_counter + 1;
                    
                    end
                end
            end        
            
            %% ua_data_save
            if get(handles.dataSaveButton, 'UserData')
                
                %disp(settings.shuffle_order);
                
                if ~settings.count

                    settings.count = 1;

                    folderName = [settings.folderName ''];

                    set(handles.dataSaveButton,'BackgroundColor',[0,204,102]./255,...
                        'FontWeight','bold','Enable','inactive');

                    vessel     = get(handles.VesselMenu,'UserData');
                    vesselsize = get(handles.VesselSizeEdit,'String');
                    tissue     = get(handles.TissueMenu,'UserData');
                    thickness  = get(handles.ThicknessEdit,'String');
                    baseline   = num2str(settings.set_to);
                    comment    = get(handles.CommentsEdit,'String');
                    tool       = handles.tool_select_pop_up.String{handles.tool_select_pop_up.Value};
                    line       = handles.line_select_pop_up.String{handles.line_select_pop_up.Value};
                    user       = handles.user_select_pop_up.String{handles.user_select_pop_up.Value};
                    orientation= handles.orientation_select_pop_up.String{handles.orientation_select_pop_up.Value};
                    approach   = handles.approach_select_pop_up.String{handles.approach_select_pop_up.Value};
                    ja_manual  = get(handles.manual_jaw_angle_edit,'String');
                    
                    pig_1_id        = get(handles.pig_1_id_edit,'String');
                    pig_1_sample    = get(handles.sample_1_id_edit,'String');
                    pig_1_subsample = get(handles.subsample_1_id_edit,'String');

                    pig_2_id        = get(handles.pig_2_id_edit,'String');
                    pig_2_sample    = get(handles.sample_2_id_edit,'String');
                    pig_2_subsample = get(handles.subsample_2_id_edit,'String');
                    
                    pri_start  = get(handles.primary_start_edit,'String');
                    pri_end    = get(handles.primary_end_edit,'String');
                    sec_start  = get(handles.secondary_start_edit,'String');
                    sec_end    = get(handles.secondary_end_edit,'String');

                    current_time = strsplit(datestr(datetime('now')));
                    current_time = strrep(current_time,':','_');
                    
                    p_or_m = num2str(settings.pixels_or_markers);


                    settings.fileName = strcat('x15_ua','_v_',vessel,'_vs_',vesselsize,'_t_',tissue,'_th_', thickness,...
                                                '_b_',baseline, '_c_',comment,'_tool_', tool,'_line_',line,'_user_', user ,...
                                                '_o_',orientation,'_app_',approach,'_ja_',ja_manual,'_s1_',pig_1_id,'-',pig_1_sample,'-',pig_1_subsample,...
                                                '_s2_',pig_2_id,'-',pig_2_sample,'-',pig_2_subsample,...
                                                '_pos1_',pri_start,'-',pri_end,...
                                                '_pos2_',sec_start,'-',sec_end,'_pix_',p_or_m,'_clock_',current_time{2},'.csv');

                    if exist(folderName,'dir')~=7
                        mkdir(folderName);
                    end

                    settings.savefolderName = folderName;

                    % Create a Temp File
                    settings.tempName = tempname();
                    settings.temp_fid = fopen(settings.tempName,'w');
                    fprintf(settings.temp_fid,settings.columnNames);
                    tic;
                else        
                    settings.dataSaveTime= 5;
                    if toc < settings.dataSaveTime
                        set(handles.dataSaveButton,'String','Saving Data');
                        set(handles.dataSaveButton,'BackgroundColor',[0.5 1 0.5]);
                        
                        if settings.operation_mode == 0
                            saveData = [settings.ledI];
                            for i = 1:settings.n_lines * settings.bands_per_line
                                saveData = [saveData settings.all_lines(i,:)];
                            end
                        else
                            saveData = [settings.ledI settings.cmos_data'];
                        end
                        
                        fprintf(settings.temp_fid, settings.dataType, saveData);
                    else
                        name = fullfile(pwd, settings.savefolderName, settings.fileName);
                        copyfile(settings.tempName, name);
                        settings.count = 0;
                        fclose(settings.temp_fid);
                        set(handles.dataSaveButton,'BackgroundColor',0.94*ones(1,3),...
                            'FontWeight','bold','Enable','on', 'String','Save Data');
                        set(handles.dataSaveButton,'UserData',0);
                        disp('Data saved.');
                    end
                end
            end    
            
            %% Calibration Module
            if ~settings.calibrate
                
                if settings.calibLED == 10
                    
                    set(handles.calibrateButton,'Value',0);
                    msgbox('Calibration Done!');
                    
                    code = zeros(20,1);
                    code(1:16) = sprintf('%16s','LED_intensity');
                    fwrite(settings.serialport,code);
                    settings.calibLED = 1;
                    settings.beginCalibrate = 0;
                    settings.calibrate = 0;
                    settings.calibcount = 1;
                    settings.calibwait = 1;
                end

                if get(handles.calibrateButton,'Value') && settings.calibLED < 10 && ...
                        ~settings.beginCalibrate
                    
                    % Send the Command to the Serial Port
                    code = zeros(20,1);
                    code(1:16) = sprintf('%16s','calibrate_i0');
                    selbits = '00000000';
                    if settings.calibLED == 7
                        selbits = '00010101';
                    elseif settings.calibLED == 8
                        selbits = '00101010';
                    elseif settings.calibLED == 9
                        selbits = '00111111';
                    else
                        inds = [1,2,3,4,5,6];
                        selbits(8 - inds(settings.calibLED) + 1) = '1';               
                    end
                    
                    maxIntensity = settings.max_calib_intensity;
                    
                    code(17) = uint8(bin2dec(selbits));
                    code(18) = uint8(bitshift(maxIntensity,-8));
                    code(19) = uint8(mod(maxIntensity,256));
                    fwrite(settings.serialport,code);
                    settings.calibrate = 1;
                    settings.calibLED = settings.calibLED + 1;
                end

            else
                if ~settings.beginCalibrate
                    
                    settings.tempName       = tempname();
                    settings.temp_fid       = fopen(settings.tempName,'w');
                    
                    if settings.operation_mode == 0
                        fprintf(settings.temp_fid,settings.columnNames);
                    else
                        fprintf(settings.temp_fid,settings.sp_calib_columns);
                    end
                    
                    settings.beginCalibrate = 1;
                    
                else
                    if settings.calibcount > 10 && settings.calibcount <= settings.n_calib_led_sweeps
                        
                        if settings.operation_mode == 0
                            
                            calib_content = [settings.ledI];
                            for j = 1:settings.n_lines * settings.bands_per_line
                                calib_content = [calib_content settings.all_lines(j,:)]; 
                            end
                            settings.calib_data(settings.calibcount - 10, : ) = calib_content;
                        else
                            calib_content = [settings.ledI];
                            for j = 1:settings.bands_per_line
                                calib_content = [calib_content settings.line1_data(j,:)];
                            end
                            settings.calib_data(settings.calibcount-10,:) = calib_content;
                        end
                        
                    end
                    settings.calibcount = settings.calibcount+1;

                    maxIntensity = settings.max_calib_intensity;
                    thr          = maxIntensity-20;

                    if any(settings.ledI>=thr)
                        if ~mod(settings.calibwait,40)
                            settings.beginCalibrate = 0;
                            settings.calibrate = 0;
                            settings.calibcount = 1;
                            settings.calibwait = 1;

                            tempfile = tempname();
                            fid = fopen(tempfile,'w');
                            
                            if settings.operation_mode == 0
                                fprintf(fid,settings.columnNames);
                            else
                                fprintf(fid,settings.sp_calib_columns);
                            end
                            
                            %commented
                            if settings.operation_mode == 0
                                fprintf(fid,settings.dataType,settings.calib_data');
                            else
                                %settings.var_out = [settings.var_out;settings.calib_data'];
                                fprintf(fid,settings.sp_calib_type,settings.calib_data');
                            end
                            

                            vessel     = get(handles.VesselMenu,'UserData');
                            vesselsize = get(handles.VesselSizeEdit,'String');
                            tissue     = get(handles.TissueMenu,'UserData');
                            thickness  = get(handles.ThicknessEdit,'String');
                            baseline   = num2str(settings.set_to);
                            comment    = get(handles.CommentsEdit,'String');
                            tool       = handles.tool_select_pop_up.String{handles.tool_select_pop_up.Value};
                            line       = handles.line_select_pop_up.String{handles.line_select_pop_up.Value};
                            user       = handles.user_select_pop_up.String{handles.user_select_pop_up.Value};
                            orientation= handles.orientation_select_pop_up.String{handles.orientation_select_pop_up.Value};
                            approach   = handles.approach_select_pop_up.String{handles.approach_select_pop_up.Value};
                            ja_manual  = get(handles.manual_jaw_angle_edit,'String');
                            
                            pig_1_id        = get(handles.pig_1_id_edit,'String');
                            pig_1_sample    = get(handles.sample_1_id_edit,'String');
                            pig_1_subsample = get(handles.subsample_1_id_edit,'String');

                            pig_2_id        = get(handles.pig_2_id_edit,'String');
                            pig_2_sample    = get(handles.sample_2_id_edit,'String');
                            pig_2_subsample = get(handles.subsample_2_id_edit,'String');
                            
                            pri_start  = get(handles.primary_start_edit,'String');
                            pri_end    = get(handles.primary_end_edit,'String');
                            sec_start  = get(handles.secondary_start_edit,'String');
                            sec_end    = get(handles.secondary_end_edit,'String');

                            current_time = strsplit(datestr(datetime('now')));
                            current_time = strrep(current_time,':','_');
                            
                            p_or_m = num2str(settings.pixels_or_markers);


                            filename   = strcat('x15_calib','_v_',vessel,'_vs_',vesselsize,'_t_',tissue,'_th_', thickness,...
                                                '_b_',baseline, '_c_',comment,'_tool_', tool,'_line_',line,'_user_', user ,...
                                                '_o_',orientation,'_app_',approach,'_ja_',ja_manual,'_s1_',pig_1_id,'-',pig_1_sample,'-',pig_1_subsample,...
                                                '_s2_',pig_2_id,'-',pig_2_sample,'-',pig_2_subsample,...
                                                '_pos1_',pri_start,'-',pri_end,...
                                                '_pos2_',sec_start,'-',sec_end,'_pix_',p_or_m,'_clock_',current_time{2});

                            foldername = settings.calibfolder;
                            
                            %filename = strcat('hsi_x15_calib_','v_',vessel,'_vs_',vesselsize,'_t_',tissue,'_th_', thickness, '_c_',comment,'_tool_', tool, '_line_', line,'_user_', user , '_ja_' ,jawangle);
                            
                            if settings.calibLED < 8
                                filename = strcat(filename,'_led_',num2str(settings.calibLED-1),'.csv');
                            elseif settings.calibLED == 8
                                filename = strcat(filename,'_led_135.csv');
                            elseif settings.calibLED == 9 
                                filename = strcat(filename,'_led_246.csv');
                            else
                                filename = strcat(filename,'_led_123456.csv');
                            end

                            % Check if the folder Already Exists. Create one if it does not
                            if exist(foldername,'dir')~=7
                                mkdir(foldername);
                            end
                            %copyfile(tempfile,fullfile(foldername,filesep,filename));
                            copyfile(tempfile,fullfile(foldername,filename));
                            fclose(fid);
                            settings.calibfile = settings.calibfile+1;

                        else
                            settings.calibwait = settings.calibwait+1;
                        end
                    end  
                end
            end
            set(hSettings, 'UserData', settings);
        else
            disp('Unusual packet.');
            settings.unusual_encounters_ctr = settings.unusual_encounters_ctr + 1;

        end
end
set(hSettings, 'UserData', settings);

%% Functions that are used in the main function

function update_led_intensity(handles,val1,val2,n)
    code= zeros(20,1);
    code(1:16) = sprintf('%16s',['LED' num2str(n) '_intensity']);
    code(17) = val1;
    code(18) = val2;
    fwrite(handles.serialport,code);     


function jawAngleEdit_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function jawAngleEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to jawAngleEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in dataSaveButton.
function dataSaveButton_Callback(hObject, eventdata, handles)
% hObject    handle to dataSaveButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject,'UserData',1);

function LED2_Edit_Callback(hObject, eventdata, handles)
% hObject    handle to LED2_Edit (see GCBO)
% eventdata  re`served - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of LED2_Edit as text
%        str2double(get(hObject,'String')) returns contents of LED2_Edit as a double

thr = 9999;

val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));

update_led_intensity(handles,val1,val2,2);

% --- Executes during object creation, after setting all properties.
function LED2_Edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LED2_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function LED4_Edit_Callback(hObject, eventdata, handles)
% hObject    handle to LED4_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of LED4_Edit as text
%        str2double(get(hObject,'String')) returns contents of LED4_Edit as a double

thr = 9999;

val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));

update_led_intensity(handles,val1,val2,4);

% --- Executes during object creation, after setting all properties.
function LED4_Edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LED4_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function LED6_Edit_Callback(hObject , eventdata, handles)
% hObject    handle to LED6_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of LED6_Edit as text
%        str2double(get(hObject,'String')) returns contents of LED6_Edit as a double

thr = 9999;
%s = get(hSettings,'UserData');
val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

%t = hObject.UserData;

val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));

update_led_intensity(handles,val1,val2,6);
%adaptWhitevals(handles,handles.serialport,val);

% --- Executes during object creation, after setting all properties.
function LED6_Edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LED6_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in autoUpdateLEDCheckbox.
function autoUpdateLEDCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to autoUpdateLEDCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of autoUpdateLEDCheckbox

function yAxisMinEdit_Callback(hObject, eventdata, handles)
% hObject    handle to yAxisMinEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of yAxisMinEdit as text
%        str2double(get(hObject,'String')) returns contents of yAxisMinEdit as a double

handles.yAxesMinEdit = str2double(get(hObject,'String'));
set(handles.plot_axis_0,'YLim',[handles.yAxesMinEdit, handles.yAxesMaxEdit]);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function yAxisMinEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to yAxisMinEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function yAxisMaxEdit_Callback(hObject, eventdata, handles)
% hObject    handle to yAxisMaxEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of yAxisMaxEdit as text
%        str2double(get(hObject,'String')) returns contents of yAxisMaxEdit as a double

handles.yAxesMaxEdit = str2double(get(hObject,'String'));
set(handles.plot_axis_0,'YLim',[handles.yAxesMinEdit, handles.yAxesMaxEdit]);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function yAxisMaxEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to yAxisMaxEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in yAxisResetButton.
function yAxisResetButton_Callback(hObject, eventdata, handles)
% hObject    handle to yAxisResetButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.yMin = 100;
handles.yMax = 4500;
set(handles.plot_axis_0,'YLim',[handles.yMin, handles.yMax]);

% --- Executes on button press in saveSizeEstimatesCheckbox.
function saveSizeEstimatesCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to saveSizeEstimatesCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of saveSizeEstimatesCheckbox

% --- Executes on button press in indivDataSaveCheckbox.
function indivDataSaveCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to indivDataSaveCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of indivDataSaveCheckbox
function VesselSizeEdit_Callback(hObject, eventdata, handles)
% hObject    handle to VesselSizeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function VesselSizeEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to VesselSizeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function ThicknessEdit_Callback(hObject, eventdata, handles)
% hObject    handle to ThicknessEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function ThicknessEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ThicknessEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in VesselMenu.
function VesselMenu_Callback(hObject, eventdata, handles)
% hObject    handle to VesselMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns VesselMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from VesselMenu

contents = cellstr(get(hObject,'String'));
vessel = contents{get(hObject,'Value')};
set(hObject,'UserData',vessel);
if strcmpi(vessel,'none')
    set(handles.VesselSizeEdit,'String','0mm');
end

% --- Executes during object creation, after setting all properties.
function VesselMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to VesselMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in TissueMenu.
function TissueMenu_Callback(hObject, eventdata, handles)
% hObject    handle to TissueMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns TissueMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from TissueMenu

contents = cellstr(get(hObject,'String'));
tissue = contents{get(hObject,'Value')};
set(hObject,'UserData',tissue);
if strcmpi(tissue,'none')
    set(handles.ThicknessEdit,'String','0mm');
end

% --- Executes during object creation, after setting all properties.
function TissueMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TissueMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function JawAngleEdit_Callback(hObject, eventdata, handles)
% hObject    handle to JawAngleEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of JawAngleEdit as text
%        str2double(get(hObject,'String')) returns contents of JawAngleEdit as a double


% --- Executes during object creation, after setting all properties.
function JawAngleEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to JawAngleEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function VesselPositionEdit_Callback(hObject, eventdata, handles)
% hObject    handle to VesselPositionEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of VesselPositionEdit as text
%        str2double(get(hObject,'String')) returns contents of VesselPositionEdit as a double


% --- Executes during object creation, after setting all properties.
function VesselPositionEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to VesselPositionEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in AutoPositionFillCheckbox.
function AutoPositionFillCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to AutoPositionFillCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of AutoPositionFillCheckbox


% --- Executes on button press in LED1Checkbox.
function LED1Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED1Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED1Checkbox


% --- Executes on button press in LED3Checkbox.
function LED2Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED3Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED3Checkbox


% --- Executes on button press in LED3Checkbox.
function LED3Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED3Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED3Checkbox


% --- Executes on button press in LED4Checkbox.
function LED4Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED4Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED4Checkbox


% --- Executes on button press in LED5Checkbox.
function LED5Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED5Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED5Checkbox


% --- Executes on button press in LED5Checkbox.
function LED6Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED5Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED5Checkbox

% --- Executes during object creation, after setting all properties.
function BaselineEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to BaselineEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function DataSaveTimeEdit_Callback(hObject, eventdata, handles)
% hObject    handle to DataSaveTimeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of DataSaveTimeEdit as text
%        str2double(get(hObject,'String')) returns contents of DataSaveTimeEdit as a double

% --- Executes during object creation, after setting all properties.
function DataSaveTimeEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DataSaveTimeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function CommentsEdit_Callback(hObject, eventdata, handles)
% hObject    handle to CommentsEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CommentsEdit as text
%        str2double(get(hObject,'String')) returns contents of CommentsEdit as a double


% --- Executes during object creation, after setting all properties.
function CommentsEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CommentsEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ShowPumpReadingCheckbox.
function ShowPumpReadingCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to ShowPumpReadingCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ShowPumpReadingCheckbox

code = zeros(20,1);
code(1:16) = sprintf('%16s','monitor_pump');
code(17) = get(hObject,'Value');
fwrite(handles.serialport,code);


% --- Executes on button press in scan_push.
function scan_push(hObject, eventdata, handles)
% hObject    handle to scan_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
disp('In scan push');


v             = [0,0,0,0,0,0];
v             = round(v);
v             = uint16(v);
code          = zeros(20,1);
code(1:8)     = sprintf('%8s','LED_INT');
code(9:2:20)  = uint8(bitshift(v,-8));
code(10:2:20) = uint8(mod(v,256));
fwrite(handles.serialport,code);

s             = handles.output.UserData;
scan_folder   = uigetdir();
scan_parent   = strsplit(scan_folder,filesep);

depth         = size(scan_parent);
depth         = depth(2);
scan_parent   = scan_parent{depth};

s.sample_scan_parent = scan_parent;
s.sample_scan_target = scan_folder;
s.fp_scan            = 1;

set(handles.output,'UserData',s);

% --- Executes on button press in AutoFill_JA_Checkbox.
function AutoFill_JA_Checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to AutoFill_JA_Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of AutoFill_JA_Checkbox

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close_button_Callback(hObject,[],handles);

% % --- Executes during object creation, after setting all properties.
% function dcPlotAxes_CreateFcn(hObject, eventdata, handles)
% % hObject    handle to plot_axis_0 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    empty - handles not created until after all CreateFcns called
% 
% % Hint: place code in OpeningFcn to populate plot_axis_0

function baselineEdit_Callback(hObject, eventdata, handles)
% hObject    handle to baselineEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baselineEdit as text
%        str2double(get(hObject,'String')) returns contents of baselineEdit as a double

set(handles.baselineEdit,'BackgroundColor','White');
thr = 3600;
val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

v             = [0,0,0,0,0,0];
v             = round(v);
v             = uint16(v);
code          = zeros(20,1);
code(1:8)     = sprintf('%8s','LED_INT');
code(9:2:20)  = uint8(bitshift(v,-8));
code(10:2:20) = uint8(mod(v,256));
fwrite(handles.serialport,code);

s        = handles.output.UserData;
s.set_rt = 1;
val      = round(val);
s.adapt_queue = [];
s.adapt_queue = [val s.adapt_queue];
%s.set_to = val;

set(handles.output,'UserData',s);

% --- Executes during object creation, after setting all properties.
function baselineEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baselineEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in smoothSignalCheckbox.
function smoothSignalCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to smoothSignalCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of smoothSignalCheckbox


% --- Executes on button press in hsi_check.
function hsi_check_Callback(hObject, eventdata, handles)
% hObject    handle to hsi_check (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.LED3Checkbox,'Value',0);
set(handles.LED4Checkbox,'Value',0);
set(handles.LED5Checkbox,'Value',0);
set(handles.LED3Checkbox,'Visible',1);
set(handles.LED4Checkbox,'Visible',1);
set(handles.LED5Checkbox,'Visible',1);
set(handles.autoUpdateLEDCheckbox,'Value',0);
set(handles.led1_hsi_edit, 'Visible',1);
set(handles.led1_hsi, 'Visible',1);
set(handles.LED1Checkbox, 'Visible',1);
set(handles.led3_hsi_edit, 'Visible',1);
set(handles.led3_hsi, 'Visible',1);
set(handles.LED3Checkbox, 'Visible',1);
set(handles.led5_hsi_edit, 'Visible',1);
set(handles.led5_hsi, 'Visible',1);
set(handles.LED5Checkbox, 'Visible',1);
set(handles.LED1Checkbox,'Value',1);
set(handles.LED3Checkbox,'Value',1);
set(handles.LED5Checkbox,'Value',1);
% Hint: get(hObject,'Value') returns toggle state of hsi_check



function led5_hsi_edit_Callback(hObject, eventdata, handles)
% hObject    handle to led5_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
thr = 9999;

val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

% val = uint16(val);
% val1 = uint8(bitshift(val,-8));
% val2 = uint8(mod(val,256));

%adaptWhitevals(handles.serialport,val);
val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));

update_led_intensity(handles,val1,val2,5);
% Hints: get(hObject,'String') returns contents of led5_hsi_edit as text
%        str2double(get(hObject,'String')) returns contents of led5_hsi_edit as a double


% --- Executes during object creation, after setting all properties.
function led5_hsi_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to led5_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function led1_hsi_edit_Callback(hObject, eventdata, handles)
% hObject    handle to led1_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
thr = 9999;

val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));

update_led_intensity(handles,val1,val2,1);
% Hints: get(hObject,'String') returns contents of led1_hsi_edit as text
%        str2double(get(hObject,'String')) returns contents of led1_hsi_edit as a double


% --- Executes during object creation, after setting all properties.
function led1_hsi_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to led1_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in LED1Checkbox.
function led1_hsi_cbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED1Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED1Checkbox

function led3_hsi_edit_Callback(hObject, eventdata, handles)
% hObject    handle to led3_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
thr = 9999;

val = str2double(get(hObject,'String'));
if val > thr || val < 0 
    val = (val > thr)*thr + (val < 0)*0;
end

val = uint16(val);
val1 = uint8(bitshift(val,-8));
val2 = uint8(mod(val,256));
update_led_intensity(handles,val1,val2,3);
% adaptvals(handles.serialport,val)
% Hints: get(hObject,'String') returns contents of led3_hsi_edit as text
%        str2double(get(hObject,'String')) returns contents of led3_hsi_edit as a double


% --- Executes during object creation, after setting all properties.
function led3_hsi_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to led3_hsi_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in LED3Checkbox.
function led3_hsi_cbox_Callback(hObject, eventdata, handles)
% hObject    handle to LED3Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED3Checkbox


% --- Executes on button press in LED5Checkbox.
function led5_hsi_cbox_Callback(hObject, ~, handles)
% hObject    handle to LED5Checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LED5Checkbox


% --- Executes on button press in calibrateButton.
function calibrateButton_Callback(hObject, eventdata, handles)
% hObject    handle to calibrateButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in user_select_pop_up.
function user_select_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to user_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns user_select_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from user_select_pop_up


% --- Executes during object creation, after setting all properties.
function user_select_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to user_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in tool_select_pop_up.
function tool_select_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to tool_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

s         = handles.output.UserData;
all_tools = cellstr(get(hObject,'String'));
tool      = all_tools(get(hObject,'Value'));

if strcmp(tool , 'x15_ud1')
    
    s.shuffle_order = [14,2,4,15,13,1,11,10,8,5,12,3,6,9,7]; 
    s.line1_legend  = {'635', '520', '840', '550','770','Reference'};
    s.line2_legend  = {'870', '930', '740', '900','700','Reference'};
    s.line3_legend  = {'660', '810', '610', '450','580','Reference'};
    
    s.line1_sources = [1,1,2,1,1];
    s.line1_indices = [6,2,1,3,10];

    s.line2_sources = [2,2,1,2,1];
    s.line2_indices = [2,4,9,3,8];

    s.line3_sources = [1,1,1,1,1];
    s.line3_indices = [7,11,5,1,4];
    
elseif strcmp(tool , 'x15_ud2')
    s.shuffle_order = [11,3,5,12,15,2,13,10,8,1,14,4,6,9,7]; 
    s.line1_legend  = {'770', '635', '520', '840','550','Reference'};
    s.line2_legend  = {'870', '930', '740', '900','700','Reference'};
    s.line3_legend  = {'450', '580', '660', '810','610','Reference'};
    
    s.line1_sources = [1,1,1,2,1];
    s.line1_indices = [10,6,2,1,3];

    s.line2_sources = [2,2,1,2,1];
    s.line2_indices = [2,4,9,3,8];

    s.line3_sources = [1,1,1,1,1];
    s.line3_indices = [1,4,7,11,5];

elseif strcmp(tool , 'x15_ud3')
    s.shuffle_order = [14,3,1,13,15,4,12,10,7,5,11,2,9,6,8]; 
    s.line1_legend  = {'550', '840', '520', '635','770','Reference'};
    s.line2_legend  = {'700', '900', '740', '930','870','Reference'};
%     s.line2_legend  = {'900', '740', '930', '870','700','Reference'};
    s.line3_legend  = {'810', '660', '580', '450','610','Reference'};

    
    s.line1_sources = [1,2,1,1,1];
    s.line1_indices = [3,1,2,6,10];

    s.line2_sources = [2,1,2,2,1];
    s.line2_indices = [3,9,4,2,8];

    s.line3_sources = [1,1,1,1,1];
    s.line3_indices = [11,7,4,1,5];
    
end

legend(s.h_dc_shadow_0,s.line1_legend);
legend(s.h_dc_shadow_1,s.line2_legend);
legend(s.h_dc_shadow_2,s.line3_legend);

set(handles.output,'UserData',s);



% --- Executes during object creation, after setting all properties.
function tool_select_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tool_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in zero_led_push.
function zero_led_push_Callback(hObject, eventdata, handles)
% hObject    handle to zero_led_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s = handles.output.UserData;
set(handles.output,'UserData',s);
code = zeros(20,1);
code(1:8) = sprintf('%8s','LED_INT');
fwrite(handles.serialport,code);


% --- Executes on button press in reconnect_push.
function reconnect_push_Callback(hObject, eventdata, handles)
% hObject    handle to reconnect_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s         = handles.output.UserData;
set(handles.output,'UserData',s);
code      = zeros(20,1);
code(1:8) = sprintf('%8s','LED_INT');
fwrite(handles.serialport,code);

function sys_user_dialogue_edit_Callback(hObject, eventdata, handles)
% hObject    handle to sys_user_dialogue_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sys_user_dialogue_edit as text
%        str2double(get(hObject,'String')) returns contents of sys_user_dialogue_edit as a double

s = handles.output.UserData;
user_ip = get(hObject,'String');

if  strcmpi(user_ip , s.code_led_update_mode_0)
    s.led_update_mode = 0;
    out_string = 'LED updates set to 1 step cycle mode.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_led_update_mode_1)
    s.led_update_mode = 1;
    out_string = 'LED updates set to 2 step cycle mode.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_pixel_shift_0)
    s.line_shift = 0;
    out_string = 'Pixels shift set to 0.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_pixel_shift_1)
    s.line_shift = 1;
    out_string = 'Pixels shift set to 1.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_pixel_shift_2)
    s.line_shift = 2;
    out_string = 'Pixels shift set to 2.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_pixel_shift_3)
    s.line_shift = 3;
    out_string = 'Pixels shift set to 3.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_pixel_shift_4)
    s.line_shift = 4;
    out_string = 'Pixels shift set to 4.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_clear_adapt_queue)
    s.adapt_queue = [];
    out_string = 'Adapt queue cleared.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_disp_adapt_queue)
    out_string = num2str(s.adapt_queue);
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strfind(user_ip,s.code_add_to_adapt_queue)
    val = strsplit(user_ip,s.code_add_to_adapt_queue);
    val = val{end};
    if ~isnan(str2double(val))
        val = str2double(val);
        s.adapt_queue = [val s.adapt_queue];
        out_string = 'Value added to adapt queue.';
    else
        out_string = 'Unrecognized tail.';
    end
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_as_pixels)
    s.pixels_or_markers = 0;
    out_string = 'Position entry set to pixel units.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_set_as_markers)
    s.pixels_or_markers = 1;
    out_string = 'Position entry set to marker units.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_plot_virtual)
    s.plot_virtual = 1;
    out_string = 'Plotting virtually while in adapt.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_plot_real)
    s.plot_virtual = 0;
    out_string = 'Plotting rt while in adapt.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_replay_final_state)
    s.replay_final = 1;
    out_string = 'Mode : Replay final.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_keep_final_state)
    s.replay_final = 0;
    out_string = 'Mode : Keep final.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_stop_early)
    s.stop_early = 1;
    out_string = 'Early stopping enabled.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_no_early_stopping)
    s.stop_early = 0;
    out_string = 'Early stopping disabled.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_restore_init_conf)
    s.config_index  = 1;
    s.y_config      = zeros(1,6);
    s.y_config(1,:) = [1,1,1,1,1,1];
    s.n_configs     = 1;
    out_string      = 'Shape settings restored to defaults.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
elseif strcmpi(user_ip,s.code_user_verify_spectral)
    s.user_spectral_verification = 1;
elseif strcmpi(user_ip,s.code_user_ignore_spectral)
    s.user_spectral_verification = 0;
else
    out_string = 'No matches found under current codeword list.';
    set(handles.sys_user_dialogue_edit,'String',out_string);
end

set(handles.output,'UserData',s);


% --- Executes during object creation, after setting all properties.
function sys_user_dialogue_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sys_user_dialogue_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cmd_clear_push.
function cmd_clear_push_Callback(hObject, eventdata, handles)
% hObject    handle to cmd_clear_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in exec_config_push.
function exec_config_push_Callback(hObject, eventdata, handles)
% hObject    handle to exec_config_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 s             = handles.output.UserData;
 
 init_string   = 'Initiating run.';
 set(handles.sys_user_dialogue_edit,'String',init_string);

 s.exec_config = 1;
 
 set(handles.output,'UserData',s);


% --- Executes on button press in load_model_push.
function load_model_push_Callback(hObject, eventdata, handles)
% hObject    handle to load_model_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in load_calib_push.
function load_calib_push_Callback(hObject, eventdata, handles)
% hObject    handle to load_calib_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 s             = handles.output.UserData;
 init_string = 'Process : Calibration Select/Load. Waiting for user.';
 set(handles.sys_user_dialogue_edit,'String',init_string);

 [f,p]       = uigetfile();
 %log with time , the calib load path and if it was loadable.
 send_string = ['Selected file : ' f];
 set(handles.sys_user_dialogue_edit,'String',send_string);
 
 calib_dict = load(fullfile(p,f));
 
 s.calibration_band_indices     = [calib_dict.bands(1)+1,calib_dict.bands(2)+1];
 s.track_visible                = calib_dict.track_visible;
 s.track_nir                    = calib_dict.track_nir;
 
 s.parameters_visible           = calib_dict.parameters_visible;
 s.parameters_nir               = calib_dict.parameters_nir;
 
 s.parameters_visible.cmos_mins = double(s.parameters_visible.cmos_mins);
 s.parameters_visible.cmos_maxs = double(s.parameters_visible.cmos_maxs);
 s.parameters_visible.led_means = double(s.parameters_visible.led_means);
 s.parameters_visible.led_stds  = double(s.parameters_visible.led_stds);
 s.parameters_visible.led_mins  = s.parameters_visible.led_mins_norm;
 s.parameters_visible.led_maxs  = s.parameters_visible.led_maxs_norm;

 s.parameters_nir.cmos_mins     = double(s.parameters_nir.cmos_mins);
 s.parameters_nir.cmos_maxs     = double(s.parameters_nir.cmos_maxs);
 s.parameters_nir.led_means     = double(s.parameters_nir.led_means);
 s.parameters_nir.led_stds      = double(s.parameters_nir.led_stds);
 s.parameters_nir.led_mins      = s.parameters_nir.led_mins_norm;
 s.parameters_nir.led_maxs      = s.parameters_nir.led_maxs_norm;
 
 s.parameters_visible.c1        = -1.*(s.parameters_visible.calib_bs./s.parameters_visible.calib_as);
 s.parameters_visible.c2        = 1./s.parameters_visible.calib_as;
 
 s.parameters_nir.c1            = -1.*(s.parameters_nir.calib_bs./s.parameters_nir.calib_as);
 s.parameters_nir.c2            = 1./s.parameters_nir.calib_as;
 
 s.pixels_all  = [s.track_visible(1) ,s.track_nir(1),s.track_visible(2)  ...
                        s.track_nir(2),s.track_visible(3) ,s.track_nir(3)];
 
 set(handles.output,'UserData',s);



function instance_id_field_Callback(hObject, eventdata, handles)
% hObject    handle to instance_id_field (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of instance_id_field as text
%        str2double(get(hObject,'String')) returns contents of instance_id_field as a double


% --- Executes during object creation, after setting all properties.
function instance_id_field_CreateFcn(hObject, eventdata, handles)
% hObject    handle to instance_id_field (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in line_select_pop_up.
function line_select_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to line_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns line_select_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from line_select_pop_up


% --- Executes during object creation, after setting all properties.
function line_select_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to line_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in load_config_push.
function load_config_push_Callback(hObject, eventdata, handles)
% hObject    handle to load_config_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 
 s             = handles.output.UserData;
 init_string = 'Process : Profile config Select/Load. Waiting for user.';
 set(handles.sys_user_dialogue_edit,'String',init_string);

 [f,p]       = uigetfile('*.mat');
 send_string = ['Selected file : ' f];
 set(handles.sys_user_dialogue_edit,'String',send_string);
 
 config_dict = load(fullfile(p,f));
 s_config_dict = size(config_dict);
 
 if s_config_dict(2) == s.n_leds
     s.profile_config_list = config_dict;
     s.n_profiles_loaded   = s_config_dict(1);
     send_string = ' Profiles loaded.'; %No guards on values loaded.
     set(handles.sys_user_dialogue_edit,'String',send_string);
 else
     send_string = 'Expected a different input shape.';
     set(handles.sys_user_dialogue_edit,'String',send_string);
 end
 
 set(handles.output,'UserData',s);

% --- Executes on button press in exec_ref_queue_push.
function exec_ref_queue_push_Callback(hObject, eventdata, handles)
% hObject    handle to exec_ref_queue_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of exec_ref_queue_push


% --- Executes on button press in disp_help_list.
function disp_help_list_Callback(hObject, eventdata, handles)
% hObject    handle to disp_help_list (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
x15_acq_help;


% --- Executes on button press in ref_set_x_push.
function ref_set_x_push_Callback(hObject, eventdata, handles)
% hObject    handle to ref_set_x_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.baselineEdit,'BackgroundColor','White');

s        = handles.output.UserData;
s.set_rt = 1;
%s.adapt_queue = [2400,2200,2000,1800,1600,1400,1200];
s.adapt_queue = [2000,1600];
set(handles.output,'UserData',s);


% --- Executes on button press in unit_test_2_push.
function unit_test_2_push_Callback(hObject, eventdata, handles)
% hObject    handle to unit_test_2_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pause_push.
function pause_push_Callback(hObject, eventdata, handles)
% hObject    handle to pause_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s        = handles.output.UserData;
s.replay_final = 0;
set(handles.output,'UserData',s);




% --- Executes on button press in config_set_1_push.
function config_set_1_push_Callback(hObject, eventdata, handles)
% hObject    handle to config_set_1_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s                   = handles.output.UserData;
s.set_rt            = 0;
s.n_configs         = 5;
s.y_config          = zeros(s.n_configs,6);
s.y_config(1,:)     = [1,1,0,0,0,0];
s.y_config(2,:)     = [0,0,1,1,0,0];
s.y_config(3,:)     = [0,0,0,0,1,1];
s.y_config(4,:)     = [1,1,1,1,1,1];
s.y_config(5,:)     = [0.5,0.5,1,1,0.5,0.5];
out_string = 'Configs loaded - Coarse calibrations.';
set(handles.sys_user_dialogue_edit,'String',out_string);
set(handles.output,'UserData',s);


% --- Executes on button press in unit_test_4_push.
function unit_test_4_push_Callback(hObject, eventdata, handles)
% hObject    handle to unit_test_4_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in uts_other_push.
function uts_other_push_Callback(hObject, eventdata, handles)
% hObject    handle to uts_other_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in orientation_select_pop_up.
function orientation_select_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to orientation_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns orientation_select_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from orientation_select_pop_up


% --- Executes during object creation, after setting all properties.
function orientation_select_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to orientation_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in togglebutton4.
function togglebutton4_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton4



function primary_start_edit_Callback(hObject, eventdata, handles)
% hObject    handle to primary_start_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of primary_start_edit as text
%        str2double(get(hObject,'String')) returns contents of primary_start_edit as a double


% --- Executes during object creation, after setting all properties.
function primary_start_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to primary_start_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function primary_end_edit_Callback(hObject, eventdata, handles)
% hObject    handle to primary_end_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of primary_end_edit as text
%        str2double(get(hObject,'String')) returns contents of primary_end_edit as a double


% --- Executes during object creation, after setting all properties.
function primary_end_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to primary_end_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function secondary_start_edit_Callback(hObject, eventdata, handles)
% hObject    handle to secondary_start_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of secondary_start_edit as text
%        str2double(get(hObject,'String')) returns contents of secondary_start_edit as a double


% --- Executes during object creation, after setting all properties.
function secondary_start_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to secondary_start_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function secondary_end_edit_Callback(hObject, eventdata, handles)
% hObject    handle to secondary_end_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of secondary_end_edit as text
%        str2double(get(hObject,'String')) returns contents of secondary_end_edit as a double


% --- Executes during object creation, after setting all properties.
function secondary_end_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to secondary_end_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in approach_select_pop_up.
function approach_select_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to approach_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns approach_select_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from approach_select_pop_up


% --- Executes during object creation, after setting all properties.
function approach_select_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to approach_select_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function manual_jaw_angle_edit_Callback(hObject, eventdata, handles)
% hObject    handle to manual_jaw_angle_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of manual_jaw_angle_edit as text
%        str2double(get(hObject,'String')) returns contents of manual_jaw_angle_edit as a double


% --- Executes during object creation, after setting all properties.
function manual_jaw_angle_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to manual_jaw_angle_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over exec_ref_queue_push.
function exec_ref_queue_push_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to exec_ref_queue_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.baselineEdit,'BackgroundColor','White');
s        = handles.output.UserData;
s.set_rt = 1;
set(handles.output,'UserData',s);



function pig_1_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to pig_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pig_1_id_edit as text
%        str2double(get(hObject,'String')) returns contents of pig_1_id_edit as a double


% --- Executes during object creation, after setting all properties.
function pig_1_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pig_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pig_2_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to pig_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pig_2_id_edit as text
%        str2double(get(hObject,'String')) returns contents of pig_2_id_edit as a double


% --- Executes during object creation, after setting all properties.
function pig_2_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pig_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sample_1_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to sample_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sample_1_id_edit as text
%        str2double(get(hObject,'String')) returns contents of sample_1_id_edit as a double


% --- Executes during object creation, after setting all properties.
function sample_1_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sample_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sample_2_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to sample_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sample_2_id_edit as text
%        str2double(get(hObject,'String')) returns contents of sample_2_id_edit as a double


% --- Executes during object creation, after setting all properties.
function sample_2_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sample_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function subsample_1_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to subsample_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of subsample_1_id_edit as text
%        str2double(get(hObject,'String')) returns contents of subsample_1_id_edit as a double


% --- Executes during object creation, after setting all properties.
function subsample_1_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to subsample_1_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function subsample_2_id_edit_Callback(hObject, eventdata, handles)
% hObject    handle to subsample_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of subsample_2_id_edit as text
%        str2double(get(hObject,'String')) returns contents of subsample_2_id_edit as a double


% --- Executes during object creation, after setting all properties.
function subsample_2_id_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to subsample_2_id_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function plot_axis_0_CreateFcn(hObject, eventdata, handles)
% hObject    handle to plot_axis_0 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate plot_axis_0
