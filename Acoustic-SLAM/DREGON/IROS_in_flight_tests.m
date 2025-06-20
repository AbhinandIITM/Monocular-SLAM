%%%%%%%%%%%%%%%%%%%  Test file for IROS paper %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CHANGE HERE THE PATHS SO THEY FIT YOURS

% PATH to MBSS LOCATE V2.0
MBSS_PATH = 'mbss\';
addpath([MBSS_PATH,'localization_tools\']);
addpath([MBSS_PATH,'localization_tools\pair_angular_meths\']);

% PATH to DREGON data:
DREGON_PATH = '';

% Name of the test flight considered:
test_name = 'free-flight_whitenoise-high_room1'; %1
% test_name = 'free-flight_speech-high_room1'; %2
% test_name = 'free-flight_whitenoise-low_room1'; %3
% test_name = 'free-flight_speech-low_room1'; %4
% test_name = 'silent-flight_whitenoise-low_room1'; %6

noisy_test_path = [DREGON_PATH,test_name,'/DREGON_',test_name,'.wav'];                          
                                
audiots_path = [DREGON_PATH,test_name,'/DREGON_',test_name,'_audiots.mat'];
                                
sourcepos_path = [DREGON_PATH,test_name,'/DREGON_',test_name,'_sourcepos.mat'];

noise_training_path = [DREGON_PATH,'noise_training_room1.wav'];   

%% Parameter definitions
% Analysis window size in miliseconds
window_length_ms = 512; % ms

% Frequency of sampling
fs = 16000; 

% Microphone array
micPos = [  0.0420    0.0615   -0.0410;  % mic 1
           -0.0420    0.0615    0.0410;  % mic 2
           -0.0615    0.0420   -0.0410;  % mic 3
           -0.0615   -0.0420    0.0410;  % mic 4
           -0.0420   -0.0615   -0.0410;  % mic 5
            0.0420   -0.0615    0.0410;  % mic 6
            0.0615   -0.0420   -0.0410;  % mic 7
			0.0615    0.0420    0.0410]; % mic 8     

isArrayMoving   = false; % The microphone array is static
subArray        = 1:8; % Use all microphones
%subArray        = [3,4,6,7,8]; % Use subset of less windy microphones
sceneTimeStamps = [];    % Both array and sources are statics => no time stamps

% Localization method
angularSpectrumMeth        = 'GCC-PHAT'; % Local Angular spectrum method {'GCC-PHAT' 'GCC-NONLIN' 'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'}
pooling                    = 'sum';      % Pooling method {'max' 'sum'}
applySpecInstNormalization = 0;          % 1: Normalize instantaneous local angular spectra - 0: No normalization
% Search space
azBound                    = [-179 180]; % Azimuth search boundaries (�)
elBound                    = [-90 20];   % Elevation search boundaries (�)
gridRes                    = 1;          % Resolution (�) of the global 3D reference system {theta (azimuth),phi (elevation)}
alphaRes                   = 5;          % Resolution (�) of the 2D reference system defined for each microphone pair
% Multiple sources parameters
nsrce                      = 1;          % Number of sources to be detected
minAngle                   = 10;         % Minimum angle between peaks
% Moving sources parameters
blockDuration_sec          = [];         % Block duration in seconds (default []: one block for the whole signal)
blockOverlap_percent       = 0;          % Requested block overlap in percent (default []: No overlap) - is internally rounded to suited values
% Wiener filtering
enableWienerFiltering      = 1;             % 1: Process a Wiener filtering step in order to attenuate / emphasize the provided excerpt signal into the mixture signal. 0: Disable Wiener filtering
wienerMode                 = 'Attenuation'; % Wiener filtering mode {'[]' 'Attenuation' 'Emphasis'} - In this example considered signal (noise) is attenuated in the mixture
% Display results
specDisplay                = 0;          % 1: Display angular spectrum found and sources directions found - 0: No display
% Other parameters
speedOfSound               = 343;        % Speed of sound (m.s-1) - typical value: 343 m.s-1 (assuming 20�C in the air at sea level)
fftSize_sec                = [];         % FFT size in seconds (default []: 0.064 sec)
freqRange                  = [];         % Frequency range to aggregate the angular spectrum : [] means no specified range
% Debug
angularSpectrumDebug       = 0;          % Flag to enable additional plots to debug the angular spectrum aggregation

sMBSSParam = MBSS_InputParam2Struct(angularSpectrumMeth,...
                                    speedOfSound,...
                                    fftSize_sec,...
                                    blockDuration_sec,...
                                    blockOverlap_percent,...
                                    pooling,...
                                    azBound,...
                                    elBound,...
                                    gridRes,...
                                    alphaRes,...
                                    minAngle,...
                                    nsrce,...
                                    fs,...
                                    applySpecInstNormalization,...
                                    specDisplay,...
                                    enableWienerFiltering,...
                                    wienerMode,...
                                    freqRange,...
                                    micPos',...
                                    isArrayMoving,...
                                    subArray,...
                                    sceneTimeStamps,...
                                    angularSpectrumDebug);

%% Load noisy signal, ground truth and time stamp data
% Noisy signal
[noisy_signal,fs_true] = audioread(noisy_test_path); % load testing noise
noisy_signal = resample(noisy_signal,fs,fs_true); % resample
noisy_signal_length = size(noisy_signal,1); % get length of noise
noisy_signal = noisy_signal(:,subArray); % use subarray of microphones

% Audio timestamps
str = load(audiots_path);
audio_ts = str.audio_timestamps;
t0 = audio_ts(1);
audio_ts = audio_ts - t0; % Set audio start to time 0
audio_ts = resample(audio_ts,fs,fs_true); % resample

% Source positions
str=load(sourcepos_path);
spos = str.source_position;
s_az = spos.azimuth;
s_el = spos.elevation;
pos_ts = spos.timestamps - t0; % Set audio start to time 0

% Noise training signal for Wiener filter
[x_noise,fs_true]= audioread(noise_training_path);
x_noise = resample(x_noise,fs,fs_true);
x_noise = x_noise(:,subArray); % use subarray of microphones

fprintf('Data are loaded\n\n');

%% Perform tests for different parameters
% Initialization
count = 1;
idx = 1;

% Window length in samples:
window_length_samples = round(window_length_ms*fs/1000);

% Number of tests:
ntests = floor(noisy_signal_length/window_length_samples);

result_param = zeros(ntests,2);
analysis_ts = zeros(ntests,1);

fprintf('--------------- Start tests --------------\n\n');
while(idx + window_length_samples < noisy_signal_length)

    % Save analysis timestamp
    analysis_ts(count) = audio_ts(idx+round(window_length_samples/2));
    windowed_noisy_signal = noisy_signal(idx:idx+window_length_samples,:);
    
    % Perform SSL
    [theta,phi] = MBSS_locate_spec(windowed_noisy_signal,x_noise,sMBSSParam);

    [phi,minidx] = min(phi);
    theta = theta(minidx);

    % Save results in result vector
    result_param(count,1) = theta;
    result_param(count,2) = phi;

    idx = idx + window_length_samples;
    count = count + 1;
end

%% Plot
% Azimuth plot
fig=figure;clf(fig);
scatter(analysis_ts,result_param(:,1),'Linewidth',2);
hold on
plot(pos_ts,s_az,'Linewidth',2);
title('Azimuth estimation in flight')
xlabel('Time [s]');
ylabel('Azimuth [�]');
legend('Estimate (WF + GCC-PHAT)','Ground truth (Vicon)')

% Elevation plot
fig=figure;clf(fig);
scatter(analysis_ts,result_param(:,2),'Linewidth',2);
hold on
plot(pos_ts,s_el,'Linewidth',2);
title('Elevation estimation in flight')
xlabel('Time [s]');
ylabel('Elevation [�]');
legend('Estimate (WF + GCC-PHAT)','Ground truth (Vicon)')
