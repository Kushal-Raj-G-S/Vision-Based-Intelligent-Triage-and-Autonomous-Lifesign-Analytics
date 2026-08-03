% Simple code to read video and ground truth of the UBFCRPPG DATASET
% If you use the dataset, please cite:
% 
% S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, 
% Unsupervised skin tissue segmentation for remote photoplethysmography, 
% Pattern Recognition Letters, Elsevier, 2017.
% 
% yannick.benezeth@u-bourgogne.fr

clear all;
close all;
clc;

% dataset folder
root        =   'DATASET_2/';

% get folder list
dirs=dir(root);
dirs=dirs(~ismember({dirs.name},{'.','..','desktop.ini'}));

%Iterate through all directories
for i=1:size(dirs)
    vidFolder   =   [root dirs(i).name];    
    
    % load ground truth
	gtfilename=[vidFolder '/gtdump.xmp']; % DATASET_1
	if exist(gtfilename, 'file')==2
		gtdata=csvread(gtfilename);
		gtTrace=gtdata(:,4);
		gtTime=gtdata(:,1)/1000;
		gtHR = gtdata(:,2);
	else 
		gtfilename=[vidFolder '/ground_truth.txt']; %DATASET_2
		if exist(gtfilename, 'file')==2
			gtdata=dlmread(gtfilename);
			gtTrace=gtdata(1,:)';
			gtTime=gtdata(3,:)'; 
			gtHR = gtdata(2,:)';	
		end
	end
	% normalize data (zero mean and unit variance)
	gtTrace = gtTrace - mean(gtTrace,1);
	gtTrace = gtTrace / std(gtTrace);      
	    
    % open video file
    vidObj = VideoReader([ vidFolder '/vid.avi' ]);
    fps = vidObj.FrameRate;

    n=0;
    while hasFrame(vidObj)
        % track frame index
        n=n+1;

        % read frame by frame
        img = readFrame(vidObj);

        % perform operations on frame
        imshow( img );
    end
    %fprintf('%i: %i - %i ; %i - %i \n',i, n, length(gt_time), vidObj.Duration, gt_time(end));
end
