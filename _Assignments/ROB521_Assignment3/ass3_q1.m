% =========
% ass3_q1.m
% =========
%
% This assignment will introduce you to the idea of first building an
% occupancy grid then using that grid to estimate a robot's motion using a
% particle filter.
% 
% There are two questions to complete (5 marks each):
%
%    Question 1: code occupancy mapping algorithm 
%    Question 2: see ass3_q2.m
%
% Fill in the required sections of this script with your code, run it to
% generate the requested plot/movie, then paste the plots into a short report
% that includes a few comments about what you've observed.  Append your
% version of this script to the report.  Hand in the report as a PDF file
% and the two resulting AVI files from Questions 1 and 2.
%
% requires: basic Matlab, 'gazebo.mat'
%
% T D Barfoot, January 2016
%
clear all; close all; clc;

% set random seed for repeatability
rng(1);

% ==========================
% load the dataset from file
% ==========================
%
%    ground truth poses: t_true x_true y_true theta_true
% odometry measurements: t_odom v_odom omega_odom
%           laser scans: t_laser y_laser
%    laser range limits: r_min_laser r_max_laser
%    laser angle limits: phi_min_laser phi_max_laser
%
load gazebo.mat;

% =======================================
% Question 1: build an occupancy grid map
% =======================================
%
% Write an occupancy grid mapping algorithm that builds the map from the
% perfect ground-truth localization.  Some of the setup is done for you
% below.  The resulting map should look like "ass2_q1_soln.png".  You can
% watch the movie "ass2_q1_soln.mp4" to see what the entire mapping process
% should look like.  At the end you will save your occupancy grid map to
% the file "occmap.mat" for use in Question 2 of this assignment.

% allocate a big 2D array for the occupancy grid
ogres = 0.05;                   % resolution of occ grid
ogxmin = -7;                    % minimum x value
ogxmax = 8;                     % maximum x value
ogymin = -3;                    % minimum y value
ogymax = 6;                     % maximum y value
ognx = (ogxmax-ogxmin)/ogres;   % number of cells in x direction
ogny = (ogymax-ogymin)/ogres;   % number of cells in y direction
oglo = zeros(ogny,ognx);        % occupancy grid in log-odds format
ogp = zeros(ogny,ognx);         % occupancy grid in probability format

% precalculate some quantities
numodom = size(t_odom,1);
npoints = size(y_laser,2);
angles = linspace(phi_min_laser, phi_max_laser,npoints);
dx = ogres*cos(angles);
dy = ogres*sin(angles);

% interpolate the noise-free ground-truth at the laser timestamps
t_interp = linspace(t_true(1),t_true(numodom),numodom);
x_interp = interp1(t_interp,x_true,t_laser);
y_interp = interp1(t_interp,y_true,t_laser);
theta_interp = interp1(t_interp,theta_true,t_laser);
omega_interp = interp1(t_interp,omega_odom,t_laser);
  
% set up the plotting/movie recording
vid = VideoWriter('ass2_q1.avi');
open(vid);
figure(1);
clf;
pcolor(ogp);
colormap(1-gray);
shading('flat');
axis equal;
axis off;
M = getframe;
writeVideo(vid,M);

% precalculate some quantities
cos_angles = cos(angles);
sin_angles = sin(angles);
offset = [0.1; 0];
mapO = [ogxmin; ogymin];

alpha = 2.5;
beta = 0.115;

% loop over laser scans (every fifth)
for i=1:5:size(t_laser,1)
    
    % ------insert your occupancy grid mapping algorithm here------

        if abs(omega_interp(i)) <= 0.1
    
            % extract robot pose in 2D space
            thetaR = theta_interp(i);
            ptR = [x_interp(i); y_interp(i)];
            R_i = [cos(-thetaR), sin(-thetaR);
                   -sin(-thetaR), cos(-thetaR)];
            
            % transfrom all laser scans to robot's local frame in one go
            RLaser_i = y_laser(i, :);
            QLaser_i = [RLaser_i.*cos_angles; RLaser_i.*sin_angles];
        
            QGlobal = R_i*(QLaser_i - offset) + ptR - mapO;
        
            % parse entries 1-by-1 to update the log likelihoods 
            for j = 1 : npoints
        
                % skip NaN data entries
                if isnan(RLaser_i(j))
                    continue
                end
        
                % check ranges to accumulate log likely hoods
                for r = r_min_laser : ogres : r_max_laser
        
                    if r < RLaser_i(j)
        
                        % transform range in global frame
                        rG = ptR + R_i*(r*[cos_angles(j); sin_angles(j)] - offset) - mapO;
        
                        % reward empty region
                        oglo(floor(rG(2)/ogres), floor(rG(1)/ogres)) = oglo(floor(rG(2)/ogres), floor(rG(1)/ogres)) - beta;
        
                    end
        
                end
    
                % penalize occupied region
                oglo(floor(QGlobal(2,j)/ogres), floor(QGlobal(1,j)/ogres)) = oglo(floor(QGlobal(2, j)/ogres), floor(QGlobal(1,j)/ogres)) + alpha;
    
            end
        end
    % convert log probabilities to probabilities
    for ogpX = 1:size(ogp, 1)
        for ogpY = 1:size(ogp, 2)
            ogp(ogpX, ogpY) = exp(oglo(ogpX, ogpY)) / (1 + exp(oglo(ogpX, ogpY)));
        end
    end    
    
    % ------end of your occupancy grid mapping algorithm-------

    % draw the map
    clf;
    pcolor(ogp);
    colormap(1-gray);
    shading('flat');
    axis equal;
    axis off;
    
    % draw the robot
    hold on;
    x = (x_interp(i)-ogxmin)/ogres;
    y = (y_interp(i)-ogymin)/ogres;
    th = theta_interp(i);
    r = 0.15/ogres;
    set(rectangle( 'Position', [x-r y-r 2*r 2*r], 'Curvature', [1 1]),'LineWidth',2,'FaceColor',[0.35 0.35 0.75]);
    set(plot([x x+r*cos(th)]', [y y+r*sin(th)]', 'k-'),'LineWidth',2);
    
    % save the video frame
    M = getframe;
    writeVideo(vid,M);
    
    pause(0.1);
    
end

close(vid);
print -dpng ass2_q1.png

save occmap.mat ogres ogxmin ogxmax ogymin ogymax ognx ogny oglo ogp;

