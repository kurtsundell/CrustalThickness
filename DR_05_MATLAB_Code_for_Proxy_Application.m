clear all
close all
clc

rng('default')

Age = [10.6,13.0,13.2,14.3,16.4,16.6,16.7,17.0,17.0,17.0,17.0,17.0,17.0,18.0,18.0,18.0,18.0,19.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.6,29.8,30.0,30.0,30.0,30.0,30.0,30.0,...
	30.0,30.0,30.0,30.2,31.0,31.0,31.0,31.0,41.0,41.0,41.0,41.0,41.0,41.2,41.2,42.7,44.0,44.0,44.0,44.0,44.0,44.0,44.0,47.8,47.8,47.8,48.0,48.0,48.0,48.0,48.2,48.6,48.9,49.0,49.0,49.0,...
	49.0,49.0,49.0,49.0,49.0,49.0,49.1,49.7,49.7,49.9,50.0,50.0,50.2,50.6,50.9,51.1,51.4,51.5,52.0,52.0,52.0,52.0,52.0,52.0,52.9,53.4,53.9,54.0,54.0,55.3,55.4,55.5,56.3,60.0,60.6,60.6,...
	60.6,60.7,60.7,60.7,60.8,61.0,61.3,61.3,61.3,61.5,61.5,61.5,61.5,61.5,61.5,61.7,62.0,62.0,62.0,62.0,62.0,62.0,62.0,62.0,62.0,62.5,62.5,62.5,64.8,67.5,76.5,76.5,76.5,76.5,76.5,76.5,...
	87.0,87.0,87.0,88.0,88.0,88.4,91.1,91.1,91.1,91.1,91.1,91.3,91.8,91.8,92.0,92.0,92.0,93.3,94.0,95.0,95.0,95.0,95.0,95.4,95.4,95.4,95.4,95.4,95.4,96.0,96.0,96.0,96.0,96.0,136.5,168.0,....
	168.0,170.0,174.0,174.0,174.0,176.0,176.0,176.0,176.0,176.2,178.0]';
SrY = [68.3,124.6,117.4,91.0,187.7,140.9,104.5,151.3,138.3,134.3,127.8,100.8,132.5,136.1,151.0,121.3,131.1,150.5,108.9,127.9,82.8,126.3,79.3,71.7,83.5,72.5,72.6,66.1,95.4,108.9,127.9,82.8,...
	126.3,79.3,71.7,83.5,72.6,72.6,89.3,81.6,89.3,74.5,80.3,133.5,115.8,62.6,69.1,63.2,133.2,123.2,55.2,11.8,20.1,20.4,42.6,39.0,14.2,20.9,31.6,29.0,21.2,24.5,35.7,22.5,19.7,32.5,43.7,44.3,...
	36.7,42.7,44.4,40.3,44.1,44.3,38.9,30.9,29.5,37.0,25.6,24.5,26.1,37.0,39.6,19.9,28.6,17.4,40.3,22.1,29.8,33.6,24.2,34.6,29.3,30.3,111.6,37.2,15.7,33.4,32.4,32.4,29.5,15.9,27.9,37.2,25.0,...
	37.0,35.9,31.6,23.1,24.2,18.3,13.0,15.7,24.7,32.3,29.0,34.5,36.2,23.8,33.6,39.1,34.5,22.2,19.0,19.6,18.0,24.6,16.8,34.3,16.0,31.3,29.7,22.7,24.9,5.5,41.6,23.7,28.3,25.6,32.0,24.4,21.5,28.1,...
	130.4,45.8,43.6,52.4,37.4,43.0,148.2,86.2,20.9,23.0,56.5,44.8,60.5,59.3,118.7,48.2,39.6,44.3,40.2,26.3,35.5,59.2,62.7,60.8,67.6,59.7,62.1,56.1,55.0,61.3,86.1,76.8,61.2,51.4,56.3,13.1,15.4,...
	14.7,15.3,19.1,18.0,23.8,23.3,18.3,14.1,22.4,17.5]';
LaYb = [53.3,41.9,21.5,37.9,38.6,32.9,14.8,23.7,23.0,22.2,22.9,18.5,22.7,21.3,22.6,21.2,22.7,21.6,31.3,25.6,42.0,25.4,40.6,38.1,39.5,39.3,41.1,47.7,41.9,31.3,25.6,42.0,25.4,40.6,38.1,39.5,39.3,...
	41.1,34.5,39.9,40.1,38.5,43.3,37.0,36.5,39.5,38.6,38.1,40.0,40.0,35.5,3.5,4.7,5.9,13.1,13.4,4.1,10.6,9.9,9.6,9.8,7.0,7.0,7.5,6.0,19.5,8.1,8.1,7.4,8.3,8.8,8.8,8.5,8.3,7.5,6.8,5.4,8.2,7.3,...
	6.4,8.1,8.4,10.4,5.0,9.7,5.1,8.1,9.4,6.6,5.8,5.2,5.9,6.1,6.5,16.4,12.8,3.9,8.7,5.9,8.6,10.0,3.2,6.1,7.5,9.0,13.6,7.4,8.4,5.2,5.4,6.5,2.9,5.6,4.2,11.2,7.6,6.2,6.4,4.0,6.6,6.8,7.3,4.2,6.8,...
	7.5,6.8,6.8,6.8,14.2,7.5,6.9,7.3,7.2,6.9,11.5,9.3,7.6,5.4,5.2,6.1,5.4,4.8,6.0,21.7,21.7,21.2,10.8,10.8,10.3,24.9,20.0,4.7,5.1,19.8,13.4,12.3,14.9,28.4,13.2,13.3,13.6,7.7,8.7,9.0,16.3,...
	18.0,15.4,17.5,16.6,17.5,16.0,17.7,19.3,20.0,16.9,14.5,13.2,16.5,3.7,3.7,6.0,5.3,6.6,6.4,5.8,6.0,5.5,5.1,6.8,4.4]';

SrYT = 19.6.*log(SrY) - 24.0; % Sr/Y from proxy calibration
LaYbT = 17.0.*log(LaYb) + 6.9; % La/Yb from proxy calibration
SrYLaYbT = -10.6 + 10.3.*log(SrY) + 8.8.*log(LaYb);

Age_SrYLaYbT = [Age, SrYLaYbT];

Ma = 5; % 2s unc x dir
km = 10; % 2s unc y dir based on residuals

boots = 10000; % number of bootstrap resamples 

% Figure 3A Plot the filtered raw data
Figure_03A = figure;
hold on
yyaxis left
scatter(Age,SrY,2000,'r', 'o','filled')
xlabel('Age (Ma)')
ylabel('Sr/Y')
xticks(0:25:200)
set(gca, 'FontSize', 75)
yyaxis right
scatter(Age,LaYb,3000,'b', 's','filled')
ylabel('La/Yb')
set(gca, 'FontSize', 75)
yticks(0:10:60)

% Figure 3B Plot filtered data converted to crustal thickness
Figure_03B = figure;
hold on
scatter(Age,SrYT,2000,'r', 'o','filled')
scatter(Age,LaYbT,3000,'b', 's','filled')
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
scatter(Age,SrYLaYbT,1500,'y', 'd','filled','markeredgecolor','k')
ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

% Figure 3C Plot trends in age vs crustal thickness
Figure_03C = figure;
hold on
h = 5; % kernel bandwidth for Gaussian smoothing
for j = 1:boots
	bootsamp = datasample(Age_SrYLaYbT,length(Age_SrYLaYbT(:,1)),'Replace',true);
	
	for i = 1:length(bootsamp(:,1))
		xs(i) = i;
		ys(i) = gaussian_kern_reg(xs(i),bootsamp(:,1),bootsamp(:,2),h); %apply Gaussian kernel regression
	end
	if j <= 500
		plot(xs,ys,'k-');
	end
	ct_out(j,:) = ys; %calculate crustal thickness
end
ct_mean = mean(ct_out); %median of crustal thickness results for plot
ct_std = 2*std(ct_out); % 2s
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
scatter(Age,SrYLaYbT,1500,'y', 'd','filled','markeredgecolor','k')
plot(xs,ct_mean,'LineWidth',4,'Color','g')
plot(xs,ct_mean+ct_std,'LineWidth',2,'Color','g')
plot(xs,ct_mean-ct_std,'LineWidth',2,'Color','g')
ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

% Figure 3D Calculate and plot linear fits for specific time intervals 
Figure_03D = figure;
hold on
xax_J = [162:1:183]';
xax_K = [65:1:105]';
xax_C = [25:1:65]';
% Some bootstrap resamples only have few data points so this supresses printing the warnings in the command window
%w = warning('query','last');
%id = w.identifier;
%warning('off',id)
for j = 1:boots
	bootsamp = datasample(Age_SrYLaYbT,length(Age_SrYLaYbT(:,1)),'Replace',true);
	for i = 1:length(bootsamp(:,1))
		if bootsamp(i,1) > 150 % for linear fit
			Age_J(i,1) = bootsamp(i,1);
			ct_J(i,1) = bootsamp(i,2);
		end	
		if bootsamp(i,1) < 100 && bootsamp(i,1) > 65 % for linear fit
			Age_K(i,1) = bootsamp(i,1);
			ct_K(i,1) = bootsamp(i,2);
		end		
		if bootsamp(i,1) < 60 && bootsamp(i,1) > 30 % for linear fit
			Age_C(i,1) = bootsamp(i,1);
			ct_C(i,1) = bootsamp(i,2);
		end	
	end
	Age_J = Age_J(any(Age_J ~= 0,2),:);
	ct_J = ct_J(any(ct_J ~= 0,2),:);
	[pAge_J(j,:)] = polyfit(Age_J,ct_J,1);
	fAge_J(:,j) = polyval(pAge_J(j,:),xax_J);
	Age_K = Age_K(any(Age_K ~= 0,2),:);
	ct_K = ct_K(any(ct_K ~= 0,2),:);
	[pAge_K(j,:)] = polyfit(Age_K,ct_K,1);
	fAge_K(:,j) = polyval(pAge_K(j,:),xax_K);	
	Age_C = Age_C(any(Age_C ~= 0,2),:);
	ct_C = ct_C(any(ct_C ~= 0,2),:);
	[pAge_C(j,:)] = polyfit(Age_C,ct_C,1);
	fAge_C(:,j) = polyval(pAge_C(j,:),xax_C);	
	if j <= 500
		plot(xax_J,fAge_J(:,j),'k-')
		plot(xax_K,fAge_K(:,j),'k-')
		plot(xax_C,fAge_C(:,j),'k-')
	end
end
pAge_J_mean = mean(pAge_J);	
pAge_J_std = std(pAge_J);	
pAge_K_mean = mean(pAge_K);	
pAge_K_std = std(pAge_K);	
pAge_C_mean = mean(pAge_C);	
pAge_C_std = std(pAge_C);	
fpAge_J_mean = polyval(pAge_J_mean,xax_J);
fpAge_K_mean = polyval(pAge_K_mean,xax_K);
fpAge_C_mean = polyval(pAge_C_mean,xax_C);
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot(xax_J, fpAge_J_mean,'c','linewidth',4)
plot(xax_K, fpAge_K_mean,'c','linewidth',4)
plot(xax_C, fpAge_C_mean,'c','linewidth',4)
scatter(Age,SrYLaYbT,1500,'y', 'd','filled','markeredgecolor','k')
ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

%% Gaussian function needed to run code
% Gaussian kernel regression model
% http://youngmok.com/Blog_Data/gaussian_kern_reg.zip
% http://youngmok.com/gaussian-kernel-regression-with-matlab-code/
function ys = gaussian_kern_reg(xs,x,y,h)
% Gaussian kernel function
kerf = @(z)exp(-z.*z/2)/sqrt(2*pi);
z = kerf((xs-x)/h);
ys = sum(z.*y)/sum(z);
end