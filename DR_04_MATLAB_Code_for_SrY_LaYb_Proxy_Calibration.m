clear all
close all
clc

rng(0,'twister'); % make raandom simulations reproducible

%Data from Profeta et al. (2015). Rmv lower sample with huge uncertainties from Zellmer (2008) --> produces artifacts in the MC bc the log fit vals < 1
SrY = [11.4000000000000,2.30000000000000,18.9000000000000,4.40000000000000;15.6000000000000,2.60000000000000,28,7;24.8000000000000,9.40000000000000,38,1.90000000000000;56.5000000000000,18.5000000000000,65,0.700000000000000;31.4000000000000,5.10000000000000,44,3;11.7000000000000,4,30.2000000000000,3.60000000000000;8.70000000000000,1.30000000000000,20.5000000000000,2.70000000000000;15.5000000000000,4.70000000000000,34.6000000000000,4.80000000000000;9.90000000000000,1.60000000000000,18.3000000000000,0.900000000000000;11.8000000000000,1.40000000000000,24.7000000000000,0.700000000000000;24,10.3000000000000,30.8000000000000,1.40000000000000;9.40000000000000,0.300000000000000,14.5000000000000,1;21.6000000000000,4.50000000000000,37.2000000000000,9;29.6000000000000,0.600000000000000,40,3;9.50000000000000,1,27.1000000000000,4.30000000000000;20.1000000000000,5.60000000000000,25,3;43.4000000000000,11.2000000000000,54.4000000000000,4.70000000000000;18.2000000000000,3.50000000000000,24.5000000000000,3.40000000000000;4.50000000000000,0.300000000000000,11.8000000000000,0.100000000000000;17.9000000000000,4.80000000000000,26.2000000000000,1.20000000000000;11.9000000000000,2.80000000000000,27.4000000000000,2.20000000000000;15.7000000000000,2.90000000000000,27.8000000000000,1.80000000000000;20.7000000000000,7.70000000000000,40,5];
LaYb = [2.80000000000000,0.170000000000000,18.9000000000000,4.40000000000000;5.96000000000000,2.13000000000000,28,7;7.70000000000000,2.37000000000000,38,1.90000000000000;22.5700000000000,9.51000000000000,65,0.700000000000000;6.49000000000000,0.740000000000000,44,3;3.70000000000000,0.480000000000000,30.2000000000000,3.60000000000000;1.43000000000000,0.450000000000000,20.5000000000000,2.70000000000000;3.71000000000000,0.330000000000000,34.6000000000000,4.80000000000000;1.95000000000000,0.340000000000000,18.3000000000000,0.900000000000000;3.13000000000000,0.490000000000000,24.7000000000000,0.700000000000000;5.55000000000000,1.69000000000000,30.8000000000000,1.40000000000000;2.18000000000000,0.0500000000000000,14.5000000000000,1;7.85000000000000,1.30000000000000,37.2000000000000,9;4.73000000000000,0.140000000000000,40,3;2.54000000000000,0.540000000000000,27.1000000000000,4.30000000000000;3.61000000000000,0.850000000000000,25,3;9.54000000000000,2.95000000000000,54.4000000000000,4.70000000000000;2.78000000000000,0.390000000000000,24.5000000000000,3.40000000000000;1.32000000000000,0.510000000000000,11.8000000000000,0.100000000000000;4.93000000000000,1.23000000000000,26.2000000000000,1.20000000000000;2.35000000000000,1.10000000000000,27.4000000000000,2.20000000000000;4.42000000000000,1.03000000000000,27.8000000000000,1.80000000000000;6.48000000000000,2,40,5];

SrYx = SrY(:,1);
SrYxu = SrY(:,2); % 2 sigma
SrYy = SrY(:,3);
SrYyu = SrY(:,4); % 2 sigma

LaYbx = LaYb(:,1);
LaYbxu = LaYb(:,2); % 2 sigma
LaYby = LaYb(:,3);
LaYbyu = LaYb(:,4); % 2 sigma

KM = SrY(:,3); %same for both Sr/Y and La/Yb pairs of input data
KMu = SrY(:,4); %same for both Sr/Y and La/Yb pairs of input data

trials = 1000; % number of random simulations (100,000 used in manuscript)

%% random numbers for Sr/Y and La/Yb, filtered as in Profeta et al. (2015), complete database downloaded July 20, 2020
for i = 1:length(SrYx)
	r1 = randn(trials*2,1);
	r1 = r1(r1>-2 & r1<2); %trim outside 2 sigma
	r1 = r1(1:trials);
	r2 = randn(trials*2,1);
	r2 = r2(r2>-2 & r2<2); %trim outside 2 sigma
	r2 = r2(1:trials);
	r3 = randn(trials*2,1);
	r3 = r3(r3>-2 & r3<2); %trim outside 2 sigma
	r3 = r3(1:trials);
	r4 = randn(trials*2,1);
	r4 = r4(r4>-2 & r4<2); %trim outside 2 sigma
	r4 = r4(1:trials);
	r5 = randn(trials*2,1);
	r5 = r5(r5>-2 & r5<2); %trim outside 2 sigma
	r5 = r5(1:trials);
	r6 = randn(trials*2,1);
	r6 = r6(r6>-2 & r6<2); %trim outside 2 sigma
	r6 = r6(1:trials);
	r7 = randn(trials*2,1);
	r7 = r7(r7>-2 & r7<2); %trim outside 2 sigma
	r7 = r7(1:trials);
	for t = 1:trials
		SrYxR(i,t) = SrYx(i,1) + SrYxu(i,1)./2 * r1(t,1); % divide by 2 before multiplying by randn between -2 and 2 --> results in random 2 sigma dist
		SrYyR(i,t) = SrYy(i,1) + SrYyu(i,1)./2 * r2(t,1);
		LaYbxR(i,t) = LaYbx(i,1) + LaYbxu(i,1)./2 * r3(t,1); % divide by 2 before multiplying by randn between -2 and 2 --> results in random 2 sigma dist
		LaYbyR(i,t) = LaYby(i,1) + LaYbyu(i,1)./2 * r4(t,1);
		Xr(i,t) = SrYx(i,1) + SrYxu(i,1)./2 * r5(t,1); % divide by 2 before multiplying by randn between -2 and 2 --> results in random 2 sigma dist
		Yr(i,t) = LaYbx(i,1) + LaYbxu(i,1)./2 * r6(t,1); 
		Zr(i,t) = KM(i,1) + KMu(i,1)./2 * r7(t,1); % divide by 2 before multiplying by randn between -2 and 2 --> results in random 2 sigma dist		
	end
end

xax = 1:1:100; %make x values for plotting

%% 1st order poly fit Sr/Y
figure;
hold on
for t = 1:trials
	[pSrY1(t,:), S1] = polyfit(log(SrYxR(:,t)),SrYyR(:,t),1); % linear fit
	SrYf1(:,t) = polyval(pSrY1(t,:),log(xax)); % calculate y values along x axis to make lines for plot
	SrYf1R(:,t) = polyval(pSrY1(t,:),log(SrYxR(:,t))); % calculate y values for individual estimates based on random selections
	Resid_SrYT(:,t) = SrYyR(:,t) - SrYf1R(:,t); % residuals from random selection	
	if t <= 100 % only plot the first 100 trials or fewer
		scatter(log(SrYxR(:,t)),SrYyR(:,t),10,'r', 'o','filled')
		plot(log(xax),SrYf1(:,t),'b')
	end
	Rsquared1(t,1) = 1 - (S1.normr/norm(SrYyR(:,t) - mean(SrYyR(:,t))))^2; %calculate r squared for each random selection
end
pSrY1_med = median(pSrY1); %median of fit coefficients for all trials
pSrY1_std = std(pSrY1); %st dev of fit coefficients for all trials
pSrY1_medf1 = polyval(pSrY1_med,log(xax)); %evaluate median coefficients along x for plotting
plot(log(xax),pSrY1_medf1,'g','LineWidth',4)
scatter(log(SrYx),SrYy,120,'k', 'o','filled')
plot(log([SrYx'; SrYx']), [(SrYy+SrYyu)'; (SrYy-SrYyu)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot(log([(SrYx+SrYxu)'; (SrYx-SrYxu)']), [SrYy'; SrYy'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
xlim([1 4.5])
ylim([0 70])
xlabel('ln(Sr/Y)')
ylabel('Crustal Thickness (km)')
Rsquared1_med = median(Rsquared1); %median for all trials
Rsquared1_std = std(Rsquared1); %st dev for all trials
dim = [.45 .01 .3 .3]; %for location in plot
str = [strcat({'y ='}, {' '}, {'('}, {sprintf('%.1f', pSrY1_med(1,1))}, {' '}, {'±'}, {' '}, {sprintf('%.1f', 2*pSrY1_std(1,1))}, {')'}, {'x'}, {' '}, {'+'}, {' '},...
	{'('}, {sprintf('%.1f ', pSrY1_med(1,2))}, {' '}, {'±'}, {' '}, {sprintf('%.1f ', 2*pSrY1_std(1,2))}, {')'});...
	strcat({'R^2 = '}, {' '}, {sprintf('%.2f', Rsquared1_med)}, {' '}, {'±'}, {' '}, {sprintf('%.2f', 2*Rsquared1_std)}, {' '}, {'(2\sigma)'})];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('1st order polynomial fit')
Resid_SrYT_med = median(Resid_SrYT,'All')
Resid_SrYT_std = 2*std(Resid_SrYT,[],'All')

%% 1st order poly fit La/Yb
figure;
hold on
for t = 1:trials
	[pLaYb1(t,:), S1] = polyfit(log(LaYbxR(:,t)),LaYbyR(:,t),1); % linear fit
	LaYbf1(:,t) = polyval(pLaYb1(t,:),log(xax)); % calculate y values along x axis to make lines for plot
	LaYbf1R(:,t) = polyval(pLaYb1(t,:),log(LaYbxR(:,t))); % calculate y values for individual estimates based on random selections
	Resid_LaYbT(:,t) = LaYbyR(:,t) - LaYbf1R(:,t); % residuals from random selection
	if t <= 100 % only plot the first 100 trials or fewer
		scatter(log(LaYbxR(:,t)),LaYbyR(:,t),10,'r', 'o','filled')
		plot(log(xax),LaYbf1(:,t),'b')
	end
	Rsquared1(t,1) = 1 - (S1.normr/norm(LaYbyR(:,t) - mean(LaYbyR(:,t))))^2;
end
pLaYb1_med = median(pLaYb1); %median of fit coefficients for all trials
pLaYb1_std = std(pLaYb1); %st dev of fit coefficients for all trials
pLaYb1_medf1 = polyval(pLaYb1_med,log(xax)); %evaluate median coefficients along x for plotting
plot(log(xax),pLaYb1_medf1,'g','LineWidth',4)
scatter(log(LaYbx),LaYby,120,'k', 'o','filled')
plot(log([LaYbx'; LaYbx']), [(LaYby+LaYbyu)'; (LaYby-LaYbyu)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot(log([(LaYbx+LaYbxu)'; (LaYbx-LaYbxu)']), [LaYby'; LaYby'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
xlim([0 3.5])
ylim([0 70])
xlabel('ln(La/Yb)')
ylabel('Crustal Thickness (km)')
Rsquared1_med = median(Rsquared1); %median for all trials
Rsquared1_std = std(Rsquared1); %st dev for all trials
dim = [.45 .01 .3 .3]; %for location in plot
str = [strcat({'y ='}, {' '}, {'('}, {sprintf('%.1f', pLaYb1_med(1,1))}, {' '}, {'±'}, {' '}, {sprintf('%.1f', 2*pLaYb1_std(1,1))}, {')'}, {'x'}, {' '}, {'+'}, {' '},...
    {'('}, {sprintf('%.1f ', pLaYb1_med(1,2))}, {' '}, {'±'}, {' '}, {sprintf('%.1f ', 2*pLaYb1_std(1,2))}, {')'});...
    strcat({'R^2 = '}, {' '}, {sprintf('%.2f', Rsquared1_med)}, {' '}, {'±'}, {' '}, {sprintf('%.2f', 2*Rsquared1_std)}, {' '}, {'(2\sigma)'})];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('1st order polynomial fit')
Resid_LaYbT_med = median(Resid_LaYbT,'All')
Resid_LaYbT_std = 2*std(Resid_LaYbT,[],'All')

%% 1st order poly fit paired Sr/Y and La/Yb
F3 = figure;
hold on
for t = 1:trials
	surffit1 = fit(log([Xr(:,t),Yr(:,t)]),Zr(:,t) ,'poly11'); %linear fit of surface
	p00(t,1) = surffit1.p00; %coeff
	p10(t,1) = surffit1.p10; %coeff
	p01(t,1) = surffit1.p01; %coeff
	SrYLaYbf(:,t) = p00(t,1) + p10(t,1).*log(Xr(:,t)) + p01(t,1).*log(Yr(:,t)); % surfact fit equation z = p00 + p10*x + p01*y
	Resid_SrYLaYbT = Zr(:,t) - SrYLaYbf(:,t); %calculate residual
	if t <= 100 % only plot the first 100 trials or fewer
		scatter3(log(Xr(:,t)), log(Yr(:,t)), Zr(:,t), 10,'r', 'o','filled')
		x = log(Xr(:,t));
		y = log(Yr(:,t));
		z = surffit1(x,y);
		B = [x(:) y(:) ones(size(x(:)))] \ z(:);
		xv = linspace(min(x), max(x), 2)';
		yv = linspace(min(y), max(y), 2)';
		[X,Y] = meshgrid(xv, yv);
		Z = reshape([X(:), Y(:), ones(size(X(:)))] * B, numel(xv), []);
		colormap([0  0  1])
		surf(X, Y, Z, 'FaceAlpha', 0.05)
		shading interp
	end
end
X_uh = SrYx + SrYxu; % 2 sigma
X_ul = SrYx - SrYxu; % 2 sigma
Y_uh = LaYbx + LaYbxu; % 2 sigma
Y_ul = LaYbx - LaYbxu; % 2 sigma
Z_uh = KM + KMu; % 2 sigma
Z_ul = KM - KMu; % 2 sigma
scatter3(log(SrYx),log(LaYbx),KM,120,'b', 'o','filled')
plot3(log([X_ul,X_uh])', log([LaYbx,LaYbx])', [KM,KM]', '-k', 'LineWidth', 3) % uncertainty in x direction
plot3(log([SrYx,SrYx])', log([Y_ul,Y_uh])', [KM,KM]', '-k', 'LineWidth', 3) % uncertainty in y direction
plot3(log([SrYx,SrYx])', log([LaYbx,LaYbx])', [Z_ul,Z_uh]', '-k', 'LineWidth', 3) % uncertainty in z direction
view([295 35])
xlim([1 4.5])
ylim([0 3.5])
zlim([0 70])
grid on
xlabel('Sr/Y')
ylabel('La/Yb')
zlabel('Crustal Thickness')
p00_med = median(p00);
p10_med = median(p10);
p01_med = median(p01);
p00_std = 2*std(p00); % 2 sigma
p10_std = 2*std(p10); % 2 sigma
p01_std = 2*std(p01); % 2 sigma
dim = [.15 .01 .3 .3];
str = strcat({'y ='}, {' '}, {'('}, {sprintf('%.1f', p00_med)}, {' '}, {'±'}, {' '}, {sprintf('%.1f', p00_std)}, {')'}, {' '}, {'+'}, {' '},...
	{'('}, {sprintf('%.1f ', p10_med)}, {' '}, {'±'}, {' '}, {sprintf('%.1f ', p10_std)}, {')'}, {'x'}, {' '}, {'+'}, {' '},...
	{'('}, {sprintf('%.1f ', p01_med)}, {' '}, {'±'}, {' '}, {sprintf('%.1f ', p01_std)}, {')'}, {'y'}, {' '}, {'(2\sigma)'});
annotation('textbox',dim,'String',str,'FitBoxToText','on');
Resid_SrYLaYbT_med = median(Resid_SrYLaYbT,'All')
Resid_SrYLaYbT_std = 2*std(Resid_SrYLaYbT,[],'All')

%{
% Alternative method recommended by reviewer Dr. Allen Glazner and discussed in the manuscript main text

xax = 1:1000:10000000; % for plotting only

%% 1st order poly fit paired ln(Sr/Y)*ln(La/Yb)
F4 = figure;
hold on
for t = 1:trials
	[pSrYLaYb1(t,:), S1] = polyfit(log(SrYxR(:,t)).*log(LaYbxR(:,t)),LaYbyR(:,t),1);
	SrYLaYbf1(:,t) = polyval(pSrYLaYb1(t,:),log(xax));
	SrYLaYbf1R(:,t) = polyval(pSrYLaYb1(t,:),log(SrYxR(:,t)).*log(LaYbxR(:,t)));
	Resid_SrYLaYbT(:,t) = LaYbyR(:,t) - SrYLaYbf1R(:,t); % residuals from random selection
	if t <= 100 % only plot the first 100 trials or fewer
		scatter(log(SrYxR(:,t)).*log(LaYbxR(:,t)),LaYbyR(:,t),10,'r', 'o','filled')
		plot(log(xax),SrYLaYbf1(:,t),'b')
	end
	Rsquared1(t,1) = 1 - (S1.normr/norm(LaYbyR(:,t) - mean(LaYbyR(:,t))))^2;
end
Resid_SrYLaYbT_med = median(Resid_SrYLaYbT,'All')
Resid_SrYLaYbT_std = 2*std(Resid_SrYLaYbT,[],'All')
pSrYLaYb1_med = median(pSrYLaYb1);
pSrYLaYb1_std = std(pSrYLaYb1);
pSrYLaYb1_medf1 = polyval(pSrYLaYb1_med,log(xax));
plot(log(xax),pSrYLaYb1_medf1,'g','LineWidth',4)
scatter(log(SrYx).*log(LaYbx),LaYby,120,'k', 'o','filled')
plot([(log(SrYx).*log(LaYbx))';(log(SrYx).*log(LaYbx))'], [(LaYby+LaYbyu)'; (LaYby-LaYbyu)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot([log(SrYx+SrYxu)'.*log(LaYbx+LaYbxu)'; log(SrYx-SrYxu)'.*log(LaYbx-LaYbxu)'], [LaYby'; LaYby'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
xlim([0 15.2])
ylim([0 70])
xlabel('ln(Sr/Y)*ln(La/Yb)')
ylabel('Crustal Thickness (km)')
Rsquared1_med = median(Rsquared1);
Rsquared1_std = std(Rsquared1);
dim = [.45 .01 .3 .3];
str = [strcat({'y ='}, {' '}, {'('}, {sprintf('%.1f', pSrYLaYb1_med(1,1))}, {' '}, {'±'}, {' '}, {sprintf('%.1f', 2*pSrYLaYb1_std(1,1))}, {')'}, {'x'}, {' '}, {'+'}, {' '},...
    {'('}, {sprintf('%.1f ', pSrYLaYb1_med(1,2))}, {' '}, {'±'}, {' '}, {sprintf('%.1f ', 2*pSrYLaYb1_std(1,2))}, {')'});...
    strcat({'R^2 = '}, {' '}, {sprintf('%.2f', Rsquared1_med)}, {' '}, {'±'}, {' '}, {sprintf('%.2f', 2*Rsquared1_std)}, {' '}, {'(2\sigma)'});...
	strcat({'Residuals = '}, {' '}, {sprintf('%.2f', Resid_SrYLaYbT_med)}, {' '}, {'±'}, {' '}, {sprintf('%.2f', Resid_SrYLaYbT_std)}, {' '}, {'(2\sigma)'})];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('1st order polynomial fit')
%}

% Turn on for saving figures for AI, requires EPSCLEAN function file
%{
pause(15)
[file,path] = uiputfile('*.eps','Save file');
print(F3,'-depsc','-painters',[path file]);
epsclean([path file]);
%}
