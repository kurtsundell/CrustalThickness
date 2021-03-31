% Supplemental code for "Jurassic to Neogene Quantitative Crustal Thickness Estimates in Southern Tibet" by Sundell et al. (in revision) GSA Today

clear all
close all
clc

rng(0,'twister'); % make random simulations reproducible

% Data from Profeta et al. (2015). Rmv lower sample with huge uncertainties from Zellmer (2008) --> produces artifacts bc the log fit vals < 1
SrY = [11.4, 15.6, 24.8, 56.5, 31.4, 11.7, 8.7, 15.5, 9.9, 11.8, 24.0, 9.4, 21.6, 29.6, 9.5, 20.1, 43.4, 18.2, 4.5, 17.9, 11.9, 15.7, 20.7;...
	2.3, 2.6, 9.4, 18.5, 5.1, 4.0, 1.3, 4.7, 1.6, 1.4, 10.3, 0.3, 4.5, 0.6, 1.1, 5.6, 11.2, 3.5, 0.3, 4.8, 2.8, 2.9, 7.7]';
LaYb = [2.8, 6.0, 7.7, 22.6, 6.5, 3.7, 1.4, 3.7, 2.0, 3.1, 5.6, 2.2, 7.9, 4.7, 2.5, 3.6, 9.5, 2.8, 1.3, 4.9, 2.4, 4.4, 6.5;...
	0.2, 2.1, 2.4, 9.5, 0.7, 0.5, 0.5, 0.3, 0.3, 0.5, 1.7, 0.1, 1.3, 0.1, 0.5, 0.9, 3.0, 0.4, 0.5, 1.2, 1.1, 1.1, 2.0]';
Km = [18.9, 28.0, 38.0, 65.0, 44.0, 30.2, 20.5, 34.6, 18.3, 24.7, 30.8, 14.5, 37.2, 40.0, 27.1, 25.0, 54.4, 24.5, 11.8, 26.2, 27.4, 27.8, 40.0;...
	4.4, 7.0, 1.9, 0.7, 3.0, 3.6, 2.7, 4.8, 0.9, 0.7, 1.4, 1.0, 9.0, 3.0, 4.3, 3.0, 4.7, 3.4, 0.1, 1.2, 2.2, 1.8, 5.0]';

%x1 = log(SrY(:,1));
%x2 = log(LaYb(:,1)); 
%y = Km(:,1); % Crustal thickness
%X = [ones(size(x1)) x1 x2 x1.*x2];


%surffit1 = fit(log([Xr(:,t),Yr(:,t)]),Zr(:,t) ,'poly11'); %linear fit of surface

SrYLaYbfit = fit(log([SrY(:,1),LaYb(:,1)]),Km(:,1),'poly11') % print fit and 95% confidence intervals at 95%
p00 = SrYLaYbfit.p00; %coeff
p10 = SrYLaYbfit.p10; %coeff
p01 = SrYLaYbfit.p01; %coeff
SrYLaYbf = p00 + p10.*log(SrY(:,1)) + p01.*log(LaYb(:,1)); % surfact fit equation z = p00 + p10*x + p01*y
residuals_SrYLaYb = Km(:,1) - SrYLaYbf; %calculate residual
residuals = 2*std(residuals_SrYLaYb) % 2s residuals


%[b,bint,r,rint,stats] = regress(y,X)
%coeffs = b %print coefficients
%coeff_unc = (bint(:,2) - bint(:,1))/2 % 2s coefficient 95% confidence intervals
%residuals = 2*std(r) % 2s residuals
%R_squared = stats(1,1)

figure
hold on
X_uh = SrY(:,1) + SrY(:,2); % 2s
X_ul = SrY(:,1) - SrY(:,2); % 2s
Y_uh = LaYb(:,1) + LaYb(:,2); % 2s
Y_ul = LaYb(:,1) - LaYb(:,2); % 2s
Z_uh = Km(:,1) + Km(:,2); % 2s
Z_ul = Km(:,1) - Km(:,2); % 2s
plot3(log([X_ul,X_uh])', log([LaYb(:,1),LaYb(:,1)])', [Km(:,1),Km(:,1)]', '-k', 'LineWidth', 3) % uncertainty in x direction
plot3(log([SrY(:,1),SrY(:,1)])', log([Y_ul,Y_uh])', [Km(:,1),Km(:,1)]', '-k', 'LineWidth', 3) % uncertainty in y direction
plot3(log([SrY(:,1),SrY(:,1)])', log([LaYb(:,1),LaYb(:,1)])', [Z_ul,Z_uh]', '-k', 'LineWidth', 3) % uncertainty in z direction


x = log(SrY(:,1));
y = log(LaYb(:,1));
z = SrYLaYbfit(x,y);
B = [x(:) y(:) ones(size(x(:)))] \ z(:);
xv = linspace(min(x), max(x), 2)';
yv = linspace(min(y), max(y), 2)';
[X,Y] = meshgrid(xv, yv);
Z = reshape([X(:), Y(:), ones(size(X(:)))] * B, numel(xv), []);
colormap([0  0  1])
surf(X, Y, Z, 'FaceAlpha', 0.5)
shading interp

%x1fit = min(x1):.1:max(x1);
%x2fit = min(x2):.1:max(x2);
%[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
%YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
%colormap([0  0  1])
%surf(X1FIT,X2FIT,YFIT, 'FaceAlpha', 0.5)
%shading interp

scatter3(log(SrY(:,1)),log(LaYb(:,1)),Km(:,1),120,'r', 'o','filled')
view([295 35])
xlim([1 4.5])
ylim([0 3.5])
zlim([0 70])
grid on
xlabel('ln(Sr/Y)')
ylabel('ln(La/Yb)')
zlabel('Crustal Thickness (km)')
set(gcf, 'Position', [391, 495, 1065, 498])
% Crustal_Thickness = -10.6 + 10.3*ln(Sr/Y) +8.8*ln(La/Yb)
%Check the r^2
r = residuals_SrYLaYb;
normr = norm(r); % Euclidean norm (i.e., 2-norm, the sqrt of the sum of the squares... normr = sqrt(sum(r.^2)))
SSE = normr.^2;              % Error sum of squares.
TSS = norm(Km(:,1)-mean(Km(:,1)))^2;     % Total sum of squares.
r2 = 1 - SSE/TSS            % R-square statistic.


%% linear least squares ln(Sr/Y)
xax = 1:1:100; %make x values for plotting
[SrYp, SrYs] = polyfit(log(SrY(:,1)),Km(:,1),1); % linear fit
[SrYax,SrYc] = polyval(SrYp,log(xax),SrYs); % calculate y values along x axis to make lines for plot
SrYf = polyval(SrYp,log(SrY(:,1))); % calculate y values for individual estimates based on random selections
residuals_SrY = 2*std(Km(:,1) - SrYf); % 2s residuals	
figure
hold on
plot(log([SrY(:,1)'; SrY(:,1)']), [(Km(:,1)+Km(:,2))'; (Km(:,1)-Km(:,2))'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot(log([(SrY(:,1)+SrY(:,2))'; (SrY(:,1)-SrY(:,2))']), [Km(:,1)'; Km(:,1)'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
scatter(log(SrY(:,1)),Km(:,1),120,'r', 'o','filled','markeredgecolor','k')
plot(log(xax),SrYax,'b','LineWidth',2)
plot(log(xax),SrYax+2*SrYc,'b--',log(xax),SrYax-2*SrYc,'b--')
xlim([1 4.5])
ylim([0 70])
xlabel('ln(Sr/Y)')
ylabel('Crustal Thickness (km)')
%Crustal_Thickness = 19.6*ln(Sr/Y) - 24.0
SrYfit = fit(log(SrY(:,1)),Km(:,1),'poly1') % print fit and 95% confidence intervals at 95%
R_squared_SrY = 1 - (SrYs.normr/norm(Km(:,1) - mean(Km(:,1))))^2 %calculate r squared for each random selection
% test weighted by uncertainties in x and y --> york fit
%{
[a_york, b_york, sigma_ayork, sigma_byork] = york_fit(log(SrY(:,1))',Km(:,1)',[abs(log(SrY(:,2)))]',Km(:,2)',[0*ones(1,23)])
yy = b_york*log(xax) + a_york;
plot(log(xax),yy,'--','linewidth',3,'color','k')
%}
%Check the r^2
xcov = log(SrY(:,1)) - mean(log(SrY(:,1)));
ycov = Km(:,1) - mean(Km(:,1));
numerator = sum(xcov.*ycov);
denominator = sqrt(sum(xcov.*xcov).*sum(ycov.*ycov));
r = numerator/denominator;
r2_SrY = r*r
%Another way to test r^2
r = Km(:,1) - SrYf;
y = Km(:,1);
normr = norm(r); % Euclidean norm (i.e., 2-norm, the sqrt of the sum of the squares... normr = sqrt(sum(r.^2)))
SSE = normr.^2;              % Error sum of squares.
TSS = norm(y-mean(y))^2;     % Total sum of squares.
r2_SrY = 1 - SSE/TSS            % R-square statistic.


%% linear least squares ln(La/Yb)
xax = 1:1:100; %make x values for plotting
[LaYbp, LaYbs] = polyfit(log(LaYb(:,1)),Km(:,1),1); % linear fit
[LaYbax,LaYbc] = polyval(LaYbp,log(xax),LaYbs); % calculate y values along x axis to make lines for plot
LaYbf = polyval(LaYbp,log(LaYb(:,1))); % calculate y values for individual estimates based on random selections
residuals_LaYb = 2*std(Km(:,1) - LaYbf); % 2s residuals	
figure
hold on
plot(log([LaYb(:,1)'; LaYb(:,1)']), [(Km(:,1)+Km(:,2))'; (Km(:,1)-Km(:,2))'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
plot(log([(LaYb(:,1)+LaYb(:,2))'; (LaYb(:,1)-LaYb(:,2))']), [Km(:,1)'; Km(:,1)'], '-k', 'Color', 'k', 'LineWidth',2) % 2s error bars
scatter(log(LaYb(:,1)),Km(:,1),120,'r', 'o','filled','markeredgecolor','k')
plot(log(xax),LaYbax,'b','LineWidth',2)
plot(log(xax),LaYbax+2*LaYbc,'b--',log(xax),LaYbax-2*LaYbc,'b--')
xlim([0 3.5])
ylim([0 70])
xlabel('ln(La/Yb)')
ylabel('Crustal Thickness (km)')
%Crustal_Thickness = 17.0*ln(La/Yb) + 6.9 
LaYbfit = fit(log(LaYb(:,1)),Km(:,1),'poly1') % print fit and 95% confidence intervals at 95%
R_squared_LaYb = 1 - (LaYbs.normr/norm(Km(:,1) - mean(Km(:,1))))^2 %calculate r squared for each random selection
% test weighted by uncertainties in x and y --> york fit (York, 2004) 
%{
[a_york, b_york, sigma_ayork, sigma_byork] = york_fit(log(LaYb(:,1))',Km(:,1)',[abs(log(LaYb(:,2)))]',Km(:,2)',[0*ones(1,23)])
yy = b_york*log(xax) + a_york;
plot(log(xax),yy,'--','linewidth',3,'color','k')
%}
%Check the r^2
xcov = log(LaYb(:,1)) - mean(log(LaYb(:,1)));
ycov = Km(:,1) - mean(Km(:,1));
numerator = sum(xcov.*ycov);
denominator = sqrt(sum(xcov.*xcov).*sum(ycov.*ycov));
r = numerator/denominator;
r2_LaYb = r*r
%Another way to test r^2
r = Km(:,1) - LaYbf;
y = Km(:,1);
normr = norm(r); % Euclidean norm (i.e., 2-norm, the sqrt of the sum of the squares... normr = sqrt(sum(r.^2)))
SSE = normr.^2;              % Error sum of squares.
TSS = norm(y-mean(y))^2;     % Total sum of squares.
r2_LaYb = 1 - SSE/TSS            % R-square statistic.


