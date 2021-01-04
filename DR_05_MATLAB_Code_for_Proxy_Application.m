clear all
close all
clc

Age = [10.6000000000000;13;13.2000000000000;14.3000000000000;16.4000000000000;16.6000000000000;16.7000000000000;17;17;17;17;17;17;18;18;18;18;19;29;29;29;29;29;29;29;29;29;29.6000000000000;29.8000000000000;30;30;30;30;30;30;30;30;30;30.2000000000000;31;31;31;31;41;41;41;41;41;41.2000000000000;41.2000000000000;42.6700000000000;44;44;44;44;44;44;44;47.8000000000000;47.8000000000000;47.8000000000000;48;48;48;48;48.2000000000000;48.6000000000000;48.9000000000000;49;49;49;49;49;49;49;49;49;49.1000000000000;49.6800000000000;49.7000000000000;49.9300000000000;50;50;50.2000000000000;50.6000000000000;50.9000000000000;51.1000000000000;51.4200000000000;51.5000000000000;52;52;52;52;52;52;52.9000000000000;53.4000000000000;53.9000000000000;54;54;55.3000000000000;55.4200000000000;55.5000000000000;56.3000000000000;60;60.6000000000000;60.6000000000000;60.6000000000000;60.7000000000000;60.7000000000000;60.7000000000000;60.8000000000000;61;61.3000000000000;61.3000000000000;61.3000000000000;61.5000000000000;61.5000000000000;61.5000000000000;61.5000000000000;61.5000000000000;61.5000000000000;61.7000000000000;62;62;62;62;62;62;62;62;62;62.5000000000000;62.5000000000000;62.5000000000000;64.7700000000000;67.4500000000000;76.5000000000000;76.5000000000000;76.5000000000000;76.5000000000000;76.5000000000000;76.5000000000000;87;87;87;88;88;88.4000000000000;91.1000000000000;91.1000000000000;91.1000000000000;91.1000000000000;91.1000000000000;91.3000000000000;91.8000000000000;91.8000000000000;92;92;92;93.3000000000000;94;95;95;95;95;95.4000000000000;95.4000000000000;95.4000000000000;95.4000000000000;95.4000000000000;95.4000000000000;96;96;96;96;96;136.500000000000;168;168;170;174;174;174;176;176;176;176;176.200000000000;178];
lat = [29.7136730000000;29.7400000000000;29.5416670000000;29.5833000000000;29.4800000000000;29.4800000000000;29.5416670000000;29.5861000000000;29.5861000000000;29.5861000000000;29.6100000000000;29.6282040000000;29.5861000000000;29.6240000000000;29.6240800000000;29.6232800000000;29.6239800000000;29.6282040000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2600000000000;29.2690000000000;29.2533333300000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2915160000000;29.2691700000000;29.2688900000000;29.2500000000000;29.2711100000000;29.2497200000000;29.3543450000000;29.3543450000000;29.3543450000000;29.3543450000000;29.3543450000000;29.3543450000000;29.3543450000000;29.6277777800000;29.6211300000000;29.6700000000000;29.6700000000000;29.6700000000000;29.6700000000000;29.6211300000000;29.6700000000000;29.7400000000000;29.7400000000000;29.7300000000000;29.3097200000000;29.5366666700000;29.3350000000000;29.8983333300000;29.5161260000000;29.3502270000000;29.4927420000000;29.4927420000000;29.4927420000000;29.4927420000000;29.4921960000000;29.4921960000000;29.4921960000000;29.4865620000000;29.4865620000000;29.4865620000000;29.4749850000000;29.3000000000000;29.4524420000000;29.3400000000000;29.4749850000000;29.4749850000000;29.3511100000000;29.4433300000000;29.5388900000000;29.3446388900000;29.4400000000000;29.3675000000000;29.3595000000000;29.4024722200000;29.3446388900000;29.3446388900000;29.3446388900000;29.3302777800000;29.4638900000000;29.4905600000000;30.0800000000000;29.2586111100000;29.2547222200000;29.4816700000000;29.2547222200000;29.3575000000000;29.3281000000000;29.5699520000000;29.2500000000000;29.2500000000000;29.2500000000000;29.5416670000000;29.5416670000000;29.5416670000000;30.0400870000000;30.0400870000000;29.2622200000000;29.2661100000000;29.2647200000000;29.3552800000000;29.3544400000000;29.3547200000000;29.3547200000000;29.3544400000000;29.3513900000000;29.3552800000000;29.9675000000000;29.9520000000000;29.9168000000000;29.9725000000000;29.9641700000000;29.2650000000000;29.9531700000000;29.2622200000000;29.2658300000000;29.9358800000000;30.0101800000000;30.0101800000000;29.2700000000000;29.3200000000000;29.2577800000000;29.2580600000000;29.2577800000000;29.2566700000000;29.2580600000000;29.2569400000000;29.4586100000000;29.4586100000000;29.4586100000000;29.4416700000000;29.4416700000000;29.4416700000000;29.3067000000000;29.3067000000000;29.3538900000000;29.3527800000000;29.3067000000000;29.3500000000000;29.3038900000000;29.3061100000000;29.2811000000000;29.3500000000000;29.3500000000000;29.3500000000000;29.4167000000000;29.3382860000000;29.3382860000000;29.3382860000000;29.3382860000000;29.2500000000000;29.2500000000000;29.2500000000000;29.2500000000000;29.2500000000000;29.2500000000000;29.2952070000000;29.3014000000000;29.2952070000000;29.2952070000000;29.3014000000000;29.2497000000000;29.4105070000000;29.4114850000000;29.4963333300000;29.6450740000000;29.6550980000000;29.6591860000000;29.3888888900000;29.3888888900000;29.3888888900000;29.3888888900000;29.3888888900000;29.7350000000000];
lon = [90.3419770000000;89.8800000000000;91.3200000000000;90.0000100000000;90.8700000000000;90.8700000000000;91.3200000000000;91.6166700000000;91.6166700000000;91.6166700000000;91.6000000000000;91.6027596000000;91.6166700000000;91.6184500000000;91.6187700000000;91.6189200000000;91.6185800000000;91.6027596000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9000000000000;91.9092000000000;91.8863888900000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.7858740000000;91.9113900000000;91.9050000000000;91.9080600000000;91.8916700000000;91.9083300000000;90.1845410000000;90.1845410000000;90.1845410000000;90.1845410000000;90.1845410000000;90.1845410000000;90.1845410000000;89.0619444400000;89.0212500000000;89.0800000000000;89.0800000000000;89.0800000000000;89.0800000000000;89.0212500000000;89.0800000000000;89.8800000000000;89.8800000000000;89.9800000000000;91.6027800000000;90.9494444400000;90.2508333300000;91.9266666700000;90.5357360000000;90.1167736000000;90.9768030000000;90.9768030000000;90.9768030000000;90.9768030000000;90.9708230000000;90.9708230000000;90.9708230000000;90.9673930000000;90.9673930000000;90.9673930000000;90.9678180000000;91.6100000000000;91.0124000000000;91.6900000000000;90.9678180000000;90.9678180000000;90.0969400000000;90.9563900000000;89.6225000000000;90.6763611100000;91.8800000000000;90.7244400000000;90.7118055600000;89.7175000000000;90.6763611100000;90.6763611100000;90.6763611100000;91.7927777800000;90.8966700000000;90.9611100000000;90.5200000000000;91.8400000000000;91.8150000000000;90.8738900000000;91.8150000000000;90.7166700000000;91.8119000000000;89.0617700000000;91.8100000000000;91.8100000000000;91.8100000000000;91.3200000000000;91.3200000000000;91.3200000000000;90.9701510000000;90.9701510000000;91.8086100000000;91.8133300000000;91.8105600000000;91.4152800000000;91.4183300000000;91.4133300000000;91.4133300000000;91.4183300000000;91.4169400000000;91.4152800000000;91.2076700000000;91.1980000000000;91.0513000000000;91.1760000000000;91.1888300000000;91.8166700000000;91.1986700000000;91.8086100000000;91.8213900000000;90.5939000000000;91.1917600000000;91.1917600000000;91.7100000000000;91.6900000000000;91.8516700000000;91.8419400000000;91.8511100000000;91.8455600000000;91.8419400000000;91.8452800000000;90.1358340000000;90.1358340000000;90.1358340000000;90.2333400000000;90.2333400000000;90.2333400000000;91.8078000000000;91.8078000000000;91.4325000000000;91.4338900000000;91.8078000000000;91.4167000000000;91.4608300000000;91.4500000000000;91.8772000000000;91.4167000000000;91.4167000000000;91.4167000000000;89.0833400000000;91.5179610000000;91.5179610000000;91.5179610000000;91.5179610000000;91.8100000000000;91.8100000000000;91.8100000000000;91.8100000000000;91.8100000000000;91.8100000000000;91.8102800000000;91.8058000000000;91.8102800000000;91.8102800000000;91.8058000000000;91.9836000000000;89.3660900000000;89.3645680000000;89.6308333300000;91.4176600000000;91.3872010000000;91.3789160000000;89.0394444400000;89.0394444400000;89.0394444400000;89.0394444400000;89.0394444400000;91.4150000000000];
SrY = [68.2989690700000;124.555555600000;117.442857100000;90.9738717300000;187.678571400000;140.937500000000;104.483146100000;151.299589600000;138.253012000000;134.309623400000;127.804878000000;100.752688200000;132.469304200000;136.103151900000;151.020408200000;121.257861600000;131.091180900000;150.476190500000;108.900523600000;127.927927900000;82.8037383200000;126.284875200000;79.3396226400000;71.6964285700000;83.5221421200000;72.5233644900000;72.5961538500000;66.1475409800000;95.4444444400000;108.876725400000;127.900627900000;82.8377230300000;126.284875200000;79.3310463100000;71.6720779200000;83.5127797000000;72.5573492000000;72.6398601400000;89.2708333300000;81.5740740700000;89.2893923800000;74.4736842100000;80.3462321800000;133.490566000000;115.789473700000;62.5663716800000;69.0833333300000;63.1858407100000;133.177570100000;123.214285700000;55.2352941200000;11.8222222200000;20.1083032500000;20.3765690400000;42.6229508200000;38.9696969700000;14.2105263200000;20.8558558600000;31.6113744100000;29.0173410400000;21.2403100800000;24.5205479500000;35.7142857100000;22.5454545500000;19.7391304300000;32.5490196100000;43.7333333300000;44.3382352900000;36.7295597500000;42.6811594200000;44.4029850700000;40.2857142900000;44.1481481500000;44.2968750000000;38.9130434800000;30.8938547500000;29.5135135100000;36.9930069900000;25.5514705900000;24.5197740100000;26.0769672600000;37.0068027200000;39.6240601500000;19.8650234700000;28.6219081300000;17.3779315600000;40.2857142900000;22.0750551900000;29.7657082000000;33.6464088400000;24.1964285700000;34.5833333300000;29.3069306900000;30.2500000000000;111.625000000000;37.1891038300000;15.7230142600000;33.4496124000000;32.4375000000000;32.4166666700000;29.5213620300000;15.8571428600000;27.8578290100000;37.2361809000000;24.9760765600000;36.9642857100000;35.9166666700000;31.6279069800000;23.1139896400000;24.2471264400000;18.3414634100000;13.0320699700000;15.6854838700000;24.6774193500000;32.2950819700000;29.0370370400000;34.5270270300000;36.1805555600000;23.7560975600000;33.5947712400000;39.1200000000000;34.4680851100000;22.1973094200000;18.9565217400000;19.5652173900000;17.9523809500000;24.6086956500000;16.7826087000000;34.2727272700000;15.9545454500000;31.2698412700000;29.6899224800000;22.7155172400000;24.9152542400000;5.53846153800000;41.5767634900000;23.6547085200000;28.3490566000000;25.5940594100000;32;24.4059405900000;21.5189873400000;28.1111111100000;130.416666700000;45.8119658100000;43.6000000000000;52.4264705900000;37.3913043500000;43.0201342300000;148.200000000000;86.2393162400000;20.9268292700000;23.0412371100000;56.4843750000000;44.7500000000000;60.5063291100000;59.2857142900000;118.734491300000;48.2278481000000;39.6000000000000;44.2771084300000;40.1530612200000;26.2855360800000;35.4926146000000;59.1733559400000;62.6597137500000;60.8411215000000;67.5892857100000;59.7222222200000;62.1100917400000;56.0526315800000;55;61.2977099200000;86.1165048500000;76.7889908300000;61.2195122000000;51.3709677400000;56.3392857100000;13.0790085200000;15.3507161200000;14.7290640400000;15.3448275900000;19.1387559800000;18.0203045700000;23.7770897800000;23.2653817600000;18.3368794300000;14.0569715100000;22.3582709900000;17.4563591000000];
LaYb = [53.3081340800000;41.8731410400000;21.4757243300000;37.8809983600000;38.6142571600000;32.8855005800000;14.7817657400000;23.7103258300000;22.9951476800000;22.2453555000000;22.8880233700000;18.4743233500000;22.7274137500000;21.2706647300000;22.5674032800000;21.2158390100000;22.6825430900000;21.6047589400000;31.3395218000000;25.5808280600000;41.9583023100000;25.4232195400000;40.6015111400000;38.1137019800000;39.5464135000000;39.2759105700000;41.0868791600000;47.6972798300000;41.8670886100000;31.3395218000000;25.5808280600000;41.9583023100000;25.4232195400000;40.6015111400000;38.1137019800000;39.5464135000000;39.2759105700000;41.0868791600000;34.4699814100000;39.8763713100000;40.0990389100000;38.4733041200000;43.2511351500000;37.0314912000000;36.4732770700000;39.5401925800000;38.6055366900000;38.0937883900000;39.9550299800000;39.9855792300000;35.4888694900000;3.51176428500000;4.71102704600000;5.92744270700000;13.1390383800000;13.3630357500000;4.06638141100000;10.6416412000000;9.93755274300000;9.60574054700000;9.81555153700000;7.02521354300000;6.99188195500000;7.48361977300000;5.95801341900000;19.4628666100000;8.09319164500000;8.13276282200000;7.43817764200000;8.34171010200000;8.76380970800000;8.77896786800000;8.46694796100000;8.31412557500000;7.52765423700000;6.75893960700000;5.40538088100000;8.22583885900000;7.27943244800000;6.44308147400000;8.07823699900000;8.36092177900000;10.3926569700000;4.99991209600000;9.74479848700000;5.06739651000000;8.09195830200000;9.42848242900000;6.61867383300000;5.82808789300000;5.24786529900000;5.87332981700000;6.10706218300000;6.54030882500000;16.4468132400000;12.8316924500000;3.94570286500000;8.69965477600000;5.93476144100000;8.63308720100000;10.0101810300000;3.21135404700000;6.05589127100000;7.54112956300000;9.00105485200000;13.5864978900000;7.37971648000000;8.35748390000000;5.18692862200000;5.39643483600000;6.52625060400000;2.90881138300000;5.64633678600000;4.16858458000000;11.2406277500000;7.62596978400000;6.16755496300000;6.42851074700000;4.01419255800000;6.60713253600000;6.79324894500000;7.31217768400000;4.23088311500000;6.79324894500000;7.47257384000000;6.76105345300000;6.79324894500000;6.79324894500000;14.2144452700000;7.47257384000000;6.92821415600000;7.32259301900000;7.15078836300000;6.88185654000000;11.5122234000000;9.27719264300000;7.55896197700000;5.40225035200000;5.23577235800000;6.10006027700000;5.41581597900000;4.76669148700000;5.97657420900000;21.7383966200000;21.6805817400000;21.1789525900000;10.8101265800000;10.7982240700000;10.3215255300000;24.8979733000000;19.9721519000000;4.67227981700000;5.14640071600000;19.7941219300000;13.3506211900000;12.3191007000000;14.9208860800000;28.4391930700000;13.1528862600000;13.3200959700000;13.6320901700000;7.72285143200000;8.73641953700000;9.03729175000000;16.2753855400000;17.9629829700000;15.3809410100000;17.4771768300000;16.6434599200000;17.4683544300000;16.0345155300000;17.7132379000000;19.3445850900000;20.0179761800000;16.8524829600000;14.4990238700000;13.1693685700000;16.5371009700000;3.72691692100000;3.74987341800000;6.01274800300000;5.29473814800000;6.58649789000000;6.35021097000000;5.83364315000000;6.03989505600000;5.53831288000000;5.08868525300000;6.84142801600000;4.41475624100000];

SrYT = 19.0.*log(SrY) - 22.3; % Sr/Y from proxy calibration
LaYbT = 16.6.*log(LaYb) + 7.6; % La/Yb from proxy calibration
SrYLaYbT = -9.4 + 9.8.*log(SrY) + 9.1.*log(LaYb); % Both, from proxy calibration
%SrYLaYbT = 4.2.*log(SrY).*log(LaYb) + 13.1; % Paired Sr/Y and La/Yb alternative approach recommended by Reviewer Dr. Allen Glazner

Ma = 5; % 2 sigma unc x dir
km = 10; % 2 sigma unc y dir based on residuals

%plot the filtered raw data
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

% Plot filtered data converted to crustal thickness
Figure_03B = figure;
hold on
scatter(Age,SrYT,2000,'r', 'o','filled')
scatter(Age,LaYbT,3000,'b', 's','filled')
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
scatter(Age,SrYLaYbT,1500,'y', 'd','filled')
ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

%% Monte Carlo simulation
trials = 1000; % number of simulations (100,000 used in manuscript)
h = 10; % kernel bandwidth for Gaussian smoothing

for i = 1:length(SrYLaYbT(:,1))
	r1 = randn(trials*2,1); %make random numbers for Monte Carlo
	r1 = r1(r1>-2 & r1<2); %trim outside 2 sigma
	r1 = r1(1:trials);
	r2 = randn(trials*2,1); %make random numbers for Monte Carlo
	r2 = r2(r2>-2 & r2<2); %trim outside 2 sigma
	r2 = r2(1:trials);
	% Linear fits for specific time intervals 
	for t = 1:trials
		AgeR(i,t) = Age(i,1) + Ma/2 * r1(t,1); % divide by 2 before multiplying by randn between -2 and 2 --> results in random 2 sigma dist
		ctR(i,t) = SrYLaYbT(i,1) + km/2 * r2(t,1); % ct = crustal thickness, R = random
		if Age(i) > 150 % for linear fit 
			AgeR_JK1(i,t) = AgeR(i,t);
			ctR_JK1(i,t) = ctR(i,t);
		end		
		if Age(i) <= 173 && Age(i) > 120  % for linear fit
			AgeR_JK(i,t) = AgeR(i,t);
			ctR_JK(i,t) = ctR(i,t);
		end
		if Age(i) <= 150 && Age(i) > 80 % for linear fit 
			AgeR_K1(i,t) = AgeR(i,t);
			ctR_K1(i,t) = ctR(i,t);
		end		
		if Age(i) <= 100 && Age(i) > 65 % for linear fit 
			AgeR_K(i,t) = AgeR(i,t);
			ctR_K(i,t) = ctR(i,t);
		end
		if Age(i) <= 60 && Age(i) > 30 % for linear fit 
			AgeR_Cz(i,t) = AgeR(i,t);
			ctR_Cz(i,t) = ctR(i,t);
		end
		if Age(i) < 40  % for linear fit
			AgeR_N(i,t) = AgeR(i,t);
			ctR_N(i,t) = ctR(i,t);
		end		
	end
end
% remove zeros generated from ignoring estimates outside of specific time intervals
AgeR_JK1 = AgeR_JK1(any(AgeR_JK1 ~= 0,2),:);
ctR_JK1 = ctR_JK1(any(ctR_JK1 ~= 0,2),:);
AgeR_JK = AgeR_JK(any(AgeR_JK ~= 0,2),:);
ctR_JK = ctR_JK(any(ctR_JK ~= 0,2),:);
AgeR_K1 = AgeR_K1(any(AgeR_K1 ~= 0,2),:);
ctR_K1 = ctR_K1(any(ctR_K1 ~= 0,2),:);
AgeR_K = AgeR_K(any(AgeR_K ~= 0,2),:);
ctR_K = ctR_K(any(ctR_K ~= 0,2),:);
AgeR_Cz = AgeR_Cz(any(AgeR_Cz ~= 0,2),:);
ctR_Cz = ctR_Cz(any(ctR_Cz ~= 0,2),:);
AgeR_N = AgeR_N(any(AgeR_N ~= 0,2),:);
ctR_N = ctR_N(any(ctR_N ~= 0,2),:);

Figure3C = figure;
hold on
for t = 1:trials
	if t <= 100
		scatter(AgeR(:,t),ctR(:,t),50,'c','o','filled')
	end
	for i = 1:length(Age)
		xs(i) = i;
		ys(i) = gaussian_kern_reg(xs(i),AgeR(:,t),ctR(:,t),h); %apply Gaussian kernel regression
	end
	if t <= 100
		ct_out(t,:) = ys; %calculate crustal thickness
		plot(xs,ys,'k-');
	end
end
ct_med = median(ct_out); %median of crustal thickness results for plot
ct_std = 2*std(ct_out); % 2 sigma
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
scatter(Age,SrYLaYbT,1500,'y', 'd','filled')
plot(xs,ct_med,'LineWidth',4,'Color','g')
plot(xs,ct_med+ct_std,'LineWidth',2,'Color','g')
plot(xs,ct_med-ct_std,'LineWidth',2,'Color','g')
ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

%calculate linear fits for specific time intervals 
Figure_03D = figure;
hold on
for t = 1:trials
	[pctR_JK1(t,:)] = polyfit(AgeR_JK1(:,t),ctR_JK1(:,t),1);
	fpctR_JK1(:,t) = polyval(pctR_JK1(t,:),AgeR_JK1(:,t));
	[pctR_JK(t,:)] = polyfit(AgeR_JK(:,t),ctR_JK(:,t),1);
	fpctR_JK(:,t) = polyval(pctR_JK(t,:),AgeR_JK(:,t));
	[pctR_K1(t,:)] = polyfit(AgeR_K1(:,t),ctR_K1(:,t),1);
	fpctR_K1(:,t) = polyval(pctR_K1(t,:),AgeR_K1(:,t));
	[pctR_K(t,:)] = polyfit(AgeR_K(:,t),ctR_K(:,t),1);
	fpctR_K(:,t) = polyval(pctR_K(t,:),AgeR_K(:,t));
	[pctR_Cz(t,:)] = polyfit(AgeR_Cz(:,t),ctR_Cz(:,t),1);
	fpctR_Cz(:,t) = polyval(pctR_Cz(t,:),AgeR_Cz(:,t));
	[pctR_N(t,:)] = polyfit(AgeR_N(:,t),ctR_N(:,t),1);
	fpctR_N(:,t) = polyval(pctR_N(t,:),AgeR_N(:,t));	
end

%plot random selections (first 100 trials or fewer)
for t = 1:trials
	if t <= 100
		scatter(AgeR(:,t),ctR(:,t),50,'c','o','filled')
	end
end

%plot linear fits of random selections (first 100 trials or fewer)
for t = 1:trials
	if t <= 100
		plot(AgeR_JK1(:,t),fpctR_JK1(:,t),'LineWidth',1,'Color','m')
		plot(AgeR_JK(:,t),fpctR_JK(:,t),'LineWidth',1,'Color','m')
		plot(AgeR_K1(:,t),fpctR_K1(:,t),'LineWidth',1,'Color','m')
		plot(AgeR_K(:,t),fpctR_K(:,t),'LineWidth',1,'Color','m')
		plot(AgeR_Cz(:,t),fpctR_Cz(:,t),'LineWidth',1,'Color','m')
		plot(AgeR_N(:,t),fpctR_N(:,t),'LineWidth',1,'Color','m')
	end
end

%plot crustal thickness estimates and uncertainties
plot([Age'; Age'], [(SrYLaYbT+km)'; (SrYLaYbT-km)'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
plot([(Age+Ma)'; (Age-Ma)'], [SrYLaYbT'; SrYLaYbT'], '-k', 'Color', 'k', 'LineWidth',2) % 2 sig error bars
scatter(Age,SrYLaYbT,1500,'y', 'd','filled')

%Calculate median and st dev of linear fits for each time interval then plot 
pctR_JK1_med = median(pctR_JK1);
pctR_JK1_std = 2*std(pctR_JK1); % 2 sigma
pctR_JK_med = median(pctR_JK);
pctR_JK_std = 2*std(pctR_JK); % 2 sigma
pctR_K1_med = median(pctR_K1);
pctR_K1_std = 2*std(pctR_K1); % 2 sigma
pctR_K_med = median(pctR_K);
pctR_K_std = 2*std(pctR_K); % 2 sigma
pctR_Cz_med = median(pctR_Cz);
pctR_Cz_std = 2*std(pctR_Cz); % 2 sigma
pctR_N_med = median(pctR_N);
pctR_N_std = 2*std(pctR_N); % 2 sigma

fpctR_JK1_med = median(fpctR_JK1,2);
fpctR_JK1_std = std(fpctR_JK1,[],2); % 2 sigma
plot(Age(Age > 150),fpctR_JK1_med,'LineWidth',4,'Color','b')
plot(Age(Age > 150),fpctR_JK1_med + fpctR_JK1_std,'LineWidth',2,'Color','b')
plot(Age(Age > 150),fpctR_JK1_med - fpctR_JK1_std,'LineWidth',2,'Color','b')

fpctR_JK_med = median(fpctR_JK,2);
fpctR_JK_std = std(fpctR_JK,[],2); % 2 sigma
plot(Age(Age <= 173 & Age > 120),fpctR_JK_med,'LineWidth',4,'Color','b')
plot(Age(Age <= 173 & Age > 120),fpctR_JK_med + fpctR_JK_std,'LineWidth',2,'Color','b')
plot(Age(Age <= 173 & Age > 120),fpctR_JK_med - fpctR_JK_std,'LineWidth',2,'Color','b')

fpctR_K1_med = median(fpctR_K1,2);
fpctR_K1_std = std(fpctR_K1,[],2); % 2 sigma
plot(Age(Age <= 150 & Age > 80),fpctR_K1_med,'LineWidth',4,'Color','b')
plot(Age(Age <= 150 & Age > 80),fpctR_K1_med + fpctR_K1_std,'LineWidth',2,'Color','b')
plot(Age(Age <= 150 & Age > 80),fpctR_K1_med - fpctR_K1_std,'LineWidth',2,'Color','b')

fpctR_K_med = median(fpctR_K,2);
fpctR_K_std = std(fpctR_K,[],2); % 2 sigma
plot(Age(Age <= 100 & Age > 65),fpctR_K_med,'LineWidth',4,'Color','b')
plot(Age(Age <= 100 & Age > 65),fpctR_K_med + fpctR_K_std,'LineWidth',2,'Color','b')
plot(Age(Age <= 100 & Age > 65),fpctR_K_med - fpctR_K_std,'LineWidth',2,'Color','b')

fpctR_Cz_med = median(fpctR_Cz,2);
fpctR_Cz_std = std(fpctR_Cz,[],2); % 2 sigma
plot(Age(Age <= 60 & Age > 30),fpctR_Cz_med,'LineWidth',4,'Color','b')
plot(Age(Age <= 60 & Age > 30),fpctR_Cz_med + fpctR_Cz_std,'LineWidth',2,'Color','b')
plot(Age(Age <= 60 & Age > 30),fpctR_Cz_med - fpctR_Cz_std,'LineWidth',2,'Color','b')

fpctR_N_med = median(fpctR_N,2);
fpctR_N_std = std(fpctR_N,[],2); % 2 sigma
plot(Age(Age < 40),fpctR_N_med,'LineWidth',4,'Color','b')
plot(Age(Age < 40),fpctR_N_med + fpctR_N_std,'LineWidth',2,'Color','b')
plot(Age(Age < 40),fpctR_N_med - fpctR_N_std,'LineWidth',2,'Color','b')

ylim([10 90])
yticks(10:10:90)
xlabel('Age (Ma)')
ylabel('Crustal Thickness (km)')
xticks(0:25:200)
set(gca, 'FontSize', 75)

% save plots in easy to edit fashion
%pause(15) % pause 15 s to resize fig before saving
%[file,path] = uiputfile('*.eps','Save file');
%print(Figure_03D,'-depsc','-painters',[path file]);

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