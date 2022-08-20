%{
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% INDIVIDUAL TEST
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%}

clear; 

i=1;

stocks = readtable('stocks.csv');
prices = carga_datos(stocks,i,0);
stockname = stocks.stock{i};
practice = stocks.practice{i};

%grafica_precios(prices, stockname, practice, 0)

%graficas_stf_espectrogramas(prices, stockname, practice, 0)

%graficas_espectro_pot(prices, stockname, practice, 0)

%graficas_cwt(prices, stockname, practice, 0)

%graficas_espectrograma(prices, stockname, practice, 0)


%Volatilidad
w=5;
logreturns = diff(log(prices{:,1}));
avgreturn = sum(logreturns)/length(logreturns);
movmean = movmean(logreturns,[w 0]);
movvar = movvar(logreturns,[w 0]);

%https://www.mathworks.com/help/matlab/timetables.html
%volat = sqrt(movvar);
%temp = prices.Date(1:length(logreturns))

prices = timetable(prices.Date(1:length(logreturns)),sqrt(movvar),'VariableNames',{'Close'});
prices.Properties.DimensionNames{1} = 'Date';



%Transformada Wavelet continua con ventana Gaussiana
dayscount = 1:length(prices.Date);
figure
cwt(prices{:,1});
figure
cwt(prices{:,1},'amor');
cwtG2 = cwtft(prices{:,1},'wavelet',{'dog',2},'plot');


%%
%{
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% CYCLES TEST
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%}

stocks = readtable('stocks.csv');
periods = [0 365 30 90];

%for i = 1:length(stocks.stock)

i=6;

disp(stocks.stock{i});
disp(stocks.practice{i});

stockname = stocks.stock{i};
practice = stocks.practice{i};

var_name_file=stockname+".csv";
data = readtimetable(var_name_file);

lim1 = stocks.start_date(i); 
lim2 = stocks.end_date(i); 

for j = 1:length(periods)
    
    disp(lim1-periods(j))

    tr = timerange(lim1-periods(j), lim2+365); 
    prices = data(tr,"Close");

   % grafica_precios(prices, stockname, practice, periods(j))

   % graficas_stf_espectrogramas(prices, stockname, practice, periods(j))

   % graficas_cwt(prices, stockname, practice, periods(j))

    graficas_espectro_pot(prices, stockname, practice, periods(j))

end
        
%end

%{

for i = 1:length(stocks.stock)
    disp(stocks.stock{i});
    for j = 1:length(r)
        disp(stocks.start_date(i)-r(j))
    end
end

%}

%{
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% FUNCIONES
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%}

function [prices] = carga_datos(stocks,i,p)
    disp(stocks.stock{i});
    disp(stocks.practice{i});
   
    lim1 = stocks.start_date(i); 
    lim2 = stocks.end_date(i); 
    tr = timerange(lim1-p, lim2+365); 
    
    var_name_file=stocks.stock{i}+".csv";
    data = readtimetable(var_name_file);
    prices = data(tr,"Close");
end 

function [] = grafica_precios(prices, stockname, practice, period)
    figure
    plot(prices.Date, prices{:,1});
    title(stockname+" Price Series " + "("+practice+")");
    xlabel('Trading period (days)','FontSize',8);
    ylabel('Prices (USD)','FontSize',8);
    subtitle("Estimated days prior manipulation: "+period,'FontAngle','italic');

end

function [] = graficas_stf_espectrogramas(prices, stockname, practice, period)

%{
    figure
    spectrogram(prices{:,1},[],[],'power','yaxis') 
    xlabel('')
    ylabel('')
    title("Short-Time Fourier Spectrogram (Default values)");
    subtitle("Previous data: "+period+" days",'FontAngle','italic');
        
    figure
    spectrogram(prices{:,1},kaiser(110,2.5),55,'power','yaxis') 
    xlabel('')
    ylabel('')
    subtitle("Previous data: "+period+" days",'FontAngle','italic');
%}

    figure
    tiledlayout(4,1)
    nexttile([2 1]);
    spectrogram(prices{:,1},[],[],'power','yaxis') 
    xlabel('')
    ylabel('Normalized Freq. (cycles/sample)','FontSize',8);
    title("Short-Time Fourier Spectrogram (Default values)");
    subtitle("Estimated days prior manipulation: "+period,'FontAngle','italic');
    
    nexttile([2 1]);
    spectrogram(prices{:,1},kaiser(110,2.5),55,'power','yaxis') 
%    spectrogram(prices{:,1},kaiser(55,2.5),25,'power','yaxis') 
    xlabel('Samples (Sample Rate=1 day)','FontSize',8);
    ylabel('')
    subtitle('(Custom values)','FontAngle','italic');

end


function [] = graficas_cwt(prices, stockname, practice, period)

    figure
    tiledlayout(2,1)

    nexttile;
    [wt1,f1, coi1] = cwt(prices{:,1});
    pcolor(1:numel(prices{:,1}),log2(f1/0.1),abs(wt1));
    shading interp; 
    title("Scalogram CWT (Default values)");
    subtitle("Estimated days prior manipulation: "+period,'FontAngle','italic');
    ylabel('Normalized Freq. (cycles/sample)','FontSize',8);
    colorbar;
    
    hold on;
    plot(1:numel(prices{:,1}),log2(coi1/0.1),'w--','linewidth',2)
    hold off;

    nexttile;
    [wt1,f1, coi1] = cwt(prices{:,1},'bump');
    pcolor(1:numel(prices{:,1}),log2(f1/0.1),abs(wt1));
    shading interp; 
    subtitle('(Custom values)','FontAngle','italic');
    xlabel('Samples (Sample Rate=1 day)','FontSize',8);
    colorbar;
    
    hold on;
    plot(1:numel(prices{:,1}),log2(coi1/0.1),'w--','linewidth',2)
    hold off;

end

function [] = graficas_espectro_pot(prices, stockname, practice, period)

    figure
    tiledlayout(2,1)

    nexttile;
    pspectrum(prices);
    xlabel('')
    title("Power Spectrum (Default values)");
    subtitle("Estimated days prior manipulation: "+period,'FontAngle','italic');
    
    nexttile;
    pspectrum (prices{:,1},prices.Date,'Leakage',0.25);
    ylabel('')
    subtitle('(Custom values)','FontAngle','italic');


end

function [] = graficas_espectrograma(prices, stockname, practice, period)
    
    figure
    tiledlayout(2,1)

    nexttile;
    pspectrum(prices,'spectrogram');
    xlabel('');
    ylabel('Frequency (cycles/day)','FontSize',8);
    title("Power Spectrum Spectrogram (Default values)");
    subtitle("Estimated days prior manipulation: "+period,'FontAngle','italic');
    
    nexttile;
    pspectrum(prices{:,1},prices.Date,'Leakage',0.25,'spectrogram');
    xlabel('Trading period (days)','FontSize',8);
    ylabel('');
    title('');
    subtitle('(Custom values)','FontAngle','italic');

end