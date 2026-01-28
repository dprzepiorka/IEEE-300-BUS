%% PowerWorld Script using SimAuto
global VSN LD GEN mgsum pvsum WynikiPocz;
global Wyniki;
pvsum = 0;
mgsum = 0;

%% Initial Power Losses According to New EnGridand 39-Bus Network
% Standard Values for Control Variables %
t = fopen('VSN.txt','w');     % Optimal Control Variables
fprintf(t,'%d',VSN);
fclose(t);

LD.P1=[];
LD.Q1=[];
GEN.P1=[];
GEN.Q1=[];


%% Odbiory
t = fopen('LD_P1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d'],LD.P1);
fclose(t);
t = fopen('LD_Q1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d'],LD.Q1);
fclose(t);

%% Generatory

t = fopen('GEN_P1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d'],LD.P1);
fclose(t);
t = fopen('GEN_Q1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
    ' %d %d %d %d %d %d %d %d %d %d %d'],LD.Q1);
fclose(t);



% Making a Blank "ITR.txt" File, 0 => Work, 2 => End %
t = fopen('ITR.txt','w');
fprintf(t,'%d %d',0);
fclose(t);

% Making a Blank "Couple.txt" File, 0 => DIgSILENT, 1 => MATLAB %
t = fopen('Couple.txt','w');
fprintf(t,'%d %d',0);
fclose(t);

% Waiting for DIgSILENT to Complete the PowerFlow Process %
dlg = 0;
while dlg == 0
      pause(0.02);
      try
      dlg = load('Couple.txt');
      catch
        pause(0.02);
      end    
end



%% Wczytanie danych

% Wczytanie danych węzłów               (NR, Un,Ua,Ub,Uc,PhiA,PhiB,PhiC)
Wyniki.BUS = Dane('BusData.txt',247);  

% Wczytanie danych linii                (NR, Imax, Obciazenie, Spad.Nap,Ia,Ib,Ic,PhiA,PhiB,PhiC)
Wyniki.LINE = Dane('LineData.txt',317); 

% Wczytanie danych transformatorów      (Sn, Obc, Spad.Nap , Bus1, Bus2)
Wyniki.TRAFO = Dane('TrafoData.txt',1).';  

% Wczytanie danych źródeł               (NR, P, Q, S)
Wyniki.GEN = Dane('GenData.txt',63).';  

% Wczytanie danych odbiorow             (NR, P, Q, S, PF)
Wyniki.LOAD = Dane('LoadData.txt',161);  

% Wczytanie danych PV                   (NR, P, Q, S, PF, Obc)
%Wyniki.PV = Dane('PVData.txt',164);  

% Wczytanie danych magazynów            (NR, P, Q, S, PF, Obc)
%Wyniki.MG = Dane('MGData.txt',60);  

% Wczytanie wartości dla calego systemu (Pg,Qg,Pl,Ql,Ploss,Qloss,ExpP,ExpQ,ImpP,ImpQ)
Wyniki.GRID = Dane('GridData.txt',4);  

Wyniki.SumGenP = Wyniki.GEN(1,2);
Wyniki.SumGenQ = Wyniki.GEN(1,3);

Wyniki.SumLD_P = sum(Wyniki.LOAD(:,2));
Wyniki.SumLD_Q = sum(Wyniki.LOAD(:,3));

%Wyniki.SumPV_P = sum(Wyniki.PV(:,2));
%Wyniki.SumPV_Q = sum(Wyniki.PV(:,3));
%Wyniki.SumMG_P = sum(Wyniki.MG(:,2));
%Wyniki.SumMG_Q = sum(Wyniki.MG(:,3));

Wyniki.SumPloss = Wyniki.GRID(1,1);


End();