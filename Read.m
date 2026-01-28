%% Wczytanie danych

% Wczytanie danych węzłów               (NR, Un,Ua,Ub,Uc,PhiA,PhiB,PhiC)
Wyniki.BUS = Dane('BusData.txt',245);  

% Wczytanie danych linii                (NR, Imax, Obciazenie, Spad.Nap,Ia,Ib,Ic,PhiA,PhiB,PhiC)
Wyniki.LINE = Dane('LineData.txt',322); 

% Wczytanie danych transformatorów      (Sn, Obc, Spad.Nap , Bus1, Bus2)
Wyniki.TRAFO = Dane('TrafoData.txt',57);  

% Wczytanie danych źródeł               (NR, P, Q, S)
Wyniki.GEN = Dane('GenData.txt',65);  

% Wczytanie danych odbiorow             (NR, P, Q, S, PF)
Wyniki.LOAD = Dane('LoadData.txt',192);  

% Wczytanie danych PV                   (NR, P, Q, S, PF, Obc)
Wyniki.PV = Dane('PVData.txt',107);  

% Wczytanie danych magazynów            (NR, P, Q, S, PF, Obc)
Wyniki.MG = Dane('MGData.txt',196);  
Wyniki.FW(:,:) = Wyniki.MG(25:131,1:7);
Wyniki.GENs(:,:) = Wyniki.MG(132:196,1:5);
sortrows(Wyniki.GEN,2);
sortrows(Wyniki.GENs,2);
for i=1:1:65
    if Wyniki.GEN(i,3)==0
       Wyniki.GEN(i,1:5)=Wyniki.GENs(i,1:5);
    else
       Wyniki.GEN(i,1:5)=Wyniki.GEN(i,1:5);
    end
end 
Wyniki.MG = Wyniki.MG(1:24,1:7);

% Wczytanie wartości dla calego systemu (Pg,Qg,Pl,Ql,Ploss,Qloss,ExpP,ExpQ,ImpP,ImpQ)
Wyniki.GRID = Dane('GridData.txt',4);  

Wyniki.SumGenP = Wyniki.GRID(1,1);
Wyniki.SumGenQ = Wyniki.GRID(1,2);

Wyniki.SumLD_P = Wyniki.GRID(1,3);
Wyniki.SumLD_Q = Wyniki.GRID(1,4);



%Wyniki.SumGenP = sum(Wyniki.GEN(:,2));
%Wyniki.SumGenQ = sum(Wyniki.GEN(:,3));

%Wyniki.SumLD_P = sum(Wyniki.LOAD(:,2));
%Wyniki.SumLD_Q = sum(Wyniki.LOAD(:,3));