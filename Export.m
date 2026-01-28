clear Wiersz
prad = 0;
prad1=0;

Badanie.Linie(Badanie.NR,:)=Wyniki.LINE(:,5).';
Wiersz(1,1)=Badanie.NR;

Wiersz(1,2)=Wyniki.GEN(1,3).'; % Gen.1
Wiersz(1,3)=Wyniki.GEN(2,3).'; % Gen.10
Wiersz(1,4)=Wyniki.GEN(4,3).'; % Gen.108
Wiersz(1,5)=Wyniki.GEN(5,3).'; % Gen.11
Wiersz(1,6)=Wyniki.GEN(22,3).'; % Gen.17
Wiersz(1,7)=Wyniki.GEN(31,3).'; % Gen.2
Wiersz(1,8)=Wyniki.GEN(32,3).'; % Gen.20
Wiersz(1,9)=Wyniki.GEN(38,3).'; % Gen.23
Wiersz(1,10)=Wyniki.GEN(43,3).'; % Gen.24
Wiersz(1,11)=Wyniki.GEN(47,3).'; % Gen.3
Wiersz(1,12)=Wyniki.GEN(48,3).'; % Gen.372
Wiersz(1,13)=Wyniki.GEN(49,3).'; % Gen.44
Wiersz(1,14)=Wyniki.GEN(50,3).'; % Gen.45
Wiersz(1,15)=Wyniki.GEN(51,3).'; % Gen.49
Wiersz(1,16)=Wyniki.GEN(52,3).'; % Gen.55
Wiersz(1,17)=Wyniki.GEN(53,3).'; % Gen.57
Wiersz(1,18)=Wyniki.GEN(54,3).'; % Gen.61
Wiersz(1,19)=Wyniki.GEN(55,3).'; % Gen.62
Wiersz(1,20)=Wyniki.GEN(56,3).'; % Gen.63
Wiersz(1,21)=Wyniki.GEN(57,3).'; % Gen.7039
Wiersz(1,22)=Wyniki.GEN(58,3).'; % Gen.71
Wiersz(1,23)=Wyniki.GEN(59,3).'; % Gen.76
Wiersz(1,24)=Wyniki.GEN(60,3).'; % Gen.772
Wiersz(1,25)=Wyniki.GEN(61,3).'; % Gen.8
Wiersz(1,26)=Wyniki.GEN(62,3).'; % Gen.84
Wiersz(1,27)=Wyniki.GEN(63,3).'; % Gen.9001
Wiersz(1,28)=Wyniki.GEN(64,3).'; % Gen.91
Wiersz(1,29)=Wyniki.GEN(65,3).'; % Gen.92


dim=size(Wiersz(1,:),2);
Wiersz(1,dim+1)=Wyniki.MG(3,3);        % ES 11
Wiersz(1,dim+2)=Wyniki.MG(7,3);        % ES 23
Wiersz(1,dim+3)=Wyniki.MG(20,3);       % ES 7
Wiersz(1,dim+4)=Wyniki.MG(21,3);       % ES 74
Wiersz(1,dim+5)=Wyniki.MG(22,3);       % ES 77

%{
Wiersz(1,dim+6)=Wyniki.LOAD(13,2);     % LD 112
Wiersz(1,dim+7)=Wyniki.LOAD(38,2);     % LD 15
Wiersz(1,dim+8)=Wyniki.LOAD(74,2);     % LD 20
Wiersz(1,dim+9)=Wyniki.LOAD(73,2);     % LD 2
Wiersz(1,dim+10)=Wyniki.LOAD(115,2);   % LD 26
Wiersz(1,dim+11)=Wyniki.LOAD(136,2);   % LD 44
%}
obc=size(Wyniki.LOAD,1);
Wiersz(1,dim+6:dim+obc+5)=Wyniki.LOAD(:,2);     % LD 112


dim=size(Wiersz(1,:),2);
Wiersz(1,dim+1)=Wyniki.LINE(257,5);   % Linia 5.73
Wiersz(1,dim+2)=Wyniki.LINE(306,5);   % Linia 8.14
Wiersz(1,dim+3)=Wyniki.LINE(289,5);   % Linia 73.71
Wiersz(1,dim+4)=Wyniki.LINE(124,5);   % Linia 175.173
Wiersz(1,dim+5)=Wyniki.LINE(120,5);   % Linia 173.172
Wiersz(1,dim+6)=Wyniki.LINE(226,5);   % Linia 35.77
Wiersz(1,dim+7)=Wyniki.LINE(258,5);   % Linia 5.9

%Wiersz(1,dim+3)=Wyniki.LINE(304,5);   % Linia 79.73
%Wiersz(1,dim+4)=Wyniki.LINE(226,5);   % Linia 35.77
%Wiersz(1,dim+5)=Wyniki.LINE(271,5);   % Linia 6.2
%Wiersz(1,dim+7)=Wyniki.LINE(286,5);   % Linia 71.72

lin=size(Wyniki.LINE,1);
for s=1:1:lin
                if Wyniki.LINE(s,5)>100
                    prad1=(Wyniki.LINE(s,5)-100);
                else
                    prad1=0;
                end
                prad=prad+prad1;
end   


Wiersz(1,dim+8)=prad;   % Suma przec


Badanie.Dane(Badanie.NR,:)=Wiersz(1,:);

%Wiersz(1,79)=Wyniki.SumGenP;
T1=array2table(Badanie.Dane(:,:));
Nr=Badanie.NR+1;
%Range="A"+Nr+":au"+Nr;
Range="A"+Nr+":IQ"+Nr;
writetable(T1(Badanie.NR,:),'Wyniki.xlsx','WriteVariableNames',false,'Sheet',1,'Range',Range);

