
%% Źródła
ES.P(1,1:5)=[-50 -50 -50 -50 -50];
ES.Q(1,1:5)=[0 0 0 0 0];
%ES.Wyj=ES.P;

G=(1/3)*0.2;
L=(1/3)*0.1;
M=(1/3)*1;

%Generatory
s=size(GEN.P);
R=random('Normal',1,G,s);
GEN.P(1,1)=GEN.Wyj(1,1).*R(1,1);	GEN.P(1,2)=GEN.Wyj(1,2).*R(1,2);		GEN.P(1,4)=GEN.Wyj(1,4).*R(1,4);	GEN.P(1,5)=GEN.Wyj(1,5).*R(1,5);																	GEN.P(1,22)=GEN.Wyj(1,22).*R(1,22);									GEN.P(1,31)=GEN.Wyj(1,31).*R(1,31);	GEN.P(1,32)=GEN.Wyj(1,32).*R(1,32);						GEN.P(1,38)=GEN.Wyj(1,38).*R(1,38);					GEN.P(1,43)=GEN.Wyj(1,43).*R(1,43);				GEN.P(1,47)=GEN.Wyj(1,47).*R(1,47);	GEN.P(1,48)=GEN.Wyj(1,48).*R(1,48);	GEN.P(1,49)=GEN.Wyj(1,49).*R(1,49);	GEN.P(1,50)=GEN.Wyj(1,50).*R(1,50);	GEN.P(1,51)=GEN.Wyj(1,51).*R(1,51);	GEN.P(1,52)=GEN.Wyj(1,52).*R(1,52);	GEN.P(1,53)=GEN.Wyj(1,53).*R(1,53);	GEN.P(1,54)=GEN.Wyj(1,54).*R(1,54);	GEN.P(1,55)=GEN.Wyj(1,55).*R(1,55);	GEN.P(1,56)=GEN.Wyj(1,56).*R(1,56);	GEN.P(1,57)=GEN.Wyj(1,57).*R(1,57);	GEN.P(1,58)=GEN.Wyj(1,58).*R(1,58);	GEN.P(1,59)=GEN.Wyj(1,59).*R(1,59);	GEN.P(1,60)=GEN.Wyj(1,60).*R(1,60);	GEN.P(1,61)=GEN.Wyj(1,61).*R(1,61);	GEN.P(1,62)=GEN.Wyj(1,62).*R(1,62);	GEN.P(1,63)=GEN.Wyj(1,63).*R(1,63);	GEN.P(1,64)=GEN.Wyj(1,64).*R(1,64);	GEN.P(1,65)=GEN.Wyj(1,65).*R(1,65);

%GEN.P=GEN.Wyj.*R;
SumG=sum(GEN.P);

%ES
s=size(ES.P);
R=random('Normal',0,M,s);

ES.P=ES.Wyj.*R;


% LD
s=size(LD.P);
R=random('Normal',0.5,L,s);
R=rand(s);
LD.P=LD.WyjP.*R;



