LD.W(2,1:192)=num2cell(LD.P(1,1:192));
LD.W(3,1:192)=num2cell(LD.Q(1,1:192));

GEN.W(2,1:65)=num2cell(GEN.P(1,1:65));
GEN.W(3,1:65)=num2cell(GEN.Q(1,1:65));

%% Odbiory
t = fopen('LD_P1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.P(1,1:48));
fclose(t);
t = fopen('LD_P2.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.P(1,49:96));
fclose(t);
t = fopen('LD_P3.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.P(1,97:144));
fclose(t);
t = fopen('LD_P4.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.P(1,145:192));
fclose(t);

t = fopen('LD_Q1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.Q(1,1:48));
fclose(t);
t = fopen('LD_Q2.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.Q(1,49:96));
fclose(t);
t = fopen('LD_Q3.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.Q(1,97:144));
fclose(t);
t = fopen('LD_Q4.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],LD.Q(1,145:192));
fclose(t);



%% Generatory
t = fopen('GEN_P1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],GEN.P(1,1:32));
fclose(t);
t = fopen('GEN_P2.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],GEN.P(1,33:65));
fclose(t);

t = fopen('GEN_Q1.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],GEN.Q(1,1:32));
fclose(t);
t = fopen('GEN_Q2.txt','w');     
fprintf(t,['%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'],GEN.Q(1,33:65));
fclose(t);

%% ES
t = fopen('ES_P.txt','w');     
fprintf(t,['%d %d %d %d %d'],ES.P(1,1:5));
fclose(t);
t = fopen('ES_Q.txt','w');     
fprintf(t,['%d %d %d %d %d'],ES.Q(1,1:5));
fclose(t);



% Making a Blank "ITR.txt" File, 0 => Work, 2 => End %
t = fopen('ITR.txt','w');
fprintf(t,'%d %d',0);
fclose(t);

% Making a Blank "Couple.txt" File, 0 => DIgSILENT, 1 => MATLAB %
t = fopen('Couple.txt','w');
fprintf(t,'%d %d',0);
fclose(t);

