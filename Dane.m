%% File - nazwa pliku, NR - liczba elementów

function [ZZ] = Dane(File,NR)
fid = fopen(File,'rt') ;

D = fread(fid);
fclose(fid);
W = char(D.');
X = strrep(W, ',', '.');
Y = split(X,'/');
Z = split(Y(1:NR,1),'*');
ZZ = str2double(Z);
