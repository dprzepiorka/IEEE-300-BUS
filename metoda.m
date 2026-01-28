function metoda(x,fc,fval,MET,nr)
global MG MAG ZAL wez STOP Nap lm STP Pg Pm M Wartosci PV Flaga Wyniki PV_Q PV_P LD_P Funk Legend1 Legend2 LD Nastawy
Col = turbo(18);
font = 34.5;
funkcja=fcelu_min(x);


%XLSX=strcat(MET,'.xlsx');
TXT=strcat('Wyniki',MET,'.txt');

dfc=size(fc,2);
dx=size(x,2);
Funk(nr,1)=fval(1,1);  
Funk(nr,2:dx+1)=x(1,1:dx);  
Funk(nr,dx+2:dx+dfc+1)=fc(1,1:dfc);  

for i=1:dfc
    f.itr(1,i) = i;
end


T = table(1,dx,dfc,fval,x,fc);
writetable(T,TXT,'WriteVariableNames',false);






% Wykresy
Wezly = Wyniki.BUS;
Wezly =sortrows(Wezly).';
Nap.F1(3,:)=Wezly(3,:);
Nap.F2(3,:)=Wezly(4,:);
Nap.F3(3,:)=Wezly(5,:);
Nap.U(3,:)=Wezly(13,:);


figure(nr*3);
%figure(3);

h=plot(Nap.F1(1,:),Nap.F1(3,:),Nap.F2(1,:),Nap.F2(3,:),Nap.F3(1,:),Nap.F3(3,:),Nap.U(1,:),Nap.U(3,:));
grid on;
set(h,'LineWidth',2,{'Color'},{Col(2,:);Col(3,:);Col(4,:);Col(15,:)});
h=xlabel('Numer węzła'); %or h=get(gca,'xlabel')
set(h, 'FontSize', font) 
set(h,'FontWeight','bold') %bold font
h=ylabel('Napięcie fazowe w węźle, pu');
set(h, 'FontSize', font) 
set(h,'FontWeight','bold') %bold font
set(gca,'FontSize',font);
legend({'F1','F2','F3','Asymetria U'},'FontSize',14,'Location','northwest');
ylim([0,1.3]);




%figure((nr*3)+1);
figure(4);

hold on;
h=plot(f.itr,fc);
set(h,'LineWidth',4);
grid on;
h=xlabel('Numer iteracji'); %or h=get(gca,'xlabel')
set(h, 'FontSize', font) 
set(h,'FontWeight','bold') %bold font
h=ylabel('Wartość funckji celu');
set(h, 'FontSize', font) 
set(h,'FontWeight','bold') %bold font
set(gca,'FontSize',font);
hold off



%% Wykres wskazowy napięc w węźle w

xX(1:4,1)=0;
xX(1,2)=real(Wyniki.TRAFO(1,10));
xX(2,2)=real(Wyniki.TRAFO(1,11));
xX(3,2)=real(Wyniki.TRAFO(1,12));
xX(4,2)=real(Wyniki.TRAFO(1,13));
y(1:4,1)=0;
y(1,2)=imag(Wyniki.TRAFO(1,10));
y(2,2)=imag(Wyniki.TRAFO(1,11));
y(3,2)=imag(Wyniki.TRAFO(1,12));
y(4,2)=imag(Wyniki.TRAFO(1,13));


figure((nr*3)+2);
%figure(5);

hold on;
drawArrow = @(xX,y) quiver( xX(1),y(1),xX(2)-xX(1),y(2)-y(1),MaxHeadSize=0.5);  
h=drawArrow(xX(1,:),y(1,:));
set(h,'LineWidth',2);
h=drawArrow(xX(2,:), y(2,:));
set(h,'LineWidth',2);
h=drawArrow(xX(3,:), y(3,:));
set(h,'LineWidth',2);
h=drawArrow(xX(4,:), y(4,:));
set(h,'LineWidth',2);
legend({'I_{F1}','I_{F2}','I_{F3}','Aasymetria I'},'FontSize',14);
hold off;