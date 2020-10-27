%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       PLOT SOLUTION (ARBITRARY)                       %                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear

trip = load('dataA/trip.txt');
%-----------
XU = load('dataA/xu.txt');
xu = XU(:,1);
yu = XU(:,2);
Nu = length(xu);
XF = load('dataA/xf.txt');
xf = XF(:,1);
yf = XF(:,2);
Nf = length(xf);
ph = load('dataA/xp.txt');
xp = ph(:,1);
yp = ph(:,2);
uA = ph(:,3);
uN = ph(:,4);
%-----------
tag_xs = 0;
filename = 'dataA/xs.txt';
if exist(filename,'file') == 2
   XS = load('dataA/xs.txt');
   xs = XS(:,1);
   ys = XS(:,2);
   Ns = length(xs);
   tag_xs = 1;
end
%-----------
tag_xun = 0;
filename = 'dataA/xun.txt';
if exist(filename,'file') == 2
   XUN = load('dataA/xun.txt');
   xun = XUN(:,1);
   yun = XUN(:,2);
   Nun = length(xun);
   tag_xun=1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot

hFig1 = figure;
set(hFig1,'Position',[100 100 1000 300])
%------------------------------------------------
subplot 131
trisurf(trip,xp,yp,uA,uA,'edgecolor','none','FaceColor','interp')
xlabel('x'); ylabel('y'); zlabel('uA'); title('Analytical solution')
grid on
view(3)
%------------------------------------------------
subplot 132
trisurf(trip,xp,yp,uN,uN,'edgecolor','none','FaceColor','interp')
xlabel('x'); ylabel('y'); zlabel('uN'); title('Predicted solution')
grid on
view(3)
%------------------------------------------------
subplot 133
str1 = sprintf('N_u=%d',Nu);
str2 = sprintf('N_f=%d',Nf);
if tag_xs==1
    plot(xu,yu,'ok',xf,yf,'.r',xs,ys,'sm');
    str3 = sprintf('N_{s}=%d',Ns);
    legend(str1,str2,str3)
elseif tag_xun==1
    plot(xun,yun,'*b',xu,yu,'ok',xf,yf,'.r');
    str0 = sprintf('N_{un}=%d',Nun);
    legend(str0,str1,str2)
else
    plot(xu,yu,'ok',xf,yf,'.r');
    legend(str1,str2)
end
xlabel('x'); ylabel('y'); title('Points')
axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
