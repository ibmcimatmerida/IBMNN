%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       PLOT SOLUTION (RECTANGULAR)                       %                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear

XU = load('dataR/xu.txt');
xu = XU(:,1);
yu = XU(:,2);
Nu = length(xu);
XF = load('dataR/xf.txt');
xf = XF(:,1);
yf = XF(:,2);
Nf = length(xf);
xp = load('dataR/xp_x.txt');
yp = load('dataR/xp_y.txt');
uN = load('dataR/xp_uN.txt');
uA = load('dataR/xp_uA.txt');
%-----------
tag_xs = 0;
filename = 'dataR/xs.txt';
if exist(filename,'file') == 2
   XS = load('dataR/xs.txt');
   xs = XS(:,1);
   ys = XS(:,2);
   Ns = length(xs);
   tag_xs = 1;
end
%-----------
tag_xun = 0;
filename = 'dataR/xun.txt';
if exist(filename,'file') == 2
   XUN = load('dataR/xun.txt');
   xun = XUN(:,1);
   yun = XUN(:,2);
   Nun = length(xun);
   tag_xun = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot

hFig1 = figure;
set(hFig1,'Position',[100 100 1000 300])
%------------------------------------------------
subplot 131
surface(xp,yp,uA)
xlabel('x'); ylabel('y'); zlabel('uA'); title('Analytical solution')
grid on
view(3)
%------------------------------------------------
subplot 132
surface(xp,yp,uN)  
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%