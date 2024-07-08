clc;
clear; 
close all;

choosebg = 0;
frame = 1;
choosepsf = 1;
alphabeta = 5;
shift = 0;
use_gpu = 0;
bsline = 1;
noisereduction = 1;

illuminationsave = 0;
dispfigure = 1;
interpolation = 1;

for i=1:90
    tar_1(:,:,i) = im2double(imread('data.tif',i+(frame-1)*90));
end
    
min1 = min(tar_1(:)); 

tar_1 = tar_1-min1; tar_1(tar_1<0)=0;

sigma = 0.1;
tttarget = tar_1;
pixel_size      = 25e-9; 
W               = size(tttarget,1);
L               = size(tttarget,3);
for i = 1:L
    target(:,:,i) = imgaussfilt(tttarget(:,:,i),sigma);
end

if bsline
    for i = 1:L
        base(:,:,i) = imgaussfilt(target(:,:,i),100)/1.5;
        target(:,:,i) = target(:,:,i)-base(:,:,i);
    end
    target(target<0)=0;
end

saveon = false;
savetiff = 1;

if interpolation
    interpol      = 3;
else
    interpol      = 1;
end
lambda2       = alphabeta*1e-2/5;
lambda3       = alphabeta*1e-2/2;
iter_max      = 200;   
noisefac      = 1;

max3            = @(x) max(max(max(x)));
max2            = @(x) max(max(x));

if noisereduction
    Target = target - max2(mean(target,3))*0.2*choosebg/10;
    Target(Target<0)=0;
    
else
    Target = target;
end

if use_gpu
    Target = gpuArray(Target);
end

Target_save = Target;

w2 = tukeywin(W,0.5);
w = w2*w2'; 
w = fspecial('Gaussian',W,W/2);
w = w-w(W,W);
w = w.*(w>0);
w = w./max(w(:))+0.1; 

for i=1:L
    Target(:,:,i) = w.*Target(:,:,i);
end     

if use_gpu
    w = gpuArray(w);
end

wavelengths = [515]; 

for i = 1: length(wavelengths)
    psf_mod = 1;
    wavelength = wavelengths(i)*1e-9;
    NA = 1.49;
    L1 = psf_mod*(0.61*wavelength/NA)/pixel_size; 
    u = 2*(floor(L1)-2);
    [Xpsf,Ypsf] = meshgrid(-u:u,-u:u);
    R = sqrt(Xpsf.^2 + Ypsf.^2);
    R = R*1.220*pi/L1;
    psf = (2*besselj(1,R)./R).^2; psf(u+1,u+1)=1;
    psf = psf/sum(psf(:));
    f_list(:,:,i) =  (psf); 
end

f = imgaussfilt(f_list(:,:,choosepsf),sigma);
if use_gpu
    f = gpuArray(f);
end

if interpolation
    
    Wi = W*interpol; 
    Target2 = zeros(Wi,Wi,L);
    for i=1:L
        Target2(:,:,i) = imresize(Target(:,:,i),[Wi Wi],'bicubic');
        Target2(Target2<0) = 0; 
    end
    
    Wp = size(f,2)*interpol;  
    fi = zeros(Wp,Wp);
    fi(:,:) = imresize(f(:,:),[Wp Wp],'bicubic');
    f=fi/interpol^2;
    f(f<0) = 0; 
    Wp = size(w,2)*interpol;  
    wini = zeros(Wp,Wp);
    wini(:,:) = imresize(w(:,:),[Wp Wp],'bicubic');
    w=wini/interpol^2;
    w(w<0) = 0; 
    zero_pad = 1;
    z_edgei = zero_pad*2*ceil(size(f,2)/2);
    mzp = Wi + z_edgei;
    M = zeros(mzp,mzp,L);  
    M(1+z_edgei/2:z_edgei/2+Wi,1+z_edgei/2:z_edgei/2+Wi,1:L) = Target2;
    clear Target;
    Target = M;
    maskc =  (zeros(mzp,mzp));
    maskc(1+z_edgei/2:z_edgei/2+Wi,1+z_edgei/2:z_edgei/2+Wi) = 1;
    Wi = size(Target,1);    
else    
    Wi = W;
    zero_pad = 1;
    z_edge = zero_pad*2*ceil(size(f,2)/2);
    mzp = Wi + z_edge;
    M = zeros(mzp,mzp,L); r_x = M;  
    M(1+z_edge/2:z_edge/2+Wi,1+z_edge/2:z_edge/2+Wi,1:L) = Target;
    Target = M;
    maskc =  (zeros(mzp,mzp));
    maskc(1+z_edge/2:z_edge/2+W,1+z_edge/2:z_edge/2+W) = 1;
    Wi = size(Target,1);
    z_edgei = z_edge;    
end
if use_gpu
    maskc = gpuArray(maskc);
end

SS = sum(Target(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi,:),3);
SSn(:,:)=SS(:,:)-min(min(SS(:,:))); SSn(:,:)=SSn(:,:)/max(max(SSn(:,:))); 

if dispfigure
    figure(2);
    subplot 131; imagesc(SS./w); axis image; colormap(jet); colorbar; title('SS');
end

lambda1     =   0;
lambda4     =   0;
t_old       =   1;
rho_old     =   ones(Wi);    
iter        =   0;
  
rho_ani(:,:,1) = rho_old;    
y_old       =   rho_old;
L1          = @(x) norm(x, 1);

if use_gpu
    rho_old = gpuArray(rho_old);
    rho_ani = gpuArray(rho_ani);
    y_old = gpuArray(y_old);
end

for i1=1:L
    I(:,:,i1)=ones(Wi).*mean2(Target(:,:,i1));         
end            

if use_gpu
    I = gpuArray(I);
end

epsil_old       =   sqrt(y_old);
for i1=1:L
    smalli(:,:,i1) = sqrt(I(:,:,i1));           
end   

if use_gpu
    epsil_old = gpuArray(epsil_old);
    smalli = gpuArray(smalli);
end

r = zeros(Wi,Wi,L);
for i4=1:L
    Rho(:,:,i4)=conv_fft2((y_old.*I(:,:,i4)),f,'same').*maskc; 
end  
for i=1:L
    r(:,:,i)=Target(:,:,i)-Rho(:,:,i);
end                              

if use_gpu
    Rho = gpuArray(Rho);
    r = gpuArray(r);
end

Grho=zeros(Wi);
for ii=1:L
    Gr=-1*I(:,:,ii).*conv_fft2(r(:,:,ii),f,'same');
    Grho=Grho+Gr;
end
drho=Grho.*epsil_old;  
GI=zeros(Wi,Wi,L);
GI(:,:,1)=-1*y_old.*smalli(:,:,1).*conv_fft2((2*r(:,:,1)-r(:,:,L)),f,'same');
GI(isnan(GI))=0;
GI(:,:,L-1)=-1*y_old.*smalli(:,:,L-1).*conv_fft2((2*r(:,:,L-1)-r(:,:,L)),f,'same');
GI(isnan(GI))=0;
for ii2=2:L-2
    GI(:,:,ii2)=-1*y_old.*smalli(:,:,ii2).*conv_fft2((r(:,:,ii2)),f,'same');
    GI(isnan(GI))=0;
end
dI=GI;  

if use_gpu
    Gr = gpuArray(Gr);
    drho = gpuArray(drho);
    Grho = gpuArray(Grho);
    GI = gpuArray(GI);
    dI = gpuArray(dI);
end

ts=0; tm=0;
for i1=1:L
    ss=sum(dot(conv_fft2(drho.*I(:,:,i1),f,'same'),r(:,:,i1)));
    ts=ts+ss;
end
for i1=1:L
    sm=sum(dot(conv_fft2(drho.*I(:,:,i1),f,'same'),conv_fft2(drho.*I(:,:,i1),f,'same')));
    tm=tm+sm;
end
alpha           =   ts/tm*lambda2; 
epsil_new       =   epsil_old+drho.*alpha; 
rho_new         =   epsil_new.^2;

ts=0; tm=0; beta=zeros(1,L);
for i4=1:L-1
    ts=sum(dot(conv_fft2(y_old.*dI(:,:,i4),f,'same'),r(:,:,i4)));
    tm=sum(dot(conv_fft2(y_old.*dI(:,:,i4),f,'same'),conv_fft2(y_old.*dI(:,:,i4),f,'same')));
    if tm == 0
        beta(1,i4)=0;
    else
        beta(1,i4)=ts/tm*lambda3;
    end
end

for i3=1:L-1
    smallin(:,:,i3)=smalli(:,:,i3)+dI(:,:,i3).*beta(1,i3);
    In(:,:,i3)=smallin(:,:,i3).^2;
end

I_L=(In(:,:,1)+In(:,:,L-1))/2;
In(:,:,L)=I_L;

if use_gpu
    smallin = gpuArray(smallin);
    epsil_new = gpuArray(epsil_new);
    In = gpuArray(In);
    I_L = gpuArray(I_L);
end

for i4=1:L 
    Rhon(:,:,i4)=conv_fft2((rho_new.*In(:,:,i4)),f,'same').*maskc;   
end
F=0;
for i6=1:L
    Fsub(i6)=abs(norm(Target(:,:,i6)-Rhon(:,:,i6)));
    F=F+Fsub(i6);
end

if use_gpu
    Rhon = gpuArray(Rhon);
    rho_new = gpuArray(rho_new);
end

for i7 = 1:L
    normIn(i7) = norm(In(:,:,i7));
end

F = F + lambda1 * L1(rho_new) + lambda4 * sum(normIn);      
Fmat=zeros(10001,1);  Fmat(1,1)=F;
rho_ani(:,:,2)=rho_new;
epsil_old=epsil_new;
I=In; 
smalli=smallin;

t_new = (1+sqrt(1+4*t_old^2))/2;
y_new = rho_new+((t_old-1)/t_new)*(rho_new-rho_old);

rho_old = rho_new;
y_old   = y_new;
t_old   = t_new;

if use_gpu
    y_new = gpuArray(y_new);
end

iter=1;

if dispfigure
    figure(2); 
    if interpolation
    subplot 132; imagesc(real(rho_new(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi)./w)); axis image; colormap(jet); colorbar; 
    else
    subplot 132; imagesc(real(rho_new(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi)./w)); axis image; colormap(jet); colorbar; 
    end
    title(['iteration : ',num2str(iter)]);
    subplot 133; plot(log(Fmat(:)));
end

tic;
while iter<iter_max

    r = zeros(Wi,Wi,L);
    for i4=1:L
        Rho(:,:,i4)=conv_fft2((y_old.*I(:,:,i4)),f,'same').*maskc;
    end
    for iiiii=1:L
        r(:,:,iiiii)=Target(:,:,iiiii)-Rho(:,:,iiiii);  
    end    
    
    GR2=sum(dot(Grho,Grho));
    Grholat=zeros(Wi);
    for ii=1:L
        Gr=-1*I(:,:,ii).*epsil_old.*conv_fft2(r(:,:,ii),f,'same');
        Grholat=Grholat+Gr;
    end
    gammarho=sum(dot(Grholat,(Grholat-Grho)))/GR2; 
    drho=Grholat+gammarho.*drho;
    gammaI=zeros(1,L); 
    GIlat=zeros(Wi,Wi,L);
    
    GIn2=sum(dot(GI(:,:,1),GI(:,:,1))); 
    GIlat(:,:,1)=-1*y_old.*smalli(:,:,1).*conv_fft2((2*r(:,:,1)-r(:,:,L)),f,'same'); 
    GIlat(isnan(GIlat))=0;
    gammaI(1,i3)=sum(dot(GIlat(:,:,1),(GIlat(:,:,1)-GI(:,:,1))))/GIn2;
    gammaI(isnan(gammaI))=0;
    GIn2=sum(dot(GI(:,:,L-1),GI(:,:,L-1))); 
    GIlat(:,:,L-1)=-1*y_old.*smalli(:,:,L-1).*conv_fft2((2*r(:,:,L-1)-r(:,:,L)),f,'same');
    GIlat(isnan(GIlat))=0;
    gammaI(1,i3)=sum(dot(GIlat(:,:,L-1),(GIlat(:,:,L-1)-GI(:,:,L-1))))/GIn2; 
    gammaI(isnan(gammaI))=0;
    for ii2=2:L-2
        GIn2=sum(dot(GI(:,:,ii2),GI(:,:,ii2))); 
        GIlat(:,:,ii2)=-1*y_old.*smalli(:,:,ii2).*conv_fft2((r(:,:,ii2)),f,'same'); 
        GIlat(isnan(GIlat))=0;
        gammaI(1,i3)=sum(dot(GIlat(:,:,ii2),(GIlat(:,:,ii2)-GI(:,:,ii2))))/GIn2; 
        gammaI(isnan(gammaI))=0;
    end
    for i3=1:L-1
        dI(:,:,i3)=GIlat(:,:,i3)+gammaI(1,i3).*dI(:,:,i3);   
        dI(isnan(dI))=0;
    end

    if use_gpu
        Grholat = gpuArray(Grholat);
        GIlat = gpuArray(GIlat);
    end
    ts=0; tm=0;
    for i1=1:L-1
        ss=sum(dot(conv_fft2(drho.*I(:,:,i1),f,'same'),r(:,:,i1)));
        ts=ts+ss;
    end
    for i1=1:L-1
        sm=sum(dot(conv_fft2(drho.*I(:,:,i1),f,'same'),conv_fft2(drho.*I(:,:,i1),f,'same')));
        tm=tm+sm;
    end
    alpha           =   ts/tm*lambda2; 
    epsil_new       =   epsil_old+drho.*alpha;
    rho_new         =   epsil_new.^2;
    ts=0; tm=0; beta=zeros(1,L);
    for i4=1:L-1
        ts=sum(dot(conv_fft2(y_old.*dI(:,:,i4),f,'same'),r(:,:,i4)));
        tm=sum(dot(conv_fft2(y_old.*dI(:,:,i4),f,'same'),conv_fft2(y_old.*dI(:,:,i4),f,'same')));
        if tm == 0
            beta(1,i4)=0;
        else
            beta(1,i4)=ts/tm*lambda3;
        end
    end
     
    for i3=1:L-1
        smallin(:,:,i3)=smalli(:,:,i3)+dI(:,:,i3).*beta(1,i3); 
        In(:,:,i3)=smallin(:,:,i3).^2;
    end
    
    I_L=(In(:,:,1)+In(:,:,L-1))/2;
    In(:,:,L)=I_L; 
    for i4=1:L
        Rhon(:,:,i4)=conv_fft2((rho_new.*In(:,:,i4)),f,'same').*maskc;   
    end
    Flat=0;
    for i6=1:L
        Fsub=norm(Target(:,:,i6)-Rhon(:,:,i6));
        Flat=Flat+Fsub;
    end
    for i7 = 1:L
        normIn(i7) = norm(In(:,:,i7));
    end
    
    Flat = Flat + lambda1 * L1(rho_new) + lambda4 * sum(normIn);
    
    Grho    =   Grholat; 
    GI      =   GIlat;
    rho_old =   rho_new;
    epsil_old = epsil_new;
    I       =   In; 
    smalli  =   smallin;
    
    t_new   =   (1+sqrt(1+4*t_old^2))/2;
    y_new   =   rho_new+((t_old-1)/t_new)*(rho_new-rho_old);
    y_old   =   y_new;
    t_old   =   t_new;

    if Flat > 2*max(Fmat)
        break;
    end
    F       =   Flat;
    Fmat(iter+1,1) = Flat;
    rho_ani(:,:,iter+2)=rho_new;  
    disp(iter);

    if dispfigure
        figure(2); 
        if interpolation
        subplot 132; imagesc(real(rho_new(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi)./w)); axis image; colormap(jet); colorbar; 
        else
        subplot 132; imagesc(real(rho_new(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi)./w)); axis image; colormap(jet); colorbar; 
        end
        title(['iteration : ',num2str(iter+1)]);
        subplot 133; plot(log(Fmat(:)));
    end
    iter=iter+1;
end
toc;

I_final = circshift(I,shift,3);
In_final = circshift(In,shift,3);
Target_final = circshift(Target,shift,3);
S = SS./w; %% figure, imshow(save_SS,[]);
save_SSn(:,:) = save_SS(:,:)/max2(save_SS); 
saveDir = 'Outpu\';
imwrite(uint16(save_SSn*65535),[saveDir 'SS_',num2str(filename),'.tif'],'Writemode','append');
    
if interpolation
save_rho = rho_ani(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi,:)./w; 
else
save_rho = rho_ani(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi,:)./w;
end

for i=1:iter+1
    save_rhon(:,:,i) = save_rho(:,:,i)/max3(save_rho); 
end 
for i=1:iter+1
    imwrite(uint16(save_rhon(:,:,i)*65535),[saveDir 'rho_ani_',num2str(filename),'.tif'],'Writemode','append');
end  

if interpolation
save_I = In_final(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi,:);
else
save_I = In_final(1+z_edgei/2:-z_edgei/2+Wi,1+z_edgei/2:-z_edgei/2+Wi,:)./w;
end

for i=1:L
    save_In(:,:,i) = save_I(:,:,i)/max3(save_I); 
end 
for i=1:L
    imwrite(uint16(save_In(:,:,i)*65535),[saveDir 'I_ani_',num2str(filename),'.tif'],'Writemode','append');
end 