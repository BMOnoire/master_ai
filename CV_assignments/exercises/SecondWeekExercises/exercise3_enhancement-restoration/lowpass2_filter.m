%compute Ideal Low Pass Filter

u0 = 50; % set cut off frequency


u=0:(M-1);
v=0:(N-1);
idx=find(u>M/2);
u(idx)=u(idx)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);
D=sqrt(U.^2+V.^2);
Hl=double(D<=u0);

% display
figure(3);
imshow(fftshift(Hl));


%compute Ideal High Pass Filter

u0 = 50; % set cut off frequency


u=0:(M-1);
v=0:(N-1);
idx=find(u>M/2);
u(idx)=u(idx)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);
D=sqrt(U.^2+V.^2);
Hh=double(D>=u0);

% display
figure(3);
imshow(fftshift(Hh));