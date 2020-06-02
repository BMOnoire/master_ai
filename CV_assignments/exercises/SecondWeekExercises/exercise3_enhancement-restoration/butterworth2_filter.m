%compute Butterworth Low-Pass Filter
[M,N]=size(image);


u0 = 50; % set cut off frequency
n=2;

u=0:(M-1);
v=0:(N-1);
idx=find(u>M/2);
u(idx)=u(idx)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);

for i = 1: M
    for j = 1:N
      %Apply a 2nd order Butterworth  
      UVw = double((U(i,j)*U(i,j) + V(i,j)*V(i,j))/(u0*u0));
      
      Hl(i,j) = 1/(1 + (UVw)^n);
    end
end    
% display
figure(3)
imshow(fftshift(Hl))


%compute Butterworth High-Pass Filter

u0 = 50; % set cut off frequency
n=2;

u=0:(M-1);
v=0:(N-1);
idx=find(u>M/2);
u(idx)=u(idx)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);

for i = 1: M
    for j = 1:N
      %Apply a 2nd order Butterworth  
      UVw = double((u0*u0)/(U(i,j)*U(i,j) + V(i,j)*V(i,j)));
      Hh(i,j) = 1/(1 + (UVw)^n);
    end
end    

% display
figure(3)
imshow(fftshift(Hh))