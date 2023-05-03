function [H,S] = CBCA(Y,n,alpha)
%Y：signal mixture
%n: number of sourses
%H: separated channel with ambiguity
%S: signals obtained by Least Square Method corresponding to H
%alpha: depend on the type of modulation
MaxIter=1500; % the maximum iterations 
checkpoint=100; % the number of iterations, after that the program begins to check whether the convergence 
beta=2; % penalty factor %the  
[W,Rank]=whitening(Y,n); % whitening 
m=Rank;
z=W*Y; 
q=15*(2*m*n);                     
H=randn(m,q)+1i*randn(m,q);
B= H./repmat(sqrt(sum(abs(W'*H).^2)),m,1); %the matrix consists of one-dimension projection vector
[~,D]=eig(cov(Y'));
[v,j]=sort(diag(D)); 
vx=flipud([v,j]);
Lr=0.8*sqrt(sum(vx((n+1):size(Y,1),1)));
C=perimeter(B,z)-Lr; %obtain the vector of convex perimeter after projection
G=(1/sqrt(2))*(randn(q,n)+1i*randn(q,n)); %initialization of H after projection
R=eye(q)-B'*inv(B*B')*B; %the matrix in the constrain condition
func=@(G) norm(C-alpha*sum(abs(G),2))^2+beta*norm(R*G*[1;1;1])^2;%object function
Cost(1)=func(G);
t=1;
t0=1000;
Continue=1; %one of the flags of the loop
while Continue&&(t<=MaxIter)
t=t+1;
for j=1:q 
GradG(j,:)=-2*alpha*(C(j,1)-alpha*norm(G(j,:),1))*[(G(j,1)^(1/2)*(G(j,1)')^(-1/2))/2 (G(j,2)^(1/2)*(G(j,2)')^(-1/2))/2 (G(j,3)^(1/2)*(G(j,3)')^(-1/2))/2];
end
GradG_p=beta*R'*R*G; %part of gradient
NGrad=GradG+GradG_p; % gradient of whole function
if t<t0 % step-size
    mu=0.7*Cost(t-1)/(trace(NGrad'*NGrad));
else
    mu=0.01/log10(10+((t-t0)/1000));
end

c=0.5;
Cost(t)=Cost(t-1)+5;
while (c>1e-3)&&(Cost(t)>Cost(t-1)- 0.7*c*mu*(trace(NGrad'*NGrad)))
    G1=G-c*mu*NGrad;
    Cost(t)=func(G1);
    c=c*0.8;
    if c<1e-3&&(Cost(t)>Cost(t-1)- 0.7*mu*c*(trace(NGrad'*NGrad)))
        c=1;
        G1=G-c*mu*NGrad;
        Cost(t)=func(G1);
        break;
    end
end
G=G1;
%Check whrether the function reach its convergence
if t>checkpoint
    if trace(NGrad'*NGrad)<1
    Dist=3;
    Continue=std(Cost(t-Dist:t)/(size(Y,2)))>1e-3;
    end   
end
end
H=pinv(W)*pinv(B')*G;
S=pinv(H)*Y;
end
%%----------------------------------------------------------------------
function [W,Rank]=whitening(Y,n)
% Estimates the whitening matrix W and its rank m
M   =size(Y,1);
[Q,D]=eig(cov(Y'));
[v,i]=sort(diag(D)); vy=flipud([v,i]);
if M>n
    vn=mean(vy(n+1:M,1)); % Estimated noise power
else    
    vn=0;                 % Estimated noise power
end
Rank=min(M,n); % m =rank{W}.
D=diag(sqrt(max(vy(1:Rank,1)-vn,eps)));
W=pinv(D)*Q(:,vy(1:Rank,2))'; % whitening matrix
end
%%-------------------------------------------------------------------
function L=perimeter(B,y)
% Returns L, the vector of perimeters of projections of the observations
q=size(B,2);
for i=1:q
    y_=B(:,i)'*y;
    k = convhull(real(y_),imag(y_));
    L(i,1)=sum(abs(y_(k(2:length(k)))-y_(k(1:(length(k)-1)))));
end
end