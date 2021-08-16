function [xAdam, outAdam] = ADAM(x0, A, b)
%Adam
[m n]=size(A);
e=20;
r1=0.95;
r2=0.95;
mu=1000;
m=10;
d=1e-10;
t=0;
s=zeros(n,1);
si=zeros(n,1);
r=zeros(n,1);
x=x0;
btA=A'*b;
for i=1:2000
    for k=1:n
        if x(k)>=0
            si(k)=1;
        else
            si(k)=-1;
        end
    end
    g=A'*(A*x)-btA+mu*si;
    t=t+1;
    s=r1*s+(1-r1)*g;
    r=r2*r+(1-r2)*g.*g;
    ss=s/(1-r1^t);
    rr=r/(1-r2^t);
    for k=1:n
%         g(k)=-e*ss(k)/(sqrt(d+rr(k)));
        g(k)=-e*ss(k)/(sqrt(rr(k))+d);
    end
    x=x+g;
    if mod(i,200)==0
        mu=max(mu/m,1e-3);
        e=e/20;
    end
end
xAdam=x;
outAdam=1/2*norm(A*x-b,2)^2+mu*norm(x,1);
end

