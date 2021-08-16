function [y] = HDSCN(pn_train,tn_train,pn_test,hiddennodenum)
hiddenlayernum=4;
C=1000;

for i=1:hiddenlayernum
    if i==1
        a(:,:,i)=rand(size(pn_train,2),hiddennodenum)*2-1;
        b(:,:,i)=(repmat((rand(hiddennodenum,1)*2-1),1,size(pn_train,1)))';
        tempH(:,:,i)=tansig(pn_train*a(:,:,i)+b(:,:,i));
    end
    if i>1
        a1(:,:,i-1)=rand(size(tempH(:,:,i-1),2),hiddennodenum)*2-1;
        b1(:,:,i-1)=(repmat((rand(hiddennodenum,1)),1,size(tempH(:,:,i-1),1)))';
        tempH(:,:,i)=tansig(tempH(:,:,i-1)*a1(:,:,i-1)+b1(:,:,i-1));
    end
end
clear i;
%计算密集型节点输出
for k=1:hiddenlayernum
    for i=1:hiddennodenum-1
        xishu(1,i)=(mse(tempH(:,i,k)))/(1+sqrt(mse(tempH(:,i,k))));
    end
    clear i;
    for i=1:hiddennodenum-1
        q=i+1;
        for j=q:hiddennodenum
            tempH(:,i,k)=xishu(1,i)*tempH(:,i,k)+tempH(:,i,k);
        end
    end
    clear i;
    clear q;
    clear j;
    clear xishu;
end

%基于误差的PID
kp=0.1028;
ki=0.0095;
kd=0.113;
e=zeros(hiddennodenum,1);
beta1=zeros(hiddennodenum,1);
e(1,1)=sqrt(mse(tn_train));
error=tn_train;
temp=0;
temp1=0;
eck1=0;
eck2=0;
for i=1:hiddennodenum
    P=kp*(e(i,1)-eck1);
    I=ki*e(i,1);
    D=kd*(e(i,1)-2*eck1+eck2);
    u=P+I+D;
    beta1(i,1)=temp1+u;
    error=error-beta1(i,1)*tempH(:,i,1);
    e(i+1,1)=sqrt(mse(error));
    temp=error(i,1);
    temp1=beta1(i,1);
    clear u;
    eck2=eck1;
    eck1=e(i,1);
end
clear kp;
clear ki;
clear kd;
clear e;
clear error;
clear temp;
clear temp1;
clear eck1;
clear eck2;
x1=tempH(:,1:hiddennodenum,1)*beta1(1:hiddennodenum,1);

%共轭梯度
error=tn_train-x1;
clear x1;
u=tempH(:,:,2)'*tempH(:,:,2);
wq=tempH(:,:,2)'*error;
x0 = randn(size(u,2),1);
eps = 1.0e-1;
[beta2,steps,erros,out] = CG(eye(size(u,1))/C+u,wq,x0,eps);
clear steps;
clear erros;
clear out;
clear u;
clear wq;
clear x0;
clear eps;

%ADMM
xk=rand(1,size(tempH(:,:,3),2))*2-1;%产生初始的xk
zk=rand(1,size(tempH(:,:,3),2))*2-1;%产生初始的zk
uk=rand(1,size(tempH(:,:,3),2))*2-1;%产生初始的uk
error1=error-tempH(:,:,2)*beta2;
clear error;
maxin=300;%最大迭代次数300
eps=1e-10; %最小收敛误差
lambda=0.1;
kexi=0.0001;
for in=1:maxin %开始迭代计算
    xk=(error1'*tempH(:,:,3)+kexi*(zk-uk))*pinv(tempH(:,:,3)'*tempH(:,:,3)+eye(size(tempH(:,:,3)'*tempH(:,:,3)))*kexi);%论文16页公式3-9。 对于对称发方阵用inv 或pinv都可以。用inv时kexi为一个较为小的数防止矩阵奇异
    for jn=1:length(zk)
        if xk(jn)+uk(jn)>lambda
            zk(jn)=xk(jn)+uk(jn)-lambda;%论文17页公式3-10
        elseif abs(xk(jn)+uk(jn))<=lambda
            zk(jn)=0;
        elseif (xk(jn)+uk(jn))<-lambda
            zk(jn)=xk(jn)+uk(jn)+lambda;
        end
    end
    uk=uk+xk-zk;%论文16页公式3-10
    erros=norm(xk-zk);  
end
beta3=xk';%获得输出的权值
clear xk;
% 
%ADAM
error2=error1-tempH(:,:,3)*beta3;
clear error1;
x00=zeros(size(tempH(:,:,4),2),1);
[beta4, outAdam] = ADAM(x00,tempH(:,:,4),error2);
clear x00;
clear outAdam;
clear error2;
beta=[beta1;beta2;beta3;beta4];
y_train=tempH(:,:,1)*beta1+tempH(:,:,2)*beta2+tempH(:,:,3)*beta3+tempH(:,:,4)*beta4;
for i=1:hiddenlayernum
    if i==1
        b_test(:,:,i)=(repmat((b(i,:,i)'),1,size(pn_test,1)))';
        tempH_test(:,:,i)=tansig(pn_test*a(:,:,i)+b_test(:,:,i));
    end
    if i>1      
        b1_test(:,:,i-1)=(repmat((b1(i-1,:,i-1)'),1,size(tempH_test(:,:,i-1),1)))';
        tempH_test(:,:,i)=tansig(tempH_test(:,:,i-1)*a1(:,:,i-1)+b1_test(:,:,i-1));
    end
%     error1=error1+tempH(:,:,i)*beta(:,i);
end

clear i;
%计算密集型节点输出
for k=1:hiddenlayernum
    for i=1:hiddennodenum-1
        xishu(1,i)=(mse(tempH_test(:,i,k)))/(1+sqrt(mse(tempH_test(:,i,k))));
    end
    clear i;
    for i=1:hiddennodenum-1
        q=i+1;
        for j=q:hiddennodenum
            tempH_test(:,i,k)=xishu(1,i)*tempH_test(:,i,k)+tempH_test(:,i,k);
        end
    end
    clear i;
    clear q;
    clear j;
    clear xishu;
end

y=tempH_test(:,:,1)*beta1+tempH_test(:,:,2)*beta2+tempH_test(:,:,3)*beta3+tempH_test(:,:,4)*beta4;
end

