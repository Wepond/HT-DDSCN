function [x,steps,erros,out] = CG(A,b,x0,eps)
r0 = b - A*x0;
p0 = r0;
if nargin == 3
    eps = 1.0e-10;
end
steps = 0;

set=90;
out=zeros(size(x0,1),set);
erros=zeros(set,1);
while 1
     if gather(abs(p0)) < gather(eps)
        break;
     end
    steps = steps + 1;
    if steps==set
        break;
    end
    a0 = r0'*r0/(p0'*A*p0);%����õ����Դ�һ����
    x1 = x0 + a0*p0;

    r1 = r0 -a0*A*p0;
  
    erros(steps)=gather(r0'*r0);
    b0 = r1'*r1/(r0'*r0);
    %�����r'* r��Ȼ������ܻ����õ����������ڼ���������û�б�Ҫ������±�����
    %������ˣ��ڴ��ϵĿ�����������
    p1 = r1 + b0*p0;

    %ֻ���õ�ǰ����������������Խ�ʡ�ڴ濪�������������һ�㣬�������ظ��ǵ�û��
    %�ı�����

    x0 = x1;
    r0 = r1;
    p0 = p1;
    out(:,steps)=gather(x0);
end
x =gather( x0);
end