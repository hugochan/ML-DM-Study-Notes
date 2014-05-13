%calculating chisquare
%����˵��
%m=2��ÿ�αȽϵ���������2����
%k=�������
%Aij=��i�����j���ʵ��������
%Ri=��i�����ʵ������
%Cj=��j���ʵ������
%N=�ܵ�ʵ������
%Eij= Aij������Ƶ��
%Eij=Ri*Cj/N

function f=chi2calc(s,t)
m=2; % the two intervals being compared
k=3; % # of classes
r=zeros(1,m);
c=zeros(1,k);
for i=t:t+m-1
    r(i+1-t)=sum(s(i,2:4));
    for j=1:k
        c(j)=c(j)+sum(s(i,j+1));
    end
end
sumd=0;
for i=t:t+m-1
    for j=1:k
        e=r(i+1-t)*c(j)/sum(c);
        if e==0
            e=0.1;
        end
        sumd=sumd+(s(i,j+1)-e)^2/e; 
    end
end
f=sumd;