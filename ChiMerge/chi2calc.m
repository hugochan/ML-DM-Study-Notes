%calculating chisquare
%参数说明
%m=2（每次比较的区间数是2个）
%k=类别数量
%Aij=第i区间第j类的实例的数量
%Ri=第i区间的实例数量
%Cj=第j类的实例数量
%N=总的实例数量
%Eij= Aij的期望频率
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