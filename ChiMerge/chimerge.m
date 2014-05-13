%chimerge algorithm
%算法流程
%第一步，整理数据
%读入鸢尾花数据集，构造可以在其上使用ChiMerge的数据结构[属性值n,类别1实例数,类别2实例数,类别3实例数,chi2值]
%第二步，离散化
%使用ChiMerge方法对具有最小卡方值的相邻区间进行递归合并，直到满足最大区间数(max_interval)为6
%程序最终返回区间的分裂点及对应各类别实例数,chi2值

fid = fopen('./iris/iris.data');
 dataSet=textscan(fid,' %f32,%f32,%f32,%f32,%s\n');
 fclose(fid);
 
 label={'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};

 %依次计算四种属性值的离散化
for n=1:4
    iattr=n;%定义属性号
    attr_array=dataSet{iattr};%获得属性值数组（如所有关于属性1的数据）
    Length=length(attr_array);
    label_array=zeros(Length,1);

    for i=1:Length%获得类别数组（对应属性值数组）
        switch dataSet{5}{i}
            case label{1}
                label_array(i)=1;
             case label{2}
                label_array(i)=2;
             case label{3}
                label_array(i)=3;
        end
    end
    attr_label_array=[attr_array, label_array];%属性值-类别数组
    [V,I]=sort(attr_label_array(:,1));%按属性值排序
    attr_label_array_sort=[attr_label_array(I,1),attr_label_array(I,2)];

    hold=zeros(Length,5);%hold第一列为属性值，第二三四列为class1 2 3实例数,第五列为卡方值
    hold(1,1)=attr_label_array_sort(1,1);
    hold(1,attr_label_array_sort(1,2)+1)=1;
    s=1;
    for i=2:Length%统计各个属性值各类别实例数
        if hold(s,1)==attr_label_array_sort(i,1)
            hold(s,attr_label_array_sort(i,2)+1)=hold(s,attr_label_array_sort(i,2)+1)+1;
        else
            s=s+1;
            hold(s,1)=attr_label_array_sort(i,1);
            hold(s,attr_label_array_sort(i,2)+1)=1; 
        end
    end
    hold=hold(1:s,:);%截取有效数据

     class=3;%类别数
     max_interval=6;%最大区间数（算法收敛条件）
     current_interval=length(hold(:,1));
     while current_interval>max_interval
        for t=1:current_interval-1
            hold(t,5)=chi2calc(hold,t);
        end
        hold(t+1,5)=99;%last row
        [V,I]=min(hold(:,5));
        %merge rows
        for j=2:class+1
            hold(I,j)=hold(I,j)+hold(I+1,j);
        end
        %shifting
        h=hold(1:I,:);
        k=length(hold);
        if I+1<k
            h(I+1:k-1,:)=hold((I+2):k,:);
        end
        hold=h;
        current_interval=current_interval-1;
     end
     fprintf('attr%d class1 class2 class3 chi2\n',iattr);
     disp(hold);
end